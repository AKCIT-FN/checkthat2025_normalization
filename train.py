from itertools import product
import zipfile
import random
import json
import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, disable_caching

disable_caching()

from sentence_transformers import SentenceTransformer
from nltk.tokenize import RegexpTokenizer
from more_itertools import chunked
from datasets import concatenate_datasets
from transformers import (
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    Seq2SeqTrainer,
    AutoTokenizer,
)
from loguru import logger
from tqdm import tqdm
import transformers
import pandas as pd
import numpy as np
import accelerate
import evaluate
import torch


logger.debug(f"Transformers: {transformers.__version__}") #4.47.1
logger.debug(f"Accelerate: {accelerate.__version__}") #1.2.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/"

logger.debug(f"Device: {DEVICE}")

from bert_score.utils import model2layers # Permite uso do modelo da NeuralMind no BERTScore

def tokenize_sample_data(data, tokenizer):
    features = tokenizer(
        [f"{config['prefix']}{p}" for p in data["post"]],
            truncation=True,
            padding=True,
            max_length=config["input_max_length"]
    )

    features["labels"] = tokenizer(
        data["normalized claim"], truncation=True, padding=True, max_length=config["output_max_length"]
    )["input_ids"]

    return features

def metrics_func(eval_arg, tokenizer):
    preds, labels = eval_arg
    # Replace -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Insert a line break (\n) in each sentence for scoring
    text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
    sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]

    # METEOR
    meteor_result = meteor_metric.compute(
        predictions=text_preds,
        references=text_labels,
    )

    # BLEU
    bleu_result = bleu_metric.compute(
        predictions=text_preds,
        references=text_labels
    )
    # ROUGE
    rouge_result = rouge_metric.compute(
        predictions=text_preds,
        references=text_labels
    )

    # Calcular BERTScore
    bertscore_result = bertscore_metric.compute(
        predictions=text_preds,
        references=text_labels,
        model_type="neuralmind/bert-base-portuguese-cased"
    )
    bert_precision = np.mean(bertscore_result["precision"])
    bert_recall   = np.mean(bertscore_result["recall"])
    bert_f1       = np.mean(bertscore_result["f1"])

    # Calcular similaridade média (cosine similarity) utilizando embeddings do sentence-transformers
    pred_embeddings = similarity_model.encode(text_preds)
    label_embeddings = similarity_model.encode(text_labels)
    pred_embeddings_t = torch.tensor(pred_embeddings)
    label_embeddings_t = torch.tensor(label_embeddings)
    cos_sim = torch.nn.functional.cosine_similarity(pred_embeddings_t, label_embeddings_t, dim=1)
    avg_similarity = cos_sim.mean().item()

    # Retornar todas as métricas em um único dicionário
    return {
        "meteor": meteor_result["meteor"],
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
        "bertscore_precision": bert_precision,
        "bertscore_recall": bert_recall,
        "bertscore_f1": bert_f1,
        "avg_similarity": avg_similarity
    }

def train_model(config: dict):
    experiment = "_".join(str(value).split("/")[-1] for value in config.values())
    
    predictions_dir = os.path.join("predictions", experiment)
    os.makedirs(predictions_dir, exist_ok=True)
    
    lang = config['data'].replace("_cleaned", "")
    pred_file = f"task2_{lang}"

    if os.path.exists(os.path.join(predictions_dir, pred_file + ".csv")):
        return

    generation_config = GenerationConfig(
        length_penalty=0.6,
        no_repeat_ngram_size=2,
        num_beams=config["n_beams"],
        max_length=config["output_max_length"]
    )

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(DATA_DIR, f"train-{config['data']}.csv"),
            "validation": os.path.join(DATA_DIR, f"dev-{config['data']}.csv"),
        }
    )

    logger.debug(ds)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    all_claims = ds["train"]["normalized claim"] + ds["validation"]["normalized claim"]

    if config["dynamic_output_max_length"]:
        max_tokens = max([len(tokenizer.tokenize(claim)) for claim in all_claims])
        config["output_max_length"] = min([max_tokens + 2, 128])

    output_dir = f"output/{experiment}"
    os.makedirs(output_dir, exist_ok=True)

    tokenized_ds = ds.map(
        lambda d: tokenize_sample_data(d, tokenizer),
        remove_columns=list(ds["train"].features.keys()),
        batched=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"],
        config=generation_config,
    ).to(DEVICE)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        num_train_epochs = config["num_train_epochs"],  # epochs
        learning_rate = config["learning_rate"],
        lr_scheduler_type = "linear",
        warmup_steps = 90,
        optim = config["optm"],
        weight_decay = 0.01,
        per_device_train_batch_size = config["batch_size"],
        per_device_eval_batch_size = config["batch_size"],
        gradient_accumulation_steps = 32 // config["batch_size"],
        predict_with_generate=True,
        seed = 42,
        save_strategy="best",
        eval_strategy="steps",
        metric_for_best_model=config["metric_for_best_model"],
        logging_steps = 10,
        push_to_hub = False,
        bf16=config["bf16"],
    )

    if config["concat_train_dev"]:
        train = concatenate_datasets(
            [tokenized_ds["train"], tokenized_ds["validation"]]
        )
    else:
        train = tokenized_ds["train"]

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        compute_metrics = lambda d: metrics_func(d, tokenizer),
        train_dataset = train,
        eval_dataset = tokenized_ds["validation"],
    )

    trainer.train()

    logger.debug("Training done")

    if hasattr(trainer.model, "module"):
        trainer.model.module.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "code_config.json"), "w") as f:
        json.dump(config, f)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    ## Avaliação

    test_data = pd.read_csv(
        os.path.join(DATA_DIR, f"test-{config['data']}.csv")
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(DEVICE)
    model.eval()

    predictions = []
    for input_texts in tqdm(list(chunked(test_data["post"].to_list(), config["batch_size"]))):
        inputs = tokenizer(
            input_texts, return_tensors="pt", truncation=True, padding=True, max_length=config["input_max_length"]
        )
        
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                generation_config=generation_config
            )

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [g.replace(config["prefix"], "") for g in generated_text]
        predictions.extend(generated_text)
    
    test_data["normalized claim"] = predictions

    logger.debug("Predição gerada com sucesso!")
    logger.debug(test_data)

    test_data.to_csv(
        os.path.join(predictions_dir, pred_file + ".csv"), index=False
    )

    test_data.to_csv(
        os.path.join(pred_file + ".csv"), index=False
    )

    with zipfile.ZipFile(os.path.join(predictions_dir, pred_file + ".zip"), 'w', zipfile.ZIP_DEFLATED) as myzipf:
        myzipf.write(pred_file + ".csv")

    #from avaliador import AvaliadorDeClaims

    #avaliador = AvaliadorDeClaims(
    #    caminho_csv=os.path.join(output_dir, "avaliador_input.csv"),
    #    coluna_post="post",
    #    coluna_predicao="normalized claim",
    #    coluna_referencia="generated claim"
    #)

    #avaliador.avaliar_todas(
    #    usar_meteor=True, usar_bertscore=True, usar_similaridade=True
    #)

    #logger.info(avaliador.resumo_das_metricas())
    #avaliador.exportar_resultados(f"{output_dir}/report.tsv")


if __name__ == "__main__":
    configs = {
        "model_name": [
            'unicamp-dl/ptt5-v2-large',
            'unicamp-dl/monoptt5-large',
            'google/byt5-large',
            'google/mt5-large'
        ],        
        "data": ["por_cleaned"],
        "batch_size": [4],
        "input_max_length": [1024],
        "output_max_length": [128],
        "dynamic_output_max_length": [True],
        "num_train_epochs": [10],
        "optm": ["adamw_torch", "adafactor"],
        "concat_train_dev": [True],
        "prefix": ["", "summarize: "],
        "n_beams": [15],
        "bf16": [False],
        "metric_for_best_model": ["eval_meteor", "eval_bertscore_f1"],
        "learning_rate": [0.003]
    }

    available_experiments = []
    for experiment in product(*configs.values()):
        available_experiments.append(experiment)

    logger.debug(f"Grid-Search: {len(available_experiments)} runs")

    model2layers["neuralmind/bert-base-portuguese-cased"] = model2layers["bert-base-uncased"]
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    meteor_metric = evaluate.load("meteor")
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    bertscore_metric = evaluate.load("bertscore")


    random.shuffle(available_experiments)
    for experiment in available_experiments:
        config = dict(zip(configs.keys(), experiment))

        logger.debug(config)
        train_model(config)