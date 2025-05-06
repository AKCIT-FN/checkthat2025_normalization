import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, disable_caching

disable_caching()

from nltk.tokenize import RegexpTokenizer

from sentence_transformers import SentenceTransformer

from transformers import (
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    Seq2SeqTrainer,
    AutoTokenizer, 
)
from loguru import logger
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

model2layers["neuralmind/bert-base-portuguese-cased"] = model2layers["bert-base-uncased"]
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

meteor_metric = evaluate.load("meteor")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def tokenize_sample_data(data):
    input_feature = tokenizer(
        [f"{config['prefix']}{p}" for p in data["post"]], truncation=True, max_length=config["input_max_length"]
    )
    
    label = tokenizer(
        data["normalized claim"], truncation=True, max_length=config["output_max_length"]
    )
    
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label["input_ids"],
    }

def tokenize_sentence(arg):
    encoded_arg = tokenizer(arg)
    
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

def metrics_func(eval_arg):
    preds, labels = eval_arg
    # Replace -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Convert id tokens to text
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


if torch.cuda.is_available():
    logger.debug("GPU is enabled.")
    logger.debug("device count: {}, current device: {}".format(torch.cuda.device_count(), torch.cuda.current_device()))
else:
    logger.debug("GPU is not enabled.")


if __name__ == "__main__":
    config = {
        "model_name": 'unicamp-dl/ptt5-v2-large',
        "data": "por_cleaned",
        "batch_size": 8,
        "input_max_length": 400,
        "output_max_length": 128,
        "num_train_epochs": 10,
        "optm": "adamw_torch",
        #"optm": "adafactor",
        "prefix": "summarize: ",
        #"prefix": "",
        "n_beams": 15,
        "bf16": False
    }

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(DATA_DIR, f"train-{config['data']}.csv"),
            "validation": os.path.join(DATA_DIR, f"dev-{config['data']}.csv")
        }
    )
    
    logger.debug(ds)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    all_claims = ds["train"]["normalized claim"] + ds["validation"]["normalized claim"]
    max_tokens = max([len(tokenizer.tokenize(claim)) for claim in all_claims])
    config["output_max_length"] = min([max_tokens + 2, 128])

    logger.debug(config)

    experiment = "_".join(str(value).split("/")[-1] for value in config.values())
    output_dir = f"output/{experiment}"
    os.makedirs(output_dir, exist_ok=True)


    tokenized_ds = ds.map(
        tokenize_sample_data,
        remove_columns=list(ds["train"].features.keys()),
        batched=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model_name"],
        config=GenerationConfig(
            length_penalty=0.6,
            no_repeat_ngram_size=2,
            num_beams=config["n_beams"],
            max_length=config["output_max_length"]
        ),
    ).to(DEVICE)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir = output_dir,
        num_train_epochs = config["num_train_epochs"],  # epochs
        learning_rate = 0.003,
        lr_scheduler_type = "linear",
        warmup_steps = 90,
        optim = config["optm"],
        weight_decay = 0.01,
        per_device_train_batch_size = config["batch_size"],
        per_device_eval_batch_size = config["batch_size"],
        gradient_accumulation_steps = 32 // config["batch_size"],
        predict_with_generate=True,
        seed = 42,
        save_strategy="no",
        load_best_model_at_end=True,
        logging_steps = 10,
        push_to_hub = False,
        bf16=config["bf16"],
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        data_collator = data_collator,
        compute_metrics = metrics_func,
        train_dataset = tokenized_ds["train"],
        eval_dataset = tokenized_ds["validation"],
        tokenizer = tokenizer
    )

    trainer.train()


    if hasattr(trainer.model, "module"):
        trainer.model.module.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)

    logger.debug("Training done")

    with open(os.path.join(output_dir, "code_config.json"), "w") as f:
        json.dump(config, f)

    ## Avaliação

    dev_data = pd.read_csv(
        os.path.join(DATA_DIR, f"dev-{config['data']}.csv")
    )

    predictions = trainer.predict(tokenized_ds["validation"]).predictions
    dev_data["generated claim"] = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    logger.debug(dev_data.columns)

    dev_data.to_csv(
        os.path.join(output_dir, "avaliador_input.csv"), index=False
    )
    logger.debug("Arquivo CSV 'avaliador_input.csv' gerado com sucesso!")

    from avaliador import AvaliadorDeClaims

    avaliador = AvaliadorDeClaims(
        caminho_csv=os.path.join(output_dir, "avaliador_input.csv"),  
        coluna_post="post",
        coluna_predicao="normalized claim",
        coluna_referencia="generated claim"
    )

    avaliador.avaliar_todas(
        usar_meteor=True, usar_bertscore=True, usar_similaridade=True
    )

    logger.info(avaliador.resumo_das_metricas())
    avaliador.exportar_resultados(f"{output_dir}/report.tsv")
