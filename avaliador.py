
from statistics import mean


from sentence_transformers import SentenceTransformer
from nltk.tokenize import RegexpTokenizer
from bert_score.utils import model2layers
import pandas as pd
import numpy as np
import evaluate
import torch

# Permite uso do modelo da NeuralMind no BERTScore
model2layers["neuralmind/bert-base-portuguese-cased"] = model2layers["bert-base-uncased"]

class AvaliadorDeClaims:
    def __init__(
        self,
        caminho_csv,
        coluna_post="Post",
        coluna_predicao="Claim Gerada",
        coluna_referencia="Claim Real"
    ):
        self.caminho_csv = caminho_csv
        self.coluna_post = coluna_post
        self.coluna_predicao = coluna_predicao
        self.coluna_referencia = coluna_referencia

        self.tokenizador_frases = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
        self.modelo_bert_score = "neuralmind/bert-base-portuguese-cased"
        self.modelo_similaridade = SentenceTransformer("BAAI/bge-m3")

        self.metricas_meteor = evaluate.load("meteor")
        self.metricas_bertscore = evaluate.load("bertscore")

        self._carregar_dados()
        self.resultados_metricas = None

    def _carregar_dados(self):
        self.dados = pd.read_csv(self.caminho_csv)
        self.posts = self.dados[self.coluna_post].astype(str).tolist() if self.coluna_post in self.dados.columns else [""] * len(self.dados)
        self.predicoes = self.dados[self.coluna_predicao].astype(str).tolist()
        self.referencias = self.dados[self.coluna_referencia].astype(str).tolist()

    def _preparar_para_meteor(self, texto):
        if not texto.endswith(("!", "！", "?", "？", "。")):
            texto += "。"
        frases = self.tokenizador_frases.tokenize(texto)
        return "\n".join(np.char.strip(frases))

    def _avaliar_meteor(self):
        preds = [self._preparar_para_meteor(p) for p in self.predicoes]
        refs = [self._preparar_para_meteor(r) for r in self.referencias]
        return [
            self.metricas_meteor.compute(predictions=[p], references=[r])["meteor"]
            for p, r in zip(preds, refs)
        ]

    def _avaliar_bertscore(self):
        return self.metricas_bertscore.compute(
            predictions=self.predicoes,
            references=self.referencias,
            model_type=self.modelo_bert_score,
            lang="pt"
        )["f1"]

    def _avaliar_similaridade(self):
        vet_pred = self.modelo_similaridade.encode(self.predicoes, convert_to_tensor=True)
        vet_ref = self.modelo_similaridade.encode(self.referencias, convert_to_tensor=True)
        return torch.nn.functional.cosine_similarity(vet_pred, vet_ref).tolist()

    def _calcular_tamanho_em_palavras(self, lista_textos):
        return [len(texto.split()) for texto in lista_textos]

    def avaliar_todas(
        self,
        usar_meteor=True,
        usar_bertscore=True,
        usar_similaridade=True,
        calcular_tamanhos=True
    ):
        resultados = {
            "Post": self.posts,
            "Claim Gerada": self.predicoes,
            "Claim Real": self.referencias
        }

        if calcular_tamanhos:
            print("📏 Calculando tamanhos das claims...")
            resultados["Tamanho Claim Gerada"] = self._calcular_tamanho_em_palavras(self.predicoes)
            resultados["Tamanho Claim Real"] = self._calcular_tamanho_em_palavras(self.referencias)

        if usar_meteor:
            print("Avaliando com METEOR...")
            resultados["METEOR Individual"] = self._avaliar_meteor()

        if usar_bertscore:
            print(" Avaliando com BERTScore...")
            resultados["BERTScore F1"] = self._avaliar_bertscore()

        if usar_similaridade:
            print(" Avaliando com Similaridade Coseno...")
            resultados["Similaridade Coseno"] = self._avaliar_similaridade()

        if self.resultados_metricas is None:
            self.resultados_metricas = pd.DataFrame(resultados)
        else:
            for chave, valores in resultados.items():
                self.resultados_metricas[chave] = valores
        return self.resultados_metricas

    def resumo_das_metricas(self):
        if self.resultados_metricas is None:
            raise ValueError("Execute .avaliar_todas() antes de solicitar o resumo.")

        resumo = {}
        if "METEOR Individual" in self.resultados_metricas:
            resumo["Média METEOR"] = mean(self.resultados_metricas["METEOR Individual"])
        if "BERTScore F1" in self.resultados_metricas:
            resumo["Média BERTScore F1"] = mean(self.resultados_metricas["BERTScore F1"])
        if "Similaridade Coseno" in self.resultados_metricas:
            resumo["Média Similaridade Coseno"] = mean(self.resultados_metricas["Similaridade Coseno"])
        if "Tamanho Claim Gerada" in self.resultados_metricas:
            resumo["Média Tamanho Claim Gerada"] = mean(self.resultados_metricas["Tamanho Claim Gerada"])
        if "Tamanho Claim Real" in self.resultados_metricas:
            resumo["Média Tamanho Claim Real"] = mean(self.resultados_metricas["Tamanho Claim Real"])

        return resumo

    def exportar_resultados(self, caminho_saida="resultados_avaliacao.tsv"):
        if self.resultados_metricas is None:
            raise ValueError("Execute .avaliar_todas() antes de exportar os resultados.")
        self.resultados_metricas.to_csv(caminho_saida, index=False, sep="\t")
        print(f"Resultados exportados para: {caminho_saida}")
