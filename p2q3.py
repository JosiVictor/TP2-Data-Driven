#Parte 2 - Questão 3

import re
from typing import Any, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI(title="Tradutor EN -> DE (LangChain + HuggingFace)", version="1.0.0")

NOME_MODELO = "Helsinki-NLP/opus-mt-en-de"

tokenizador = AutoTokenizer.from_pretrained(NOME_MODELO)
modelo = AutoModelForSeq2SeqLM.from_pretrained(NOME_MODELO)
modelo.eval()


def traduzir_frase(texto: str) -> str:
    entradas = tokenizador([texto], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        saida = modelo.generate(**entradas, max_new_tokens=512)
    return tokenizador.batch_decode(saida, skip_special_tokens=True)[0]

class TradutorHuggingFace(LLM):
    @property
    def _llm_type(self) -> str:
        return "opus-mt-en-de"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> str:
        return traduzir_frase(prompt)


llm = TradutorHuggingFace()

cadeia = llm | StrOutputParser()


class Entrada(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto em inglês")


class Saida(BaseModel):
    original: str
    traducao: str


def dividir_em_frases(texto: str) -> list[str]:
    frases = re.split(r"(?<=[.!?])\s+", texto.strip())
    return [f.strip() for f in frases if f.strip()]


@app.get("/")
def raiz():
    return {"mensagem": "Tradutor LangChain+HuggingFace (EN->DE) no ar. Use POST /traduzir"}


@app.post("/traduzir", response_model=Saida)
def traduzir(dados: Entrada):
    frases = dividir_em_frases(dados.texto)
    if not frases:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        traducoes = cadeia.batch(frases)  # LangChain chama o modelo para cada frase
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {erro}")
    return Saida(original=dados.texto, traducao=" ".join(t.strip() for t in traducoes))
