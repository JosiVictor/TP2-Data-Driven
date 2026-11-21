#Parte 1 - Questão 2

import re
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI(title="Tradutor EN -> FR (HuggingFace)", version="1.0.0")

NOME_MODELO = "Helsinki-NLP/opus-mt-en-fr"
tokenizador = AutoTokenizer.from_pretrained(NOME_MODELO)
modelo = AutoModelForSeq2SeqLM.from_pretrained(NOME_MODELO)
modelo.eval()


class Entrada(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto em inglês")


class Saida(BaseModel):
    original: str
    traducao: str


def dividir_em_frases(texto: str) -> list[str]:
    """Quebra o texto em frases. Assim textos longos não estouram o limite do modelo."""
    frases = re.split(r"(?<=[.!?])\s+", texto.strip())
    return [f.strip() for f in frases if f.strip()]


def traduzir_frases(frases: list[str], tamanho_lote: int = 8) -> list[str]:
    """Traduz as frases em pequenos lotes."""
    resultado = []
    for i in range(0, len(frases), tamanho_lote):
        lote = frases[i:i + tamanho_lote]
        entradas = tokenizador(lote, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            saida = modelo.generate(**entradas, max_new_tokens=512)
        resultado += tokenizador.batch_decode(saida, skip_special_tokens=True)
    return resultado


@app.get("/")
def raiz():
    return {"mensagem": "Tradutor EN->FR no ar. Use POST /traduzir com {'texto': '...'}"}


@app.post("/traduzir", response_model=Saida)
def traduzir(dados: Entrada):
    frases = dividir_em_frases(dados.texto)
    if not frases:
        raise HTTPException(status_code=422, detail="Texto vazio.")
    try:
        traducoes = traduzir_frases(frases)
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro na tradução: {erro}")
    return Saida(original=dados.texto, traducao=" ".join(traducoes))
