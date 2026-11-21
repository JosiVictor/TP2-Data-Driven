#Parte 1 - Questão 1

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, set_seed

app = FastAPI(title="Gerador de texto com GPT-2", version="1.0.0")
gerador = pipeline("text-generation", model="gpt2")


class Entrada(BaseModel):
    """Formato do JSON que a API recebe."""
    frase: str = Field(..., min_length=1, description="Frase inicial, de preferência em inglês")
    max_tokens: int = Field(50, ge=5, le=200, description="Tamanho máximo do texto novo")


class Saida(BaseModel):
    """Formato do JSON que a API devolve."""
    entrada: str
    texto_gerado: str


@app.get("/")
def raiz():
    return {"mensagem": "API GPT-2 no ar. Use POST /gerar com {'frase': '...'}"}


@app.post("/gerar", response_model=Saida)
def gerar(dados: Entrada):
    try:
        set_seed(42)  # deixa o resultado repetível
        resultado = gerador(
            dados.frase,
            max_new_tokens=dados.max_tokens,
            num_return_sequences=1,
            do_sample=True,
            truncation=True,
            pad_token_id=gerador.tokenizer.eos_token_id,
        )
    except Exception as erro:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar texto: {erro}")

    return Saida(entrada=dados.frase, texto_gerado=resultado[0]["generated_text"])
