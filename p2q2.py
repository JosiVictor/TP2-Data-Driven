#Parte 2 - Questão 2

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import truststore
truststore.inject_into_ssl()

load_dotenv()  # lê o arquivo .env

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Defina GROQ_API_KEY no arquivo .env (veja o .env.example).")

app = FastAPI(title="Tradutor EN -> FR (LangChain + Groq)", version="1.0.0")

MODELO = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(model=MODELO, temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional translator. Translate the user's English text "
            "into French. Reply with the French translation only, without comments.",
        ),
        ("human", "{texto}"),
    ]
)

cadeia = prompt | llm | StrOutputParser()


class Entrada(BaseModel):
    texto: str = Field(..., min_length=1, description="Texto em inglês")


class Saida(BaseModel):
    original: str
    traducao: str


@app.get("/")
def raiz():
    return {"mensagem": "Tradutor LangChain+Groq no ar. Use POST /traduzir"}


@app.post("/traduzir", response_model=Saida)
async def traduzir(dados: Entrada):
    try:
        traducao = await cadeia.ainvoke({"texto": dados.texto}) 
    except Exception as erro:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar o provedor de IA: {erro}")
    return Saida(original=dados.texto, traducao=traducao.strip())
