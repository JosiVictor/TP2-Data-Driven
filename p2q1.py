#Parte 2 - Questão 1

from fastapi import FastAPI
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

app = FastAPI(title="Chatbot com Fake LLM (LangChain)", version="1.0.0")

RESPOSTAS_PRONTAS = {
    "olá": "Olá! Eu sou um chatbot de teste. Como posso ajudar?",
    "horário": "Nosso atendimento funciona de segunda a sexta, das 9h às 18h.",
    "preço": "Os planos começam em R$ 99,90 por mês (valor fictício de teste).",
    "tchau": "Até logo! Foi um prazer conversar.",
}
RESPOSTA_PADRAO = "Desculpe, ainda não sei responder isso. Sou apenas um chatbot de teste."

prompt = PromptTemplate.from_template(
    "Você é um assistente educado e objetivo.\nPergunta do usuário: {pergunta}\nResposta:"
)


def escolher_resposta(pergunta: str) -> str:
    texto = pergunta.lower()
    for palavra, resposta in RESPOSTAS_PRONTAS.items():
        if palavra in texto:
            return resposta
    return RESPOSTA_PADRAO


class Entrada(BaseModel):
    pergunta: str = Field(..., min_length=1)


class Saida(BaseModel):
    pergunta: str
    resposta: str


@app.get("/")
def raiz():
    return {"mensagem": "Chatbot Fake LLM no ar. Use POST /chat com {'pergunta': '...'}"}


@app.post("/chat", response_model=Saida)
def chat(dados: Entrada):
    llm = FakeListLLM(responses=[escolher_resposta(dados.pergunta)])
    cadeia = prompt | llm | StrOutputParser()
    resposta = cadeia.invoke({"pergunta": dados.pergunta})
    return Saida(pergunta=dados.pergunta, resposta=resposta)
