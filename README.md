# TP2 - DR3: FastAPI e LangChain

Projeto da disciplina Desenvolvimento de Data-Driven Apps com Python.

| Arquivo | O que faz |
|---|---|
| `p1q1.py` | Gera texto com GPT-2 (FastAPI + transformers) |
| `p1q2.py` | Traduz inglês para francês com opus-mt-en-fr |
| `p2q1.py` | Chatbot com LangChain e Fake LLM |
| `p2q2.py` | Traduz inglês para francês com LangChain + Groq (ou OpenAI) |
| `p2q3.py` | Traduz inglês para alemão com LangChain + HuggingFace |

## Como rodar
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn p1q1:app --reload     # troque p1q1 pelo arquivo desejado
```
Abra http://127.0.0.1:8000/docs para testar.

Para `p2q2.py`, copie `.env.example` para `.env` e coloque sua `GROQ_API_KEY`.
Os diagramas estão em `diagramas/` e as análises no PDF.
