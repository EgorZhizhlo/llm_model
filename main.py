from opensearch import OpenSearchHandler
import config
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import requests


app = FastAPI()


class AddDocumnents(BaseModel):
    session_token: str
    file_url: str


class LlmInvokes(BaseModel):
    session_token: str
    question: str


@app.post("/add-document")
async def add_documents(
    form: AddDocumnents = Depends()
):
    opensearch_handler = OpenSearchHandler()
    try:
        response = requests.get(form.file_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, detail=f"Ошибка при загрузке файла: статус-код {response.status_code}")
        doc_text = response.content
        opensearch_handler.add_documents(form.session_token, doc_text)
        return doc_text
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/invoke_llm")
async def invoke_llm(
    form: LlmInvokes = Depends()
):
    opensearch_handler = OpenSearchHandler()
    session_token = form.session_token
    question = form.question
    return {"message": opensearch_handler.invoke_llm(session_token, question)}
