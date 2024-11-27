from opensearch import OpenSearchHandler
import config as config
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import requests


app = FastAPI()


class AddDocumnents(BaseModel):
    session_token: str
    text: str


class LlmInvokes(BaseModel):
    session_token: str
    question: str


@app.post("/add-document")
async def add_documents(
    form: AddDocumnents = Depends()
):
    opensearch_handler = OpenSearchHandler()
    try:
        opensearch_handler.add_documents(form.session_token, form.text)
        return 200
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/invoke_llm")
async def invoke_llm(
    form: LlmInvokes = Depends()
):
    opensearch_handler = OpenSearchHandler()
    session_token = form.session_token
    question = form.question
    try:
        return {"message": opensearch_handler.invoke_llm(session_token, question)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
