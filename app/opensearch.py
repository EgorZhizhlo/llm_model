from opensearchpy import OpenSearch
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config as config
import os


class OpenSearchHandler:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
            http_compress=True,
            use_ssl=False,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)
        self.llm = ChatOllama(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL)

    def _initialize_index(self, session_token):
        """Инициализация индекса OpenSearch для конкретной сессии."""
        index_name = f"{session_token}"
        mapping = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector_field": {"type": "knn_vector", "dimension": 384},
                    "text": {"type": "text"}
                }
            }
        }
        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=mapping)
        return index_name

    def add_documents(self, session_token, text=None):
        """Загрузка документов, разбиение на части и добавление в OpenSearch."""
        index_name = self._initialize_index(session_token)
        documents = [Document(page_content=text)]

        # Разбиение текста на части
        splitter = CharacterTextSplitter(
            chunk_size=config.TEXT_SPLIT_CHUNK_SIZE,
            chunk_overlap=config.TEXT_SPLIT_CHUNK_OVERLAP
        )
        texts = splitter.split_documents(documents)

        texts = [doc.page_content for doc in texts]
        vectors = self.embeddings.embed_documents(texts)

        for text, vector in zip(texts, vectors):
            doc = {"text": text, "vector_field": vector}
            self.client.index(index=index_name, body=doc)

    def view_split_text(self, session_token: str):
        """
        Извлекает и возвращает все нарезанные тексты, сохранённые в индексе OpenSearch для данной сессии.
        
        :param session_token: Токен сессии, для которой необходимо извлечь документы.
        :return: Список строк, представляющих нарезанный текст.
        """
        index_name = f"{session_token}"
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Индекс для сессии '{session_token}' не найден.")

        query = {"size": 10000, "query": {"match_all": {}}}  # Получаем до 10 000 документов за раз
        response = self.client.search(index=index_name, body=query)

        texts = []
        for hit in response["hits"]["hits"]:
            texts.append(hit["_source"]["text"])

        return texts

    def invoke_llm(self, 
                   session_token: str, 
                   question:str, 
                   base_prompt: str = None):
        """Интерфейс для выполнения запросов к LLM с использованием OpenSearch."""
        index_name = f"{session_token}"
        if not self.client.indices.exists(index=index_name):
            raise ValueError(f"Индекс для сессии '{session_token}' не найден. Добавьте данные перед запросом.")

        opensearch_vector_search = OpenSearchVectorSearch(
            f"http://{self.client.transport.hosts[0]['host']}:{self.client.transport.hosts[0]['port']}",
            index_name,
            self.embeddings,
        )
        retriever = opensearch_vector_search.as_retriever()
        docs = retriever.invoke(question)

        prompt_template = self._load_prompt() if base_prompt is None else base_prompt + '\nQuestion: {question}\nContext: {context}\nAnswer:'
        prompt = ChatPromptTemplate.from_messages([
            ("human", prompt_template)
        ])
        qa_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return qa_chain.invoke(question)

    @staticmethod
    def _format_docs(docs):
        """Форматирует документы в строку."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _load_prompt(file_path=None):
        """Загружает текст промта из файла."""
        file_path = file_path or config.PROMPT_FILE_PATH
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
