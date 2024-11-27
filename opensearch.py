from opensearchpy import OpenSearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config


class OpenSearchHandler:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{"host": config.OPENSEARCH_HOST, "port": config.OPENSEARCH_PORT}],
            http_compress=True,
            use_ssl=config.OPENSEARCH_USE_SSL,
        )
        self.index_name = config.OPENSEARCH_INDEX_NAME
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)
        self.llm = ChatOllama(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL)
        self._initialize_index()

    def _initialize_index(self):
        """Инициализация индекса OpenSearch с KNN."""
        mapping = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector_field": {"type": "knn_vector", "dimension": 384},
                    "text": {"type": "text"}
                }
            }
        }
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=mapping)

    def add_documents(self, file_path=None):
        """Загрузка документов, разбиение на части и добавление в OpenSearch."""
        file_path = file_path or config.DOCUMENT_FILE_PATH
        loader = TextLoader(file_path, encoding='UTF-8')
        documents = loader.load()

        splitter = CharacterTextSplitter(
            chunk_size=config.TEXT_SPLIT_CHUNK_SIZE,
            chunk_overlap=config.TEXT_SPLIT_CHUNK_OVERLAP
        )
        texts = splitter.split_documents(documents)

        texts = [doc.page_content for doc in texts]
        vectors = self.embeddings.embed_documents(texts)

        for text, vector in zip(texts, vectors):
            doc = {"text": text, "vector_field": vector}
            self.client.index(index=self.index_name, body=doc)

    def invoke_llm(self, question):
        """Интерфейс для выполнения запросов к LLM с использованием OpenSearch."""
        opensearch_vector_search = OpenSearchVectorSearch(
            f"http://{self.client.transport.hosts[0]['host']}:{self.client.transport.hosts[0]['port']}",
            self.index_name,
            self.embeddings,
        )
        retriever = opensearch_vector_search.as_retriever()
        docs = retriever.invoke(question)

        prompt_template = self._load_prompt()
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
