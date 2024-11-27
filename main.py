from opensearch import OpenSearchHandler
import config

if __name__ == "__main__":
    session_token = "user123"
    opensearch_handler = OpenSearchHandler()

    # Добавление документов в индекс
    opensearch_handler.add_documents(session_token, config.DOCUMENT_FILE_PATH)

    # Выполнение запроса
    question = "how fat are crewmates from amongus"
    answer = opensearch_handler.invoke_llm(session_token, question)
    print("Ответ:", answer)
