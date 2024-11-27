from opensearch import OpenSearchHandler
import config

if __name__ == "__main__":
    opensearch_handler = OpenSearchHandler()

    # Добавление документов в индекс
    opensearch_handler.add_documents(config.DOCUMENT_FILE_PATH)

    # Выполнение запроса
    question = "расскажи мне что входит в Работа с клиентами"
    answer = opensearch_handler.invoke_llm(question)
    print("Ответ:", answer)
