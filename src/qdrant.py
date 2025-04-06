import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant

load_dotenv()


class myQdrant:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def create(self):
        return qdrant_vectorstore(self.documents, self.embeddings)


def qdrant_vectorstore(documents, embeddings):
    return Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=os.getenv("QDRANT_DB_URL"),
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
