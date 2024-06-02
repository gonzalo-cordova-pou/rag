import os
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch, Qdrant
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_model(model_name: str):
    if model_name.startswith("gpt-"):
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model_name)
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.llms import Ollama

        model = Ollama(model=model_name)

    return model


def load_embedding_model(source: str, name: str = None):
    if source == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings

        if name is None:
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"), model=name
            )
    elif source == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model=name)
    else:
        raise ValueError("Invalid source")

    return embeddings


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
