import os

from dotenv import load_dotenv

load_dotenv()


def load_model(model_name: str):
    if model_name.startswith("gpt-"):
        from langchain_openai.chat_models import ChatOpenAI

        model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model_name)
    else:
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
