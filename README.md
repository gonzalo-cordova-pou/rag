# Building RAGs with LangChain

## Content


- [Notebooks](#notebooks)
- [Models](#models)
    - [OpenAI](#openai)
    - [Ollama](#ollama)
- [Embeddings](#embeddings)
    - [OpenAI](#openai-1)
    - [Ollama](#ollama-1)
- [Vectore Stores](#vectore-stores)
    - [Local In-Memory](#local-in-memory)
    - [Qdrant Cloud](#qdrant-cloud)
- [References](#references)


## Notebooks

- **Creating a RAG with an In-Memory Vector Database** ([rag_pdf_local.ipynb](./rag_pdf_local.ipynb))
    - This notebook shows how to build a RAG with a local in-memory vector store.
    - It also shows how to run OpenAI and Ollama models with LangChain.
    - Content:
        - Running an LLM
        - Loading PDF Documents
        - Prompt Engineering
        - Creating a Local In-Memory Vector Database
        - Testing End-to-End RAG
- **Creating a RAG using Qdrant as Cloud Vector Database** ([rag_pdf_qdrant.ipynb](./rag_pdf_qdrant.ipynb))
    - This notebook shows how to build a RAG with a Qdrant vector store.
    - It also shows how to use different vector search methods (similarity, MMR, ...).
    - Content:
        - Creating a vector database with Qdrant
        - Information Retrieval (Search)
        - Use an LLM to ask to the retrieved information

### Models


#### OpenAI

```python
from langchain_openai.chat_models import ChatOpenAI

MODEL = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model_name)
```

#### Ollama

```python
from langchain_community.llms import Ollama

model = Ollama(model=model_name)
```





### Embeddings

#### OpenAI

```python
from langchain_openai.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
```

Or specify a model

```python
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"), model=embedding_model
)
```

#### Ollama

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model=model_name)
```

### Vectore Stores

For this example, we will use a PDF document loader to load a document and split it into pages.

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = Path(".") / "database" / "pdfs" / "doc2.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
```

**Vector Stores**

Information is split into chunks and stored in a vector store. This allows for efficient similarity search.

**Retrievers**

Retrievers (`VectorStoreRetriever` class) are a wrapper around vector stores that allow for easy querying. Retrievers with LangChain have these options:
- `search_type`: Type of search to perform. Options are “similarity” (default), “mmr”, or “similarity_score_threshold”
- `search_kwargs`: Additional arguments to pass to the search function
    - `k`: Amount of documents to return (default 4)
    - `score_threshold`: Minimum relevance threshold (default 0)
    - `fetch_k`: Amount of documents to pass to MMR algorithm (default 20)
    - `lambda_mult`: Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum (default 0.5)
    - `filter`: Filter by document metadata

#### Local In-Memory

We can store the documents in memory and perform similarity search.

```python
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(
    pages,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()
retriever.invoke("Information to retrieve")
```

#### Qdrant Cloud

- [Qdrant Website](https://qdrant.tech/)

```python
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
        documents, embeddings, url="<qdrant-url>", api_key="<qdrant-api-key>", 
        collection_name="pdfs",
    )

vectorstore = Qdrant.from_texts(
    texts, embeddings, url="<qdrant-url>", api_key="<qdrant-api-key>", collection_name="texts"
)
```

**Similarity Search**

```python
query = "What did the president say about Ketanji Brown Jackson"
found_docs = vectorstore.similarity_search(query)
print(found_docs[0].page_content)
```

**Similarity Search with Score**

```python
query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search_with_score(query)
document, score = found_docs[0]
print(document.page_content)
print(f"\nScore: {score}")
```

**Metadata filtering**

```python
from qdrant_client.http import models as rest

query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search_with_score(query, filter=rest.Filter(...))

# Example filter
page_filter = rest.Filter(
    must=[
        rest.FieldCondition(
            key="metadata.page",
            match=rest.MatchValue(value=3)
        )
    ]
)
```


**Maximum marginal relevance search (MMR)**

Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.

```python
query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.max_marginal_relevance_search(query, k=2, fetch_k=10)
```

**Qdrant as a retriever**

```python
retriever = qdrant.as_retriever(search_type="mmr") # “similarity” (default), “mmr”, or “similarity_score_threshold”
query = "What did the president say about Ketanji Brown Jackson"
retriever.invoke(query)[0]
```

### References

- [Basic In-Memory PDF Tutorial: Ollama + LangChain](https://www.youtube.com/watch?v=HRvyei7vFSM&t)
- [Youtube-RAG Tutorial](https://www.youtube.com/watch?v=BrsocJb-fAo)
- [LangChain Docs - Qdrant Integration Docs](https://python.langchain.com/v0.1/docs/integrations/vectorstores/qdrant/)
- [Qdrant Docs - LangChain Framework](https://qdrant.tech/documentation/frameworks/langchain/)
- [Qdrant Youtube - Chatbot with RAG, using LangChain, OpenAI, and Groq](https://www.youtube.com/watch?v=O60-KuZZeQA)