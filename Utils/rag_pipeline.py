import os
import tempfile
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import streamlit as st

def _save_uploaded_file_to_temp(uploaded_file):
    suffix = Path(getattr(uploaded_file, 'name', '')).suffix or ''
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def _get_embeddings():
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIEmbeddings(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_EMBEDDING_ENDPOINT"],
            api_version=os.environ.get("AZURE_EMBEDDING_VERSION"),
            model=os.environ.get("AZURE_EMBEDDING_MODEL")
        )
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    raise ValueError(
        "No embedding key configured. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for Azure, or OPENAI_API_KEY for OpenAI."
    )


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = _get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store


# def retrieve_docs(vector_store, query):
#     docs = vector_store.similarity_search(query, k=3)
#     return "\n".join([doc.page_content for doc in docs])

def load_txt(txt_path):
    if hasattr(txt_path, 'getvalue'):
        txt_path = _save_uploaded_file_to_temp(txt_path)

    st.write(f"1. Loading document from {txt_path}")
    try:
        loader = TextLoader(txt_path, encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading text file: {e}")
        return None

    print(f"Loaded {len(documents)} documents from {txt_path}")
    print("2. Splitting the document into chunks")
    return documents

def load_pdf(pdf_path):
    if hasattr(pdf_path, 'getvalue'):
        pdf_path = _save_uploaded_file_to_temp(pdf_path)

    print(f"1. Loading document from {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    print(f"Loaded {len(documents)} documents from {pdf_path}")
    print("2. Splitting the document into chunks")
    return documents

def create_vector_store(documents):
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
  )
  chunks = splitter.split_documents(documents)
  print(f"Split into {len(chunks)} chunks")

  print(f"3. Creating embedding and storing in FAISS Vector Store")

  embeddings = _get_embeddings()
  vector_store = FAISS.from_documents(chunks, embeddings)
  print("Embeddings created and stored in FAISS vector store")

  return vector_store


# def retrieve_docs(vector_store, query):
#     docs = vector_store.similarity_search(query, k=3)
#     return "\n".join([doc.page_content for doc in docs])


def retrieve_docs(vector_store, query):
    docs = vector_store.similarity_search(query, k=3)
    return docs   # return full docs, NOT joined text



def rag(current_query: str, chat_history:str, vector_store):
  """
  Function to generate rag response
  """
  retrieved_docs_content = ""
  if vector_store:
    similarity_threshold = 0.0.6 # Define a similarity threshold
    # Use similarity_search_with_score to get documents and their scores
    retrieved_docs_with_score = vector_store.similarity_search_with_score(current_query)

    #Return the top document based on the search
    # Filter documents based on the similarity threshold
    filtered_docs = [doc for doc, score in retrieved_docs_with_score if score >= similarity_threshold]

    if filtered_docs:
      doc_contents = []
      for i, doc in enumerate(filtered_docs):
        source = doc.metadata.get('source', 'Unknown Document')
        page = doc.metadata.get('page', 'Unknown Page')
        doc_contents.append(f"Document: {source}, Page: {page}\n{doc.page_content}")
      retrieved_docs_content = "\n".join(doc_contents) + "\n"
    else:
      #Keep it empty
      retrieved_docs_content = ""

  return retrieved_docs_content