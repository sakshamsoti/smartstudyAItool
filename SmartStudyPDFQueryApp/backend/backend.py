import os
import time
import re
import json
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "docudive"
COLLECTION_NAME = "pdf_embeddings"

# Connect to MongoDB
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
pdf_collection = db[COLLECTION_NAME]

# Function to clean text
def clean_text(text: str) -> str:
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load and Prepare PDF Data (text, images, tables)
def load_and_prepare_data(pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 100):
    pdf_name = os.path.basename(pdf_path)
    existing_pdf = pdf_collection.find_one({"pdf_name": pdf_name})
    
    if existing_pdf:
        documents = json.loads(existing_pdf["documents"])
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_texts(documents, embeddings, persist_directory="chroma_db")
    else:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(split_documents, embeddings, persist_directory="chroma_db")

        document_texts = [doc.page_content for doc in split_documents]
        pdf_collection.insert_one({
            "pdf_name": pdf_name,
            "documents": json.dumps(document_texts),
            "metadata": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        })
    
    return vector_store

# Function to process user query
def process_query(user_input, selected_pdf):
    vector_store = load_and_prepare_data(f"pdfs/{selected_pdf}")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    if not context.strip():
        return "I couldn't find relevant information in the document."
    else:
        prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context:\n{context}\n\nUser question:\n{query}\n\n"
            "If you cannot find an answer, respond with: 'I couldn't find any relevant information.'"
        )
        prompt = prompt_template.format(context=context, query=user_input)
        llm = ChatOpenAI(model="gpt-4", streaming=True)
        response = ""
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            if hasattr(chunk, "content"):
                response += chunk.content
        return response
