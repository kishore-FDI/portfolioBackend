from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import os
from langchain_community.document_loaders import PyPDFLoader

class SentenceTransformerWrapper(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

def create_embeddings(pdf_path):
    # Initialize the embedding model
    embedder = SentenceTransformerWrapper()
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embedder,
        persist_directory="chroma_db"
    )
    vectorstore.persist()
    
    print(f"Created embeddings for {len(texts)} chunks of text")
    print("Vector store saved in 'chroma_db' directory")

if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "documents/resume.pdf"
    create_embeddings(pdf_path) 