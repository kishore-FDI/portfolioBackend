from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os

def create_and_save_embeddings():
    # Initialize embedder and text splitter
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Load all PDFs from documents folder
    documents = []
    pdf_dir = "documents"
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created {pdf_dir} directory")
        return
    
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"Loaded {filename}")
    
    if not documents:
        print("No PDF files found in documents folder")
        return
    
    # Split documents and create embeddings
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embedder)
    
    # Save the vectorstore
    vectorstore.save_local("vectorstore")
    print("Embeddings created and saved successfully")

if __name__ == "__main__":
    create_and_save_embeddings() 