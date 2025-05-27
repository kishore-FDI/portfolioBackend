import spacy
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import os
from langchain_community.document_loaders import PyPDFLoader

# Initialize model globally to avoid reloading
nlp = None

def get_model():
    global nlp
    if nlp is None:
        # Load the small English model
        nlp = spacy.load('en_core_web_sm')
    return nlp

class SpacyEmbeddings(Embeddings):
    def __init__(self):
        self.nlp = get_model()
    
    def embed_documents(self, texts):
        # Process in smaller batches to reduce memory usage
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            docs = list(self.nlp.pipe(batch))
            batch_embeddings = [doc.vector.tolist() for doc in docs]
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def embed_query(self, text):
        doc = self.nlp(text)
        return doc.vector.tolist()

def create_embeddings(pdf_path):
    # Initialize the embedding model
    embedder = SpacyEmbeddings()
    
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