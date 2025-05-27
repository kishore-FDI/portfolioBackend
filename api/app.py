import spacy
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from flask import Flask, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
from prompts import SYSTEM_PROMPT, BACKGROUND_INFO

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

class RAGApplication:
    def __init__(self):
        self.embedder = SpacyEmbeddings()
        genai.configure(api_key="AIzaSyDD-afERCwfUOml3Msr0KruJ9dJ6O0EKrY")
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Load the saved vectorstore from chroma_db
        if os.path.exists("chroma_db"):
            self.vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=self.embedder
            )
        else:
            raise Exception("Vector store not found. Please run create_embeddings.py first.")

    def query(self, question):
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""{SYSTEM_PROMPT}

        Additional Context from Documents:
        {context}

        Background Information:
        {BACKGROUND_INFO}

        Question: {question}"""
        
        response = self.model.generate_content(prompt)
        return {
            'answer': response.text,
            'context': context
        }

app = Flask(__name__)
rag_app = RAGApplication()

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'message': 'RAG Service is running!'})

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        result = rag_app.query(data['question'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Removed the if __name__ == '__main__' block so that the app is exposed as 'app' for Vercel hosting. 