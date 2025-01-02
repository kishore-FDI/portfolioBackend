from gpt4all import Embed4All
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from flask import Flask, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader

class Embed4AllWrapper(Embeddings):
    def __init__(self):
        self.embedder = Embed4All()
    
    def embed_documents(self, texts):
        return [self.embedder.embed(text) for text in texts]
    
    def embed_query(self, text):
        return self.embedder.embed(text)

class RAGApplication:
    def __init__(self):
        self.embedder = Embed4AllWrapper()
        genai.configure(api_key="AIzaSyDD-afERCwfUOml3Msr0KruJ9dJ6O0EKrY")
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Load single PDF file
        loader = PyPDFLoader("documents/temp.pdf")
        documents = loader.load()
        
        # Initialize vectorstore
        texts = self.text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_texts([t.page_content for t in texts], self.embedder)

    def query(self, question):
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Use the following context to answer the question. If you cannot answer
        based on the context alone, say so.
        
        Context: {context}
        
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)