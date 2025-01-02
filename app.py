import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, request, jsonify
import os

class RAGApplication:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        genai.configure(api_key="AIzaSyDD-afERCwfUOml3Msr0KruJ9dJ6O0EKrY")
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Load the saved vectorstore if it exists
        if os.path.exists("vectorstore"):
            self.vectorstore = FAISS.load_local("vectorstore", self.embedder)
        else:
            raise Exception("No vectorstore found. Please run create_embeddings.py first.")

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