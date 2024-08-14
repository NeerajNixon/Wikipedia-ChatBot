from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pydantic import BaseModel
from typing import List, Tuple
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from dotenv import load_dotenv
from chat.ingest import ingest
from chat.rag import generation

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
llm_transformer = LLMGraphTransformer(llm=llm)

vstore = None
chat_history = []

class IngestRequest(BaseModel):
    page: str

class ChatRequest(BaseModel):
    question: str

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/ingest", methods=["POST"])
def ingest_endpoint():
    global vstore, chat_history
    try:
        data = request.json
        page = data.get("page")
        vstore = ingest(page)
        chat_history = []  # Reset chat history
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    global chat_history
    try:
        data = request.json
        question = data.get("question")
        chain = generation(vstore)
        response = chain.invoke({"question": question, "chat_history": chat_history})
        chat_history.append((question, response))
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
