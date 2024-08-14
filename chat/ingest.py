from typing import Tuple, List, Optional
from neo4j import GraphDatabase
import os
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set API keys and Neo4j credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize embeddings and LLM transformer
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
llm_transformer = LLMGraphTransformer(llm=llm)


def ingest(page: str):
    # Reset the database before ingesting new data
    graph.query("MATCH (n) DETACH DELETE n")
    print("Database reset to blank state.")

    # Load and process documents
    raw_documents = WikipediaLoader(query=page).load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    print("done")
    # Add documents to the graph
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    print("done1")
    
    # Create a vector index for the graph
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    
    return vector_index

