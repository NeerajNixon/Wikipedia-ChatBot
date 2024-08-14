from typing import Tuple, List, Optional
from neo4j import GraphDatabase
import os
from langchain_community.vectorstores import Neo4jVector
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from chat.ingest import ingest

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph()

def generation(vector_index):

    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    llm_transformer = LLMGraphTransformer(llm=llm)

    class Entities(BaseModel):
        """Identifying information about entities."""

        names: List[str] = Field(
            ...,
            description="All the person, organization, or business entities that "
            "appear in the text",
        )
        
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
        
    entity_chain = prompt | llm.with_structured_output(Entities)
        
    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Fulltext index query
    def structured_retriever(question: str) -> str:
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result
    
    def retriever(question: str):
        print(f"Search query: {question}")
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
        """
        return final_data
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
                    in its original language.
                    Chat History:
                    {chat_history}
                    Follow Up Input: {question}
                    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
        )

    template = """Answer the question based only on the following context:
                {context}

                Question: {question}
                Use natural language and be concise.
                Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
            RunnableParallel(
                {
                    "context": _search_query | retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )
    
    return chain
        
        
if __name__=='__main__':
    vstore = ingest("Arda Güler")
    chain  = generation(vstore)
    print(chain.invoke({"question":"when was Arda Güler born"}))        

