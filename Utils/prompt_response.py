from urllib import response

from openai import OpenAI
from prompt_toolkit import prompt
from pydantic import BaseModel, Field
from typing import TypedDict, Optional, Dict, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from Utils.rag_pipeline import rag

class QueryType(BaseModel):
  """
  State object to get the query type
  """
  query_type: Literal["troubleshooting", "comparison", "general"]

def getQueryType (llm, model, query: str) -> QueryType:
    system_message = """You are a helpful assistant for Samsung Customer Care Exexutive that categoriezes the user qeries into one of the following categories:
    - Troubleshooting
    - Product Comparison
    - General Inquiry

    You should only respond with the category name and NOTHING ELSE
    """
    prompt = ChatPromptTemplate.from_messages([
      ("system", system_message),
      ("human", "{query}")
    ])
    
  # Format the prompt with the current query to get the messages
    messages = prompt.format_messages(query=query)
    response = llm.with_structured_output(QueryType).invoke(messages)
    return response
 

##TroubleShooting
##Possible causes should be list of strings, step by step solution should be list of strings, when to escalate should be list of strings and reference should be string with document source and page numbers if available or "No specific document references available."
class troubleshootingOutput(BaseModel):
  """
  Output format for troubleshooting query type.
  """
  possible_causes: List[str]
  step_by_step_solution: List[str]
  when_to_escalate: List[str]
  reference: str


def troubleshootReseponse(llm, current_query: str, chat_history:str, vector_embeddings):
  """
  Function to generate troubleshooting response
  """
  # Call the rag function to get the relevant context and references
  retrieved_docs_with_references = rag(current_query, chat_history, vector_embeddings)

  system_message = """You are an helpful assistant on how to troubleshoot the given problem.
Your response should be structured according to the troubleshootingOutput schema.
For the 'reference' field, you MUST extract and provide the document source and page numbers from the 'Relevant context from knowledge base' section if available.
If no relevant documents are found, or if the context does not contain specific document references, state 'No specific document references available.'
In the past chat, if the solution was already provided, do not repeat the solutions. Instead suggest next steps or ask for more information if needed.
"""

  if retrieved_docs_with_references:
    system_message += f"\nRelevant context from knowledge base:\n{retrieved_docs_with_references}"
  # No explicit else block needed here, as the system message already covers the 'no relevant docs' scenario

  if chat_history:
    system_message += f"\nChat history: {chat_history}"

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_message),
      ("human", "{query}")
  ])

  # Format the prompt with the current query to get the messages
  messages = prompt.format_messages(query=current_query)
  response = llm.with_structured_output(troubleshootingOutput).invoke(messages)
  return response


##GneralOutput
class generalOutput(BaseModel):
  """
  Output format for general query type.
  """
  direct_answer: str
  explanation: str
  additional_notes: str
  reference: str


def generalResponse(llm, current_query: str, chat_history:str, vector_embeddings):
  """
  Function to generate general response
  """
  # Call the rag function to get the relevant context and references
  retrieved_docs_with_references = rag(current_query, chat_history, vector_embeddings)
  system_message = """You are an helpful assistant to a Samsung customer care executive on how to answer the given question which
  Your response should be structured according to the generalOutput schema.
  For the 'reference' field, you MUST extract and provide the document source and page numbers from the 'Relevant context from knowledge base' section if available.
  If no relevant documents are found, or if the context does not contain specific document references, state 'No specific document references available.'
  """

  if retrieved_docs_with_references:
    system_message += f"\nRelevant context from knowledge base:\n{retrieved_docs_with_references}"
  # No explicit else block needed here, as the system message already covers the 'no relevant docs' scenario

  if chat_history:
    system_message += f"\nChat history: {chat_history}"

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_message),
      ("human", "{query}")
  ])

  # Format the prompt with the current query to get the messages
  messages = prompt.format_messages(query=current_query)
  response = llm.with_structured_output(generalOutput).invoke(messages)
  return response


##Feature Comparison
class FeatureComparisonEntry(BaseModel):
  product_names: List[str] = Field(..., description="A list containing the names of the two products being compared (e.g., ['Samsung S25', 'Samsung S26']).")
  feature_name: str = Field(..., description="The name of the feature being compared (e.g., 'Display', 'Processor').")
  value_product_1: str = Field(..., description="The value of the feature for the first product (e.g., 'S25').")
  value_product_2: str = Field(..., description="The value of the feature for the second product (e.g., 'S26').")

class comparisonOutput(BaseModel):
  """
  Output format for comparison query type.
  """
  product_names: List[str] = Field(..., description="A list containing the names of the two products being compared (e.g., ['Samsung S25', 'Samsung S26']).")
  feature_comparison_table: List[FeatureComparisonEntry] = Field(..., description="A table comparing features of two products, with each entry detailing a feature and its values for Product 1 and Product 2.")
  key_differences: List[str] = Field(..., description="A list of bullet points highlighting the key differences between the products.")
  recommendation: str = Field(..., description="A recommendation or summary based on the comparison.")
  reference: str = Field(..., description="References from the knowledge base that informed the comparison.")

def comparisonResponse(llm, current_query: str, chat_history:str, vector_embeddings):
  """
  Function to generate comparison response
  """
  # Call the rag function to get the relevant context and references
  retrieved_docs_with_references = rag(current_query, chat_history, vector_embeddings)
  system_message = """You are an helpful assistant that compares Samsung products (consumer electronics, mobile phones, laptops etc.)
  based on the user's query.
Your response should be structured according to the comparisonOutput schema.
For the 'feature_comparison_table', provide a list of objects, each with 'feature_name', 'value_product_1', and 'value_product_2'.
For 'key_differences', provide a list of strings, each being a distinct difference.
For the 'reference' field, you MUST extract and provide the document source and page numbers from the 'Relevant context from knowledge base' section if available.
If no relevant documents are found, or if the context does not contain specific document references, state 'No specific document references available.'
  """

  if retrieved_docs_with_references:
    system_message += f"\nRelevant context from knowledge base:\n{retrieved_docs_with_references}"
  # No explicit else block needed here, as the system message already covers the 'no relevant docs' scenario

  if chat_history:
    system_message += f"\nChat history: {chat_history}"

  prompt = ChatPromptTemplate.from_messages([
      ("system", system_message),
      ("human", "{query}")
  ])

  # Format the prompt with the current query to get the messages
  messages = prompt.format_messages(query=current_query)
  response = llm.with_structured_output(comparisonOutput, method="function_calling").invoke(messages)
  return response