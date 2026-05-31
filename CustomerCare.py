import streamlit as st

import os
import re
import json
import random
from datetime import datetime

from dotenv import load_dotenv

from typing import Dict, List, Optional, TypedDict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from Utils.rag_pipeline import create_vector_store, rag, load_pdf, load_txt
import Utils.prompt_response as prompt
from openai import OpenAI
from openai import AzureOpenAI

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

# Instantiating the OpenAI client with the API key and base URL
llm = AzureChatOpenAI(
    api_key = os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint= os.environ["AZURE_END_POINT"],
    deployment_name = os.environ["AZURE_OPENAPI_MODEL"],
    api_version=os.environ["AZURE_API_VERSION"],
    temperature = 0
)


# Instantiating the OpenAI Embedding model with the API key and base URL
embedding_model = AzureOpenAIEmbeddings(
    api_key = os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_EMBEDDING_VERSION"],
    model = os.environ["AZURE_EMBEDDING_MODEL"]
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Samsung AI Assistant", layout="wide")
st.title("🏥 Samsung Customer Care")
st.subheader("AI Assistant for Customer Support")


# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.subheader("⚙️ Settings")

debug_mode = st.sidebar.checkbox("Show RAG Details")
use_rag = st.sidebar.checkbox("Enable RAG", value=True)



# -----------------------------
# PDF Upload
# -----------------------------
st.subheader("📂 Upload Reference Document")

uploaded_file = st.file_uploader("Upload PDF or Text", type=["pdf", "txt"])

# -----------------------------
# Cache Vector Store
# -----------------------------
@st.cache_resource
def cached_vector_store(documents):
    return create_vector_store(documents)
    
if uploaded_file:
    with st.spinner("Processing Document..."):
        if uploaded_file.type == "application/pdf":
            document = load_pdf(uploaded_file)
        else:
            document = load_txt(uploaded_file)
            if document is not None:
                st.write(f"Loaded {len(document)} documents from uploaded text file.")

        if document is None:
            st.error("Unable to load the uploaded document. Please check the file and try again.")
        else:
            st.session_state.vector_store = cached_vector_store(document)
            st.success("Document processed!")

# -----------------------------
# Chat State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Ask the Assistant")

def display_chat_history():
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

user_query = st.chat_input("Enter your question")

if user_query:

    # -----------------------------
    # Show user message
    # -----------------------------
  #  with st.chat_message("user"):
  #      st.markdown(user_query)

    # -----------------------------
    # Convert chat history → text
    # -----------------------------
    chat_history_text = ""
    for chat in st.session_state.chat_history:
        chat_history_text += f"Customer: {chat['question']}\nAssistant: {chat['answer']}\n"

    # -----------------------------
    # Get the Query Type
    # -----------------------------
    query_type_response = prompt.getQueryType(llm, os.environ["AZURE_OPENAPI_MODEL"], user_query)
    query_type = query_type_response.query_type

    #st.write(f"Identified Query Type: **{query_type}**")
#  possible_causes: List[str]
##  step_by_step_solution: List[str]
  # when_to_escalate: List[str]
 # reference: str

    if query_type == "troubleshooting":
        vector_store = st.session_state.get("vector_store", None)
        if vector_store is None:
            st.warning("⚠️ No document uploaded. Upload a PDF/TXT file first.")
        else:
            st.info(f"✅ Vector store ready with {len(vector_store.index_to_docstore_id)} documents")

        response = prompt.troubleshootReseponse(llm, user_query, chat_history_text, vector_store)
        formatted_answer = f"""
        **Query Type:** Troubleshooting \n
        **Possible Causes:**\n """
        for entry in response.possible_causes:
            formatted_answer += f"* {entry}\n"
        formatted_answer += f"\n**Step-by-Step Solution:**\n"
        for i, entry in enumerate(response.step_by_step_solution, start=1):
            formatted_answer += f"{i}. {entry}\n"
        formatted_answer += f"\n**When to Escalate:**\n"
        for entry in response.when_to_escalate:
            formatted_answer += f"* {entry}\n"
        formatted_answer += f"\n**Reference:** *{response.reference}*" if response.reference else "\n**Reference:** No specific document references available."
        formatted_answer += "\n"
        answer = formatted_answer

    elif query_type == "comparison":
        response = prompt.comparisonResponse(llm, user_query, chat_history_text, st.session_state.get("vector_store", None))
        #create a formatted tablular response in md with | feature | product 1 | product 2 | and then list the key differences and recommendation below the table
        formatted_answer = "| Feature | " + response.product_names[0] + " | " + response.product_names[1] + " |\n|---|---|---|\n"
        for entry in response.feature_comparison_table:
            formatted_answer += f"| {entry.feature_name} | {entry.value_product_1} | {entry.value_product_2} |\n"
        formatted_answer += f"\n**Key Differences:**\n"
        for diff in response.key_differences:
            formatted_answer += f"- {diff}\n"
        formatted_answer += f"\n**Recommendation:** {response.recommendation}\n"
        formatted_answer += f"\n**Reference:** {response.reference}\n"
        answer = formatted_answer
    else:
        response = prompt.generalResponse(llm, user_query, chat_history_text, st.session_state.get("vector_store", None))
        formatted_answer = f"""
        **Query Type:** General \n
        **Direct Answer:** {response.direct_answer} \n
        **Explanation:** {response.explanation} \n
        **Additional Notes:** {response.additional_notes} \n
        **Reference:** {response.reference}
        """
        answer = formatted_answer

    #with st.chat_message("assistant"):
    #    placeholder = st.empty()
    #    placeholder.markdown(answer)

    # -----------------------------
    # Save Chat History
    # -----------------------------
    st.session_state.chat_history.append(
        {"question": user_query, "answer": answer}
    )

    # -----------------------------
# Display Chat History
# -----------------------------
display_chat_history()