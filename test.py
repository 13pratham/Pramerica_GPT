# Models
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Text Splitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF Loader
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector DBs
from langchain_community.vectorstores import FAISS

# Message History & Prompt
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# API
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# environ
import os
import pymongo
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAXy-vh0frJIgeSZ_aL_Z9ZK2bJWMzog5U"
gemini_api_key = "AIzaSyAXy-vh0frJIgeSZ_aL_Z9ZK2bJWMzog5U"
genai.configure(api_key = gemini_api_key)
# nv_api_key = "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"
# os.environ["NVIDIA_API_KEY"] = nv_api_key
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Chatbot_Testing"

st.set_page_config(page_title = "Tarang AI", page_icon = ":rock:", layout = "wide")
st.title("Tarang AI")
st.write("This a Gen AI  chatbot to help agents get there queries resolved related to different products")
st.info("""This bot is trained on some products only:
        1. Rakshak Smart
        2. Smart Income
        3. Super Investment Plan (SIP)
        4. Smart Wealth Plus (SW+)""")

def preprocess(embedding_model, llm_model, temperature, tokens):
    
    embeddings = GoogleGenerativeAIEmbeddings(model = embedding_model)
    llm = ChatGoogleGenerativeAI(api_key = gemini_api_key, model = llm_model, temperature = temperature, max_tokens = tokens)
    vectors = FAISS.load_local(folder_path = "VectorDB/", embeddings=embeddings, index_name = "Google-" + embedding_model.split("-")[-1] + "_Embeddings", allow_dangerous_deserialization = True)
    retriever = vectors.as_retriever(search_kwargs = {'k': 10})

    contextualize_q_system_prompt = """
    Given a chat history and latest user question which might refer context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    You are an insurance advisor for helping insurance agents. Try to provide detailed information.
    Only respond in context of Pramerica Life Products. Don't provide any other information.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    system_prompt = """
    You are an insurance advisor for helping insurance agents.
    Use the following pieces of retrieved context to answer the question.
    Try to provide detailed information with facts and numbers.
    If you don't know the answer, say you didn't understand and can you reframe the question?.
    \n\n
    {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}'),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def get_session_history(session : str) -> BaseChatMessageHistory:
    
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    
    return st.session_state.store[session_id]

st.sidebar.title("Tarang AI")
st.sidebar.write("Configure your Gen AI App")
embedding_model = st.sidebar.selectbox(label = "Select an Embedding Model:", options = [i.name for i in genai.list_models() if "embedContent" in i.supported_generation_methods])
llm_model = st.sidebar.selectbox(label = "Select a LLM Model:", options = [i.name.split("/")[1] for i in genai.list_models() if "generateContent" in i.supported_generation_methods])
temperature = st.sidebar.slider(label = "Set Temperature:", min_value = 0.1, max_value = 2.0, step = 0.1, value = 0.4)
tokens = st.sidebar.slider(label = "Set max tokens:", min_value = 128, max_value = 1024, step = 128, value = 512)
rag_chain = preprocess(embedding_model, llm_model, temperature, tokens)

session_id = "default"

if "store" not in st.session_state:
    st.session_state.store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key = 'input',
    history_messages_key = 'chat_history',
    output_messages_key = 'answer'
)

user_input = st._bottom.text_input("Enter your question:")

if user_input:

    session_history = get_session_history(session_id)
    ques = user_input.lower()
    ques = "Pramerica Life Super Investment Plan".join(ques.split("sip"))
    ques = "Pramerica Life RockSolid Future".join(ques.split("rsf"))
    ques = "Pramerica Life Smart Wealth Plus".join(ques.split("sw+"))
    ques = "Pramerica Life Guaranteed Return on Wealth".join(ques.split("grow"))
    ques = "Premium Paying Term".join(ques.split("ppt"))
    ques = "Policy Term".join(ques.split("pt"))
    response = conversational_rag_chain.invoke(
        {'input': ques},
        config = {'configurable': {'session_id': session_id}},
    )

    for i in range(len(session_history.messages)):
        if i%2 == 0:
            message = st.chat_message('human')
            message.write(f'Agent Message: {session_history.messages[i].content}')
            
        else:
            message = st.chat_message('assistant')
            message.write(f'Bot Message: {session_history.messages[i].content}')
            with st.expander('Documents Reffered'):
                for i, doc in enumerate(response['context']):
                    st.write(i+1)
                    st.write("Product Brochure: ", doc.metadata)
                    st.write("Content: ", doc.page_content)
                    st.write('-----------------------------------')