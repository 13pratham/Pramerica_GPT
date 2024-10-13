# Models
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Text Splittter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector DBs
# from langchain_google_genai import GoogleVectorStore
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# Message History & Prompt
import json
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
from langserve import add_routes

# environ
import os
# import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Laod the API Keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.environ['GEMINI_API_KEY']

# genai.configure(api_key = gemini_api_key)

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Chatbot_Testing"
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

def preprocess():
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # vector = embeddings.embed_query("hello, world!")

    llm = ChatGoogleGenerativeAI(api_key = gemini_api_key, model = 'gemini-1.5-flash', temperature = 0.4, max_tokens = 500)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')
    # loader = PyPDFDirectoryLoader('Product_Brochures')
    # docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    # final_documents = text_splitter.split_documents(docs)
    # vectors = FAISS.from_documents(final_documents, embeddings)

    # vectors.save_local(folder_path = "VectorDB/", index_name = "Huggingface_Embeddings")

    vectors = FAISS.load_local(folder_path = "VectorDB/", embeddings = embeddings, index_name = "Huggingface_Embeddings", allow_dangerous_deserialization = True)
    retriever = vectors.as_retriever()

    contextualize_q_system_prompt = (
        """
        Given a chat history and latest user question which might refere context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        You are an insurance advisor for helping insurance agents. Try to provide detailed information.
        Only respond in context of Pramerica Life Products. Don't  provide any other information.
        Don't give partial responses, try to end at last possible sentence.
        """
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Answer Question Prompt

    system_prompt = (
        """
        You are an insurance advisor for helping insurance agents.
        Use the following pieces of retrived context to answer the question.
        Try provide detailed information with facts and numbers if possible.
        Only respond in context of Pramerica Life Products. Don't provide any other information.
        If you don't know the answer, say you didn't understand and can you reframe the question?.
        \n\n
        {context}
        """
    )

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
    
    return (rag_chain)

def get_message_history(Session_ID):
    
    with open("Message_History.json", "r") as file:
        message_history = json.load(file)
        file.close()

    if Session_ID not in message_history:
        message_history[Session_ID] = {
            "User" : ["Hi"],
            "Assistant" : ["Hello! How can I help you?"]
        }
    
    with open("Message_History.json", "w") as file:
        file.write(json.dumps(message_history))
        file.close()
        
    return message_history

def get_session_history(Session_ID)-> BaseChatMessageHistory:
    
    messages = ChatMessageHistory()
    message_history = get_message_history(Session_ID)

    for i in range(len(message_history[Session_ID]['User'])):
        messages.add_user_message(message_history[Session_ID]['User'][i])
        messages.add_ai_message(message_history[Session_ID]['Assistant'][i])
    
    return messages

def response(user_input, Session_ID, rag_chain):
    
    message_history = get_message_history(Session_ID = Session_ID)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = 'input',
        history_messages_key = 'chat_history',
        output_messages_key = 'answer'
    )

    response = conversational_rag_chain.invoke(
        {'input' : user_input},
        config = {
            'configurable' : {'session_id' : Session_ID}
        },
    )

    message_history[Session_ID]['User'].append(user_input)
    message_history[Session_ID]['Assistant'].append(response['answer'])

    with open("Message_History.json", "w") as file:
        file.write(json.dumps(message_history))
        file.close()
    
    return(response['answer'])

# API Setup
rag_chain = preprocess()

app = FastAPI(title = "Langchain Server",
              version = "1.0",
              description = "A simple API server using Langchain runnable interfaces")

class QA(BaseModel):
    Agent_ID : str
    Session_ID : str
    Question : str

@app.post("/Response/")
def QA_bot(Question: QA):
    Session_ID = Question.Agent_ID + '_' + Question.Session_ID
    Response = response(user_input = Question.Question, Session_ID = Session_ID, rag_chain = rag_chain)
    return (Response)

@app.post("/Message_History/")
def message_history(Session_ID: QA):
    Session_ID = Session_ID.Agent_ID + '_' + Session_ID.Session_ID
    return (get_message_history(Session_ID = Session_ID)[Session_ID])

if __name__ == "__main__":
    uvicorn.run(app, host = "localhost", port = 8000)