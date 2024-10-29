# Models
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Text Splitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF Loader
# from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector DBs
from langchain_community.vectorstores import FAISS  # Make sure this line is uncommented

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
from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAXy-vh0frJIgeSZ_aL_Z9ZK2bJWMzog5U"
gemini_api_key = "AIzaSyAXy-vh0frJIgeSZ_aL_Z9ZK2bJWMzog5U"
# nv_api_key = "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"
# os.environ["NVIDIA_API_KEY"] = nv_api_key
os.environ['LANGCHAIN_API_KEY'] = "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"
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = "Chatbot_Testing"

# Global MongoDB client variable
cnxn = None

def preprocess():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    llm = ChatGoogleGenerativeAI(api_key=gemini_api_key, model='gemini-1.5-flash', temperature=0.4, max_tokens=500)
    
    vectors = FAISS.load_local(folder_path="VectorDB/", embeddings=embeddings, index_name="Google-001_Embeddings", allow_dangerous_deserialization=True)
    retriever = vectors.as_retriever()

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

def get_message_history(Agent_ID, Session_ID):
    tbl = cnxn["BusinessEnablerAdmin"]["message_history"]
    return list(tbl.find({"agent_id": Agent_ID, "session_id": Session_ID}))

def get_session_history(Session_ID) -> ChatMessageHistory:
    messages = ChatMessageHistory()
    Agent_ID, Session_ID = Session_ID.split('_')
    message_history = get_message_history(Agent_ID, Session_ID)

    for i in message_history:
        messages.add_user_message(i['question'])
        messages.add_ai_message(i['answer'])
    
    return messages

def get_client_history(Agent_ID):
    tbl = cnxn["BusinessEnablerAdmin"]["message_history"]
    return [{"agent_id": i["agent_id"], "session_id": i["session_id"], "client_name": i["client_name"]} for i in tbl.find({"agent_id": Agent_ID})]

def response(question, Agent_ID, Session_ID, Client_Name, rag_chain):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    response = conversational_rag_chain.invoke(
        {'input': question},
        config={'configurable': {'session_id': Agent_ID + '_' + Session_ID}},
    )

    tbl = cnxn["BusinessEnablerAdmin"]["message_history"]
    
    data = {
        "agent_id": Agent_ID,
        "session_id": Session_ID,
        "client_name": Client_Name,
        "question": question,
        "answer": response["answer"],
        "datime": datetime.datetime.today()
    }

    tbl.insert_one(data)
    return data

# API Setup
rag_chain = preprocess()

# Initialize FastAPI app
app = FastAPI(title="Langchain Server",
              version="1.0",
              description="A simple API server using Langchain runnable interfaces")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Connect to MongoDB at startup
@app.on_event("startup")
async def startup_event():
    global cnxn
    cnxn_string = "mongodb+srv://BusinessEnablerAdmin:MuiUJxhXkrGO8RTM@businessenabler-uat.ystpsmi.mongodb.net/"
    cnxn = pymongo.MongoClient(cnxn_string)

# Close MongoDB connection when shutting down
@app.on_event("shutdown")
async def shutdown_event():
    cnxn.close()

class Client(BaseModel):
    agent_id: str
    
class Message(BaseModel):
    agent_id: str
    session_id: str
    
class QA(BaseModel):
    agent_id: str
    session_id: str
    client_name: str
    question: str

@app.post("/response/")
def QA_bot(Question: QA):

    ques = Question.question.lower()
    ques = "Pramerica Life Super Investment Plan".join(ques.split("sip"))
    ques = "Pramerica Life RockSolid Future".join(ques.split("rsf"))
    ques = "Pramerica Life Smart Wealth Plus".join(ques.split("sw+"))
    ques = "Pramerica Life Guaranteed Return on Wealth".join(ques.split("grow"))
    ques = "Premium Paying Term".join(ques.split("ppt"))
    ques = "Policy Term".join(ques.split("pt"))

    Response = response(
        question=ques,
        Agent_ID=Question.agent_id,
        Session_ID=Question.session_id,
        Client_Name=Question.client_name,
        rag_chain=rag_chain
    )
    return {
        "agent_id": Question.agent_id,
        "session_id": Question.session_id,
        "client_name": Question.client_name,
        "question": Question.question,
        "answer": Response['answer'],
        "datime": Response['datime']
    }

@app.post("/message-history/")
def message_history(ID: Message):
    messages = get_message_history(Agent_ID=ID.agent_id, Session_ID=ID.session_id)
    return [{"agent_id": i["agent_id"], "session_id": i["session_id"], "client_name": i["client_name"], "question": i["question"], "answer": i["answer"]} for i in messages]

@app.post("/client-history/")
def client_history(Agent_ID: Client):
    df = pd.DataFrame(get_client_history(Agent_ID=Agent_ID.agent_id))
    df = df.drop_duplicates()
    clients = [{"agent_id": df.loc[i, "agent_id"], "session_id": df.loc[i, "session_id"], "client_name": df.loc[i, "client_name"]} for i in df.index]
    return clients

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
