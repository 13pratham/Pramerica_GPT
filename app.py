import streamlit as st
import openai
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser

import os
import time
from dotenv import load_dotenv
load_dotenv()

# Laod the API Keys
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key = groq_api_key, model_name = "gemma2-9b-it")

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question asked.
    <context>
    {context}
    </context>
    Question:{input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('Product_Brochures')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input('Enter your query from the Product Brochures')

if st.button('Document Embeddings'):
    create_vector_embeddings()
    st.write('Vector Database is ready')

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_prompt})
    print(f'Response time:{time.process_time()-start}')

    st.write(response['answer'])

    with st.expander('Document Reffered'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('-----------------------------------')
"""
def generate_response(question, api_key, llm, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model = llm)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot with OpenAI")

#Sidebar for setting
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI API Key:", type = 'password')

# Drop down to select various Open AI models
llm = st.sidebar.selectbox("Select an Open AI Model", ['gpt-4o', 'gpt-4-turbo', 'gpt-4'])

# Adjust response parameter
temperature = st.sidebar.slider('Temperature', min_value = 0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider('MAx Tokens', min_value = 50, max_value = 300, value = 200)

# Main interface for user input
st.write('Go ahead and ask any question')
user_input = st.text_input('You:')

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)

else:
    st.write('Please provide the query.')
"""