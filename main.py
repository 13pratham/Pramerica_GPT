# Models
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# Text Splittter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector DBs
from langchain_google_genai import GoogleVectorStore
# from langchain_chroma import Chroma
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

# environ
import os
import streamlit as st
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
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
# embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

# Set up Streamlit
st.set_page_config(page_title = 'Pramerica Advisor GPT')
st.sidebar.title('Pramerica Advisor GPT - The Conversational AI Chatbot')
st.sidebar.write('This chatbot is designed to help our agents get the client queries resolved in seconds.')

agent_id = st.sidebar.text_input('Agent ID')
if not agent_id:
    st.sidebar.warning("Enter Agent ID '70------'")
else:
    if len(agent_id) != 8 or agent_id[:2] != '70':
        st.sidebar.warning("Agent ID is invalid")
        agent = 0
    else:
        agent = 1

if 'store' not in st.session_state:
    st.session_state.store = {}

session_id = st.sidebar.selectbox('Clients', ['New Client'] + [i for i in st.session_state.store])

if session_id == 'New Client':
    session_id = st.sidebar.text_input('Client Name')

st.sidebar.subheader('Models and parameters')

selected_model = st.sidebar.selectbox('Choose an OpenAI Model', ['gemini-1.5-flash (recommended)', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'Gemma2-9b-it', 'gpt-4o-turbo', 'gpt-4o', 'gpt-3.5-turbo'], key = 'selected_model')
temperature = st.sidebar.slider('temperature', min_value = 0.1, max_value = 1.0, value = 0.5, step = 0.1)
max_tokens = st.sidebar.slider('max_tokens', min_value = 100, max_value = 1000, value = 500, step = 50)

if selected_model == 'gemini-1.5-flash-8b':
    model = 'gemini-1.5-flash-8b'
elif selected_model == 'gemini-1.5-pro':
    model = 'gemini-1.5-pro'
elif selected_model == 'gpt-4o':
    model = 'gpt-4o'
elif selected_model == 'gpt-4o-turbo':
    model = 'gpt-4o-turbo'
elif selected_model == 'gpt-3.5-turbo':
    model = 'gpt-3.5-turbo'
elif selected_model == 'Gemma2-9b-it':
    model = 'Gemma2-9b-it'
else:
    model = 'gemini-1.5-flash'

if 'gpt' in selected_model:
    llm = ChatOpenAI(api_key = openai_api_key, model = model, temperature = temperature, max_tokens = max_tokens)
elif 'gemini' in selected_model:
    llm = ChatGoogleGenerativeAI(model = model, api_key = gemini_api_key, temperature = temperature, max_tokens = max_tokens)
else:
    llm = ChatGroq(groq_api_key = groq_api_key, model = model, temperature = temperature, max_tokens = max_tokens)

st.session_state.embeddings = OpenAIEmbeddings()
st.session_state.loader = PyPDFDirectoryLoader('Product_Brochures')
st.session_state.docs = st.session_state.loader.load()
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

retriever = st.session_state.vectors.as_retriever()

contextualize_q_system_prompt = (
    """
    Given a chat history and latest user question which might refere context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    You are an insurance advisor for helping insurance agents. Try to provide detailed information.
    Only respond in context of Pramerica Life Products. Don't  provide any other information.
    If you don't know the answer or question is out of context of Pramerica Products, just say Out of Context.
    Also, remember some abbreviations like SIP is Super Investment Plan, RSF is Rock Solid Future,
    GROW is Guaranteed Return on Wealth.
    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer Question Prompt

system_prompt = (
    """
    You are an insurance advisor for helping insurance agents.
    Use the following pieces of retrived context to answer the question.
    Try provide detailed information with facts and numbers if possible.
    Only respond in context of Pramerica Life Products. Don't provide any other information.
    If you don't know the answer or question is out of context of Pramerica Products, just say Out of Context.
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

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(session:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key = 'input',
    history_messages_key = 'chat_history',
    output_messages_key = 'answer'
)

if agent_id and agent == 1 and session_id != 'New Client' and session_id != "":
    user_input = st._bottom.text_input(label = 'Your Question:')

    session_history = get_session_history(session_id)
    if user_input:
        response = conversational_rag_chain.invoke(
            {'input' : user_input},
            config = {
                'configurable' : {'session_id' : session_id}
            },
        )

    for i in range(len(session_history.messages)):
        if i%2 == 0:
            message = st.chat_message('human')
            message.write(f'Advisor Message: {session_history.messages[i].content}')
        else:
            message = st.chat_message('assistant')
            message.write(f'ChatBot Message: {session_history.messages[i].content}')

        # with st.expander('Documents Reffered'):
        #     for i, doc in enumerate(response['context']):
        #         st.write(doc.page_content)
        #         st.write('-----------------------------------')
                
    # st.write('Chat History:', session_history.messages)

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input():
#     if not openai_api_key:
#         st.info("Please add your OpenAI API key to continue.")
#         st.stop()

# client = OpenAI(api_key=openai_api_key)
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
#     response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
#     msg = response.choices[0].message.content
#     st.session_state.messages.append({"role": "assistant", "content": msg})
#     st.chat_message("assistant").write(msg)
