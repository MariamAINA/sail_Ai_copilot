# RETRIEVAL-AUGMENTED GENERATION (RAG) 

# streamlit run sail_streamlit.py
# Chat Q&A Framework for RAG Apps

# Imports 
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import CHAT OLLAMA
from langchain_community.chat_models import ChatOllama

import streamlit as st
import yaml
import uuid

# Initialize Streamlit app
st.set_page_config(page_title="SAIL AI Copilot",layout="wide")
st.title("SAIL AI Copilot")

# Load the API Key securely
OPENAI_API_KEY = yaml.safe_load(open('credentials.yml'))['openai']

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

def create_rag_chain(api_key):
    
    embedding_function = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        api_key=api_key
        #chunk_size=500,
    )
    vectorstore = Chroma(
        persist_directory="data/chroma_store",
        embedding_function=embedding_function
    )
    
    retriever = vectorstore.as_retriever()
    # retriever.search_kwargs.update({"k": 3, "score_threshold": 0.5})

    
    llm = ChatOpenAI(
         model="gpt-3.5-turbo", 
         temperature=0.4, 
         api_key=api_key,
         max_tokens=3500,
     )
    #llm = ChatOllama(
        #model="llama3"
        #model="llama3:70b"
        #)

    # COMBINE CHAT HISTORY WITH RAG RETREIVER
    # * 1. Contextualize question: Integrates RAG
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # * 2. Answer question based on Chat Context
    qa_system_prompt = """You are a SAIL AI Copilot for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    Retrive every information about the question to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use seven sentences maximum and keep the answer concise.\
    Use bullet points if necessary.\
    If you don't know any answer, don't try to make up an answer. Just say that you don't know and to contact the Communication Team at SAIL Innovation Lab./
    The ways to contact company support is: @sailnigeria./
    If questions are asked with respect to specific areas of the ministry's mandate i.e., Telecommunication, Innovation, IT Services, Digital Economy or any other specific function, ensure your response is focused on the specific area i.e., initiatives, projects, activities around connectivity falls under telecommunication./
    Don't be overconfident and don't hallucinate. Ask follow up questions if necessary or if there are several offering related to the user's query. Provide answer with complete details in a proper formatted manner./

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # * Combine both RAG + Chat Message History
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

rag_chain = create_rag_chain(OPENAI_API_KEY)


# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Enter your question about SAIL Innovation Lab"):
    with st.spinner("Hold on please ......."):
        st.chat_message("human").write(question)     
           
        response = rag_chain.invoke(
            {"input": question}, 
            config={
                "configurable": {"session_id": "any"}
            },
        )
        # Debug response
        # print(response)
        # print("\n")
  
        st.chat_message("ai").write(response['answer'])

# * NEW: View the messages for debugging
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)