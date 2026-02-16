import os
import json
from datetime import datetime
import time
import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import config


model_ = config.MODEL_NAME


# === Chat History Disk Persistence ===
HISTORY_DIR = "chat_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def get_history_filepath():
    session_id = "chat1"  # For now, static session ID
    return os.path.join(HISTORY_DIR, f"{session_id}.json")

def load_history():
    filepath = get_history_filepath()
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []

def save_history(messages):
    filepath = get_history_filepath()
    with open(filepath, "w") as f:
        json.dump([{"type": msg.type, "content": msg.content} for msg in messages], f, indent=2)

# === Context Data Loading ===
def load_context_data(filepath="context_data.json"):
    docs = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    # Format as "Key: Value" for better retrieval context
                    content = f"{key}: {value}"
                    docs.append(Document(page_content=content, metadata={"source": "context_data"}))
        except Exception as e:
            st.error(f"Error loading context data: {e}")
    return docs




# === Streamlit UI Setup ===
st.set_page_config(page_title="Local Chatbot", page_icon="")
st.title("Dogs Trust ChatBot")

# === Sidebar System Prompt ===
with st.sidebar:
    st.subheader("🤖 Bot Personality")
    system_prompt = st.text_area("Set system prompt (personality)", value=config.SYSTEM_MESSAGE)




# === Initialise Session State ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.messages = []

    for msg in load_history():
        if msg["type"] == "human":
            st.session_state.chat_history.add_user_message(msg["content"])
            st.session_state.messages.append(HumanMessage(content=msg["content"]))
        elif msg["type"] == "ai":
            st.session_state.chat_history.add_ai_message(msg["content"])
            st.session_state.messages.append(AIMessage(content=msg["content"]))

# === LLM Setup ===
llm = ChatOllama(model=model_)
retriever = None

# === Load Context & Documents ===
# Always load static context data
context_docs = load_context_data()
all_docs = context_docs.copy()

# Document Upload
uploaded_file = st.file_uploader("\U0001F4C4 Upload a PDF or .txt file", type=["pdf", "txt"])

if uploaded_file is not None:
    with open("uploaded_doc", "wb") as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader("uploaded_doc")
    else:
        loader = TextLoader("uploaded_doc")

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    all_docs.extend(chunks)


# Initialize Vector Store if we have any documents (context or uploaded)
if all_docs:
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    db = FAISS.from_documents(all_docs, embeddings)
    retriever = db.as_retriever()



# === Display Chat History ===
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user", avatar="\U0001F464").write(msg.content)
    else:
        st.chat_message("assistant", avatar="\U0001F916").write(msg.content)

# === Chat Input ===
user_input = st.chat_input("Ask me anything...")


if user_input:
    st.chat_message("user", avatar="\U0001F464").write(user_input)
    st.session_state.chat_history.add_user_message(user_input)

    user_input_with_context = user_input
    if retriever:
        retrieved_docs = retriever.invoke(user_input)
        if retrieved_docs:
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs[:config.RETRIEVAL_K]])
            user_input_with_context = f"Context:\n{context_text}\n\nQuestion: {user_input}"
    
    if not st.session_state.messages:
        messages_to_send = [SystemMessage(content=system_prompt)] + st.session_state.chat_history.messages + [HumanMessage(content=user_input_with_context)]
    else:
        # Prepend system prompt if it's not already in history (it's not saved to history list)
        messages_to_send = [SystemMessage(content=system_prompt)] + st.session_state.chat_history.messages[:-1] + [HumanMessage(content=user_input_with_context)]

    # We need to make sure we don't duplicate the last human message which is already in history
    # The original code logic for history management was a bit loose.
    # Let's clean it up:
    # 1. Add current user msg to history (done above)
    # 2. Construct prompt: System + History (which now includes current msg)
    # 3. BUT we want to inject context into the LAST message of the history for the LLM, 
    #    without altering the displayed history.
    
    # Correct approach:
    history_messages = st.session_state.chat_history.messages
    # Make a copy of the last message (User's input) and modify it with context
    last_msg = history_messages[-1] 
    last_msg_with_context = HumanMessage(content=user_input_with_context)
    
    final_messages = [SystemMessage(content=system_prompt)] + history_messages[:-1] + [last_msg_with_context]
    
    start_time = time.time()
    response = llm.invoke(final_messages)
    end_time = time.time()
    response_time = end_time - start_time

    st.session_state.chat_history.add_ai_message(response.content)
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=response.content))
    save_history(st.session_state.messages)

    st.chat_message("assistant", avatar="\U0001F916").write(response.content)
    st.caption(f"Response generated in {response_time:.2f} seconds")



