from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
import streamlit as st

load_dotenv()

st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

@st.cache_resource(show_spinner=True)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, callback_manager=callback_manager
    )


index = get_index()
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT, verbose=True
    )

st.title("Chat with LlamaIndex docs 💬🦙")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library?",
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistante"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
