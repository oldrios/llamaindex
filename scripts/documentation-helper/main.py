from dotenv import load_dotenv
import os
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer
import streamlit as st
from node_postprocessor.duplicate_postprocessing import (
    DuplicateRemoverNodePostProcessor,
)

load_dotenv()

st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)


llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager(handlers=[llama_debug])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)


@st.cache_resource(show_spinner=True)
def get_index() -> VectorStoreIndex:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, callback_manager=callback_manager
    )


index = get_index()
if "chat_engine" not in st.session_state.keys():
    postprocessor = SentenceEmbeddingOptimizer(
        embed_model=service_context.embed_model,
        percentile_cutoff=0.5,
        threshold_cutoff=0.7,
    )

    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        node_postprocessor=[postprocessor, DuplicateRemoverNodePostProcessor()],
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
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistante"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt)
            st.write(response.response)

            nodes = [node for node in response.source_nodes]
            for col, node in zip(st.columns(len(nodes)), nodes):
                with col:
                    st.header(f"Source Node - Score = {node.score}")
                    st.write(node.text)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
