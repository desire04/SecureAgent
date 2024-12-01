import streamlit as st
from codebase_rag import CodebaseRAG
import time

st.title("Codebase RAG chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How may I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    codebase_rag = CodebaseRAG()
    response = codebase_rag.perform_rag(prompt)
    with st.chat_message("assistant"):
        streamed_response = st.write_stream(response)
    st.session_state.messages.append({"role": "assistant", "content": streamed_response})