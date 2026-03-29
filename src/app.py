import streamlit as st
from chatbot import chat_with_constitution

st.set_page_config(
    page_title="Indian Constitution AI",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ Indian Constitution Legal Assistant Chatbot")
st.caption("Answers strictly from the Indian Constitution dataset")

# Session chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask your legal question...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Constitution..."):
            try:
                answer = chat_with_constitution(query)
            except Exception as e:
                answer = "⚠️ Server not responding. Please ensure Ollama & LiteLLM are running."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
