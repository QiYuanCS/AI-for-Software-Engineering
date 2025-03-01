import streamlit as st
from services import prompts
from helpers import util

st.set_page_config(
    page_title="Quick Chat",
    page_icon="💬",
    layout="wide"
)

import helpers.sidebar
import asyncio

helpers.sidebar.show()

st.header("Quick Chat")
st.write("Get instant answers to your software development and coding questions.")
ask_book = st.checkbox("Ask the Pragmatic Programmer book?", value=False)

# Ensure the session state is initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Print all messages in the session state
for message in [m for m in st.session_state.messages if m["role"] != "system"]:
    avatar = "🔎" if message["role"] == "evidence" else None
    if avatar:
        with st.chat_message(message["role"], avatar=avatar):
            page_number = message["page_number"]
            with st.expander(f"See page {page_number}", expanded=False):
                st.write(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to the user prompt
if prompt := st.chat_input("Ask a software development or coding question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if ask_book:
        # Functionality 1: Use The Pragmatic Programmer as context
        asyncio.run(util.ask_book(st.session_state.messages, prompt))
        st.rerun()
    else:
        # Functionality 2: Standard LLM conversation
        # Filter messages to only include "user" and "assistant" roles for LLM
        filtered_messages = [m for m in st.session_state.messages if m["role"] in ["user", "assistant"]]
        asyncio.run(util.chat(filtered_messages, prompt))
        st.rerun()
