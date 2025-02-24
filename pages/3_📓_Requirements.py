import asyncio

import streamlit as st
import helpers.sidebar
import helpers.util
from services.prompts import requirements_prompt, system_requirements_prompt
import services.llm

st.set_page_config(
    page_title="Requirements",
    page_icon="📓",
    layout="wide"
)

st.header("Requirements")

helpers.sidebar.show()

st.markdown("<br>", unsafe_allow_html=True)

product_name = st.text_input("What is the name of your product?", placeholder="Enter the name of your product here.")

product_description = st.text_area("Describe your product.", placeholder="Enter a description of your product here.")

requirement_type = st.selectbox("What type of requirement document should we generate?", ["Business Problem Statement", "Vision Statement", "Ecosystem map", "RACI Matrix"])

generate_button = st.button("Generate document&nbsp;&nbsp;➠", type="primary")

if generate_button:
    advice = st.empty()
    spinner_placeholder = st.empty()

    messages = []
    system_requirements_prompt = system_requirements_prompt(product_name, product_description)
    messages.append({"role": "system",
                     "content": system_requirements_prompt})

    prompt = requirements_prompt(product_name, requirement_type)
    messages.append({"role": "user", "content": prompt})
    with spinner_placeholder:
        with st.spinner("Receiving response..."):
            messages, full_response = asyncio.run(helpers.util.run_conversation(messages, advice))
    advice.empty()  # Clear the streamed chunks
    spinner_placeholder.empty()  # Clear the spinner
    st.write(full_response)

