import asyncio
import traceback

import streamlit as st
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES

import helpers.sidebar
import helpers.util
import services.extract
import services.llm
import services.prompts
from helpers import util

st.set_page_config(
    page_title="Generate Code",
    page_icon="📄",
    layout="wide"
)

# Add comments to explain the purpose of the code sections

# Show sidebar
helpers.sidebar.show()

st.markdown("""
# TODO

## Implement the following use cases

Using Streamlit, and any other Python components or libraries of your choice, implement the following use cases on the "generate code" page of the application.  It is assumed and required that the application uses an LLM to assist behind-the-scenes.

* Provide a feature to review code.  The use case is for a developer to provide some code, and to ask for a code review.


* Provide a feature to debug code.  The use case is for a developer to provide some code, along with an optional error string, and to ask for help debugging the code, assuming that the error string was associated with execution of the code.


* Provide a feature to modify or change the code using natural language conversationally.  The use case is for a developer to ask an LLM assistant to take some code, and some modification instructions.  The LLM assistant should provide modified code, and an explanation of the changes made.  Assuming the LLM is not perfect, the feature will allow the conversation to continue with more modification requests.

* Provide a feature to reset the page, to allow all existing code and history to be cleared; this effectively starts a new conversation about possibly new code.
""")
