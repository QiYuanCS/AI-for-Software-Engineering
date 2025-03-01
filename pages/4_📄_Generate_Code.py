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
# Initialize session state variables
if 'code_review_history' not in st.session_state:
    st.session_state.code_review_history = []

if 'debug_history' not in st.session_state:
    st.session_state.debug_history = []

if 'modify_history' not in st.session_state:
    st.session_state.modify_history = []
# Show sidebar
helpers.sidebar.show()

#st.markdown("""
# #TODO

#Implement the following use cases
#and any other Python components or libraries of your choice, implement the following use cases on the "generate code" page of the application.  It is assumed and required that the application uses an LLM to assist behind-the-scenes.
#Using Streamlit,

#* Provide a feature to review code.  The use case is for a developer to provide some code, and to ask for a code review.


#* Provide a feature to debug code.  The use case is for a developer to provide some code, along with an optional error string, and to ask for help debugging the code, assuming that the error string was associated with execution of the code.


#* Provide a feature to modify or change the code using natural language conversationally.  The use case is for a developer to ask an LLM assistant to take some code, and some modification instructions.  The LLM assistant should provide modified code, and an explanation of the changes made.  Assuming the LLM is not perfect, the feature will allow the conversation to continue with more modification requests.

#* Provide a feature to reset the page, to allow all existing code and history to be cleared; this effectively starts a new conversation about possibly new code.
#""")


st.title("Generate Code")

tab1, tab2, tab3, tab4 = st.tabs(["Code Review", "Debug Code", "Modify Code", "Reset"])

with tab1:
    st.subheader("Code Review")
    code_review = st_ace(
        language=LANGUAGES[57],
        theme=THEMES[0],
        keybinding=KEYBINDINGS[0],
        height=300,
        key="code_review_editor"
    )

    if st.button("Get Code Review"):
        if not code_review.strip():
            st.error("Please provide code for review.")
        else:
            with st.spinner("Generating code review..."):
                try:
                    prompt = services.prompts.review_prompt(code_review)
                    response, _ = services.llm.converse_sync(prompt, [])
                    st.session_state.code_review_history.append({
                        "code": code_review,
                        "response": response
                    })
                    st.markdown("### Code Review")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error generating code review: {str(e)}")
                    st.error(traceback.format_exc())


with tab2:
    st.subheader("Debug Code")
    debug_code = st_ace(
        language=LANGUAGES[57],
        theme=THEMES[0],
        keybinding=KEYBINDINGS[0],
        height=300,
        key="debug_code_editor"
    )
    debug_error_string = st.text_input("Error String (optional)")

    if st.button("Debug Code"):
        if not debug_code.strip():
            st.error("Please provide code to debug.")
        else:
            with st.spinner("Debugging code..."):
                try:
                    prompt = services.prompts.debug_prompt(debug_error_string, debug_code)
                    response, _ = services.llm.converse_sync(prompt, [])
                    st.session_state.debug_history.append({
                        "code": debug_code,
                        "error_string": debug_error_string,
                        "response": response
                    })
                    st.markdown("### Debugging Result")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error debugging code: {str(e)}")
                    st.error(traceback.format_exc())


with tab3:
    st.subheader("Modify Code")
    modify_code = st_ace(
        language=LANGUAGES[57],
        theme=THEMES[0],
        keybinding=KEYBINDINGS[0],
        height=300,
        key="modify_code_editor"
    )
    modification_request = st.text_input("Modification Request")

    if st.button("Modify Code"):
        if not modify_code.strip() or not modification_request.strip():
            st.error("Please provide code and modification request.")
        else:
            with st.spinner("Modifying code..."):
                try:
                    prompt = services.prompts.modify_code_prompt(modification_request, modify_code)
                    response, _ = services.llm.converse_sync(prompt, [])
                    modified_code = services.extract.extract_delimited_content(response, output_delimiter="```", label="python")
                    explanation = response.split("【Explanation】")[1].split("【Questions】")[0].strip() if "【Explanation】" in response else "No explanation provided."
                    if modified_code:
                        st.session_state.modify_history.append({
                            "request": modification_request,
                            "response": response,
                            "old_code": modify_code,
                            "new_code": modified_code,
                            "explanation": explanation
                        })
                        st.markdown("### Modified Code")
                        st.code(modified_code, language='python')
                        st.markdown("### Explanation")
                        st.markdown(explanation)
                    else:
                        st.error("Could not extract modified code from response")
                except Exception as e:
                    st.error(f"Error modifying code: {str(e)}")
                    st.error(traceback.format_exc())

with tab4:
    st.subheader("Reset")
    if st.button("Reset All"):
        st.session_state.code_review_history = []
        st.session_state.debug_history = []
        st.session_state.modify_history = []
        st.session_state.modify_code = ""
        st.sidebar.success("All history and code have been reset")
