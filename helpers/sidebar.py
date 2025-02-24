import streamlit as st


def show() -> None:
    with st.sidebar:
        st.markdown(f"""
            <a href="/" style="color:black;text-decoration: none;">
                <div style="display:table;margin-top:-21rem;margin-left:0%;">
                    <img src="app/static/logo.png" width="80"><span style="color: white">&nbsp;Ducky</span>
                    <span style="font-size: 0.8em; color: grey">&nbsp;&nbsp;v0.1.2</span>
                </div>
            </a>
            <br>
                """, unsafe_allow_html=True)

        reload_button = st.button("↪︎  Reload Page")
        if reload_button:
            st.session_state.clear()
            st.rerun()
