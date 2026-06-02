import streamlit as st

from api_client import APIClient
from components.chat import render_chat
from components.sidebar import render_sidebar

API_BASE_URL = "http://127.0.0.1:8000"


def _inject_styles(sidebar_open: bool):
    sidebar_css = """
        section[data-testid="stSidebar"] {
            transform: translateX(0);
            visibility: visible;
        }
    """
    if not sidebar_open:
        sidebar_css = """
        section[data-testid="stSidebar"] {
            transform: translateX(-110%);
            visibility: hidden;
        }
        """

    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] {
            display: none;
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        .stDeployButton {
            display: none;
        }
        .stApp {
            background: #f7f8fb;
            color: #1f2937;
        }
        section[data-testid="stSidebar"] {
            background: #e9edf5;
            border-right: 1px solid #d4dbe8;
            transition: transform 0.2s ease-in-out;
            z-index: 1000;
        }
        section[data-testid="stSidebar"] > div:first-child {
            width: 290px !important;
        }
        .block-container {
            padding-top: 0.4rem;
            padding-bottom: 6.5rem;
            max-width: 1280px;
        }
        .msg-row {
            margin: 0.35rem 0;
            padding: 0.75rem 0.95rem;
            border-radius: 0.85rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .msg-user {
            background: #dbeafe;
            border: 1px solid #bfdbfe;
            margin-left: 12%;
        }
        .msg-assistant {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            margin-right: 12%;
        }
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-28%);
            width: min(760px, 68vw);
            z-index: 999;
            background: #f7f8fb;
            padding: 0.5rem 0 0.6rem 0;
            border-top: 1px solid #e5e7eb;
        }
        @media (max-width: 1100px) {
            div[data-testid="stChatInput"] {
                left: 50%;
                transform: translateX(-50%);
                width: min(760px, 92vw);
            }
        }
        button[kind="secondary"] {
            border-radius: 10px;
        }
        div[data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{sidebar_css}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Engram", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
    client = APIClient(API_BASE_URL)

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "chat"

    _inject_styles(st.session_state.sidebar_open)

    try:
        top_left, _ = st.columns([1, 10])
        with top_left:
            if st.button("☰", help="Toggle sidebar"):
                st.session_state.sidebar_open = not st.session_state.sidebar_open
                st.rerun()

        render_sidebar(client)
        render_chat(client)
    except Exception as exc:
        st.error(f"UI error: {exc}")
        st.info("Make sure API server is running at http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
