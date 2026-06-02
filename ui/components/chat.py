import streamlit as st


def _render_message(role: str, content: str) -> None:
    css_class = "msg-user" if role == "user" else "msg-assistant"
    st.markdown(
        f'<div class="msg-row {css_class}">{content}</div>',
        unsafe_allow_html=True,
    )


def render_chat(client):
    st.markdown("## Chat")
    session_id = st.session_state.get("current_session_id")
    if not session_id:
        st.info("Create or select a session from the sidebar.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = client.get_messages(session_id)

    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.caption(f"Session: `{session_id}`")
    with top_col2:
        if st.button("End Session", use_container_width=True):
            client.end_session(session_id, generate_episodic=True)
            st.success("Session ended. Episodic memory generation triggered.")

    for message in st.session_state.messages:
        _render_message(message["role"], message["content"])

    user_prompt = st.chat_input("Send a message")
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    _render_message("user", user_prompt)

    placeholder = st.empty()
    chunks = []
    for chunk in client.stream_chat(session_id, user_prompt):
        chunks.append(chunk)
        placeholder.markdown(
            f'<div class="msg-row msg-assistant">{"".join(chunks)}</div>',
            unsafe_allow_html=True,
        )
    assistant_text = "".join(chunks)

    if assistant_text.strip():
        st.session_state.messages.append({"role": "assistant", "content": assistant_text.strip()})
