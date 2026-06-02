import streamlit as st

_TYPE_ICONS = {
    "correction": "🚫",
    "preference": "⚙️",
    "fact": "📌",
    "decision": "✅",
    "episodic": "📖",
}


def _render_sidebar_memories(client):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Memory")

    try:
        data = client.list_memories()
    except Exception:
        st.sidebar.caption("Could not load memories.")
        return

    memories = data.get("memories", [])
    if not memories:
        st.sidebar.caption("No memories yet.")
        return

    grouped: dict[str, list[dict]] = {}
    for mem in memories:
        grouped.setdefault(mem["type"], []).append(mem)

    total = data.get("total", 0)
    st.sidebar.caption(f"{total} stored")

    for memory_type in ["correction", "preference", "fact", "decision", "episodic"]:
        rows = grouped.get(memory_type, [])
        if not rows:
            continue
        icon = _TYPE_ICONS.get(memory_type, "•")
        label = f"{icon} {memory_type.capitalize()} ({len(rows)})"
        with st.sidebar.expander(label, expanded=False):
            for mem in rows:
                st.markdown(
                    f"<div style='font-size:0.82rem;padding:2px 0;color:#374151'>{mem['content']}</div>",
                    unsafe_allow_html=True,
                )
                if memory_type != "episodic":
                    forget_col, _ = st.columns([1, 2])
                    with forget_col:
                        if st.button("Forget", key=f"sb-forget-{mem['id']}"):
                            client.forget_memory(mem["id"])
                            st.rerun()


def render_sidebar(client):
    st.sidebar.title("Engram")
    st.sidebar.caption("Single-user memory agent for Uddeshya")

    if st.sidebar.button("New Session", use_container_width=True):
        created = client.create_session()
        st.session_state.current_session_id = str(created["id"])
        st.session_state.messages = []
        st.session_state.view_mode = "chat"
        st.rerun()

    sessions = client.list_sessions()
    st.sidebar.subheader("Sessions")
    if not sessions:
        st.sidebar.info("No sessions yet.")
    else:
        for sess in sessions:
            title = sess.get("title") or f"Session {str(sess['id'])[:8]}"
            sid = str(sess["id"])
            if st.sidebar.button(title, key=f"session-{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.session_state.messages = client.get_messages(sid)
                st.session_state.view_mode = "chat"
                st.rerun()

    _render_sidebar_memories(client)
