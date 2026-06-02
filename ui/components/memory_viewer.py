import streamlit as st


def render_memory_viewer(client):
    st.header("Memories")
    data = client.list_memories()
    memories = data.get("memories", [])
    if not memories:
        st.info("No memories stored yet.")
        return

    grouped: dict[str, list[dict]] = {}
    for mem in memories:
        grouped.setdefault(mem["type"], []).append(mem)

    for memory_type in ["correction", "preference", "fact", "decision", "episodic"]:
        rows = grouped.get(memory_type, [])
        if not rows:
            continue
        st.subheader(memory_type.capitalize())
        for mem in rows:
            cols = st.columns([6, 1, 1])
            with cols[0]:
                st.markdown(mem["content"])
                st.caption(f"id: {mem['id']} • confidence: {mem['confidence']:.2f}")
            if memory_type == "episodic":
                with cols[1]:
                    st.caption("read-only")
                continue
            with cols[1]:
                if st.button("Edit", key=f"edit-{mem['id']}"):
                    st.session_state[f"editing_{mem['id']}"] = True
            with cols[2]:
                if st.button("Forget", key=f"forget-{mem['id']}"):
                    client.forget_memory(mem["id"])
                    st.rerun()

            if st.session_state.get(f"editing_{mem['id']}", False):
                new_content = st.text_area(
                    "New content",
                    value=mem["content"],
                    key=f"edit_input_{mem['id']}",
                )
                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button("Save", key=f"save-{mem['id']}"):
                        client.update_memory(mem["id"], new_content.strip())
                        st.session_state[f"editing_{mem['id']}"] = False
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel-{mem['id']}"):
                        st.session_state[f"editing_{mem['id']}"] = False
                        st.rerun()
