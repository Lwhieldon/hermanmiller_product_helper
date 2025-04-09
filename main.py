from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
from datetime import datetime
from tzlocal import get_localzone
import json
import os
import re
from backend.core import run_llm

# ------------------- Page Setup -------------------
st.set_page_config(
    page_title="HermanMiller Product Helper Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------- Session State -------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "image_page" not in st.session_state:
    st.session_state["image_page"] = 0
if "filter_images" not in st.session_state:
    st.session_state["filter_images"] = False

# ------------------- Utilities -------------------
def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%b %d, %Y • %I:%M %p").lstrip("0").replace(" 0", " ")

def get_local_time() -> datetime:
    return datetime.now(get_localzone())

def export_chat_history():
    history = []
    for user, bot in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
        history.append({
            "user_prompt": user["text"],
            "user_time": user["timestamp"],
            "bot_response": bot["answer"],
            "bot_time": bot["timestamp"],
            "images": bot.get("images", []),
            "sources": bot.get("sources", [])
        })
    return json.dumps(history, indent=2)

def display_sources(sources: list):
    if not sources:
        return
    with st.expander("🔎 Sources & Citations", expanded=False):
        for src in sources:
            part = ", ".join(src.get("part_numbers", []))
            page = src.get("page", "?")
            source = src.get("source", "Unknown PDF")
            if part:
                st.markdown(f"• **Part(s):** `{part}` — 📄 Page {page} — 📁 `{source}`")
            else:
                st.markdown(f"• 📄 Page {page} — 📁 `{source}`")

def display_images(images: list, part_numbers: list = []):
    if not images:
        st.info("No images were found for this query.")
        return

    filtered = []
    for img in images:
        if not st.session_state["filter_images"]:
            filtered.append(img)
        elif part_numbers:
            if any(pn.lower() in img.get("part_numbers", []) for pn in part_numbers):
                filtered.append(img)

    per_page = 4
    total = len(filtered)
    max_pages = (total - 1) // per_page + 1 if total > 0 else 1

    page = st.session_state["image_page"]
    page = max(0, min(page, max_pages - 1))
    start = page * per_page
    end = start + per_page

    if total == 0:
        st.warning("No images matched your filter.")
        return

    st.markdown("**📸 Product Images:**")
    rows = filtered[start:end]
    cols = st.columns(len(rows))
    for img_data, col in zip(rows, cols):
        with col:
            st.image(img_data["path"], caption=img_data["caption"], use_column_width=True)
            meta = f"📄 Page {img_data.get('page')} — 📁 {img_data.get('source')}"
            st.caption(meta)

    # Pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if page > 0:
            if st.button("⬅️ Prev", key="prev"):
                st.session_state["image_page"] -= 1
    with col2:
        st.markdown(f"<div style='text-align:center;'>Page {page + 1} of {max_pages}</div>", unsafe_allow_html=True)
    with col3:
        if page < max_pages - 1:
            if st.button("Next ➡️", key="next"):
                st.session_state["image_page"] += 1

# ------------------- Sidebar -------------------
st.sidebar.markdown("### Export Chat History")
if st.session_state["user_prompt_history"]:
    export_data = export_chat_history()
    st.sidebar.download_button(
        label="📥 Download as JSON",
        data=export_data,
        file_name="chat_history.json",
        mime="application/json"
    )
else:
    st.sidebar.info("No chat history yet.")

# ------------------- Main UI -------------------
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 30px;'>HermanMiller Product Helper Bot</h1>",
    unsafe_allow_html=True,
)

st.markdown("### Prompt")
col1, col2 = st.columns([5, 1])
with col1:
    prompt = st.text_input("Enter your message here...", label_visibility="collapsed")
with col2:
    submit = st.button("Submit")

# ------------------- Process Prompt -------------------
if 'prompt' in locals() and submit and prompt:
    timestamp = format_timestamp(get_local_time())
    with st.spinner("Generating response..."):
        try:
            llm_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            processed_response = {
                "answer": llm_response["answer"],
                "images": llm_response.get("images", []),
                "sources": llm_response.get("sources", [])
            }

            st.markdown("---")
            st.markdown("**Response:**")
            st.write(processed_response["answer"])

            part_numbers = []
            for word in prompt.split():
                if re.match(r"[A-Z]{2}\d{3,4}", word.strip().upper()):
                    part_numbers.append(word.strip().upper())

            st.checkbox("Only show part-matched images", value=False, key="filter_images")
            display_images(processed_response["images"], part_numbers)
            display_sources(processed_response["sources"])

            st.session_state["user_prompt_history"].append({
                "text": prompt,
                "timestamp": timestamp
            })
            st.session_state["chat_answers_history"].append({
                "answer": processed_response["answer"],
                "images": processed_response["images"],
                "sources": processed_response["sources"],
                "timestamp": timestamp
            })
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", processed_response["answer"]))
            st.session_state["image_page"] = 0  # reset pagination

        except Exception as e:
            st.error(f"An error occurred while generating a response: {e}")

# ------------------- Chat History -------------------
if st.session_state["chat_answers_history"]:
    st.markdown("---")
    for i in range(len(st.session_state["chat_answers_history"]) - 1, -1, -1):
        user_msg = st.session_state["user_prompt_history"][i]
        bot_msg = st.session_state["chat_answers_history"][i]

        message(f"{user_msg['text']}  \n\n*{user_msg['timestamp']}*", is_user=True)
        message(f"{bot_msg['answer']}  \n\n*{bot_msg['timestamp']}*")
        display_images(bot_msg.get("images", []))
        display_sources(bot_msg.get("sources", []))

# ------------------- Footer -------------------
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>App Powered by LangChain and Streamlit</p>",
    unsafe_allow_html=True,
)
