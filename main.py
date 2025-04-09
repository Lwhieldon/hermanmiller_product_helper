from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
import hashlib
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
from tzlocal import get_localzone
import json
import os
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
if "show_all_sources" not in st.session_state:
    st.session_state["show_all_sources"] = False

# ------------------- Helper Functions -------------------
def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%b %d, %Y • %I:%M %p").lstrip("0").replace(" 0", " ")

def get_local_time() -> datetime:
    local_tz = get_localzone()
    return datetime.now(local_tz)

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

def display_images(images: list):
    if not images:
        st.info("No images were found for this query.")
        return
    st.markdown("**📸 Product Images:**")
    rows = [images[i:i + 4] for i in range(0, len(images), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for img_data, col in zip(row, cols):
            path = img_data.get("path")
            caption = img_data.get("caption", "Product image")
            page = img_data.get("page")
            source = img_data.get("source")
            meta = f"📄 Page {page} | 📁 {source}" if page and source else ""
            if os.path.exists(path):
                with col:
                    st.image(path, caption=caption, use_column_width=True)
                    if meta:
                        st.caption(meta)
            else:
                with col:
                    st.warning(f"Missing image: {caption}")

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

# ------------------- Export History -------------------
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

try:
    st.markdown("### Prompt")
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input("Enter your message here...", label_visibility="collapsed")
    with col2:
        submit = st.button("Submit")
except st.runtime.scriptrunner.script_run_context.StopException:
    st.stop()

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
            display_images(processed_response["images"])
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
