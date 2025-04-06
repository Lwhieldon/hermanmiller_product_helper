# main.py

import streamlit as st
import pytz
from datetime import datetime
from tzlocal import get_localzone
from backend.core import run_llm, process_response
import hashlib
import requests
from PIL import Image
from io import BytesIO
import os
import json

# ---------------------------------------
# Page Setup
# ---------------------------------------
st.set_page_config(page_title="Herman Miller Product Helper", layout="wide")
st.title("🪑 Herman Miller Product Assistant")

# ---------------------------------------
# Time Utilities
# ---------------------------------------
def get_local_time():
    return datetime.now(get_localzone())

def format_timestamp(dt):
    return dt.strftime("%b %d, %Y • %I:%M %p").lstrip("0").replace(" 0", " ")

# ---------------------------------------
# Session State
# ---------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "show_all_sources" not in st.session_state:
    st.session_state["show_all_sources"] = False

# ---------------------------------------
# Gravatar
# ---------------------------------------
def get_profile_picture(email: str):
    email_hash = hashlib.md5(email.strip().lower().encode()).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img

# ---------------------------------------
# Sidebar
# ---------------------------------------
with st.sidebar:
    st.title("User Profile")
    user_name = "Lee Whieldon"
    user_email = "lwhieldon1@gmail.com"
    try:
        profile_pic = get_profile_picture(user_email)
        st.image(profile_pic, width=150)
    except Exception:
        st.warning("Unable to load profile image.")

    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** [{user_email}](mailto:{user_email})")

    st.markdown("---")
    st.markdown("### Export Chat History")
    if st.session_state["user_prompt_history"]:
        export_data = [
            {
                "user_prompt": u["text"],
                "user_time": u["timestamp"],
                "bot_response": a["answer"],
                "bot_time": a["timestamp"],
                "sources": a.get("sources", []),
                "images": a.get("images", [])
            }
            for u, a in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"])
        ]
        st.download_button("📥 Download as JSON", data=json.dumps(export_data, indent=2),
                           file_name="chat_history.json", mime="application/json")
    else:
        st.info("No chat history yet.")

# ---------------------------------------
# Chat Input UI
# ---------------------------------------
st.markdown("### Ask about a product")
col1, col2 = st.columns([5, 1])
with col1:
    prompt = st.text_input("Enter your message...", label_visibility="collapsed")
with col2:
    submit = st.button("Submit")

# ---------------------------------------
# Display Image Carousel
# ---------------------------------------
def display_images(images: list):
    if not images:
        return
    st.markdown("**📸 Product Illustrations**")
    rows = [images[i:i+4] for i in range(0, len(images), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for img_data, col in zip(row, cols):
            if os.path.exists(img_data["path"]):
                with col:
                    with st.expander(img_data.get("caption", "Image")):
                        st.image(img_data["path"], use_column_width=True)
            else:
                col.warning(f"Missing image: {img_data['path']}")

# ---------------------------------------
# Handle Submission
# ---------------------------------------
if prompt and submit:
    timestamp = format_timestamp(get_local_time())
    with st.spinner("Thinking..."):
        try:
            llm_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            processed = process_response(llm_response)

            # Save chat state
            st.session_state["user_prompt_history"].append({
                "text": prompt,
                "timestamp": timestamp
            })
            st.session_state["chat_answers_history"].append({
                "answer": processed["answer"],
                "sources": processed["sources"],
                "images": processed["images"],
                "timestamp": timestamp
            })
            st.session_state["chat_history"].append(("human", prompt))
            st.session_state["chat_history"].append(("ai", processed["answer"]))

        except Exception as e:
            st.error(f"Error generating response: {e}")

# ---------------------------------------
# Chat Display
# ---------------------------------------
if st.session_state["chat_answers_history"]:
    for i in range(len(st.session_state["chat_answers_history"]) - 1, -1, -1):
        user_msg = st.session_state["user_prompt_history"][i]
        bot_msg = st.session_state["chat_answers_history"][i]

        st.markdown(f"**🧑 You**: {user_msg['text']}  \n*{user_msg['timestamp']}*")
        st.markdown(f"**🤖 Assistant**:  \n{bot_msg['answer']}  \n*{bot_msg['timestamp']}*")

        display_images(bot_msg.get("images", []))

        if bot_msg["sources"]:
            with st.expander("📚 Sources"):
                for src in bot_msg["sources"]:
                    pg = src.get("page", "Unknown")
                    heading = src.get("heading") or src.get("prev_heading") or "Section"
                    st.markdown(f"- Page {pg}: *{heading}*")

# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("<hr><p style='text-align:center; color:gray;'>Powered by LangChain + Streamlit</p>", unsafe_allow_html=True)
