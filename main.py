from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
import hashlib
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
import pytz
from tzlocal import get_localzone
import json
import os
from backend.core import run_llm


# ------------------- Page Setup -------------------
st.set_page_config(
    page_title="HermanMiller Product Helper Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
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
if "allow_similar_images" not in st.session_state:
    st.session_state["allow_similar_images"] = False


# ------------------- Helper Functions -------------------
def get_profile_picture(email: str) -> Image.Image:
    email_hash = hashlib.md5(email.strip().lower().encode()).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img

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
            "images": bot.get("images", [])
        })
    return json.dumps(history, indent=2)

def display_images(images: list):
    if not images:
        st.info("No images were found for this query.")
        return

    if not st.session_state.get("allow_similar_images", False):
        # Filter out related illustrations unless explicitly allowed
        images = [img for img in images if "caption" in img and "related" not in img["caption"].lower()]

    if not images:
        st.info("No matching illustrations found for this part. Try enabling similar results.")
        return

    st.markdown("**📸 Product Images:**")
    rows = [images[i:i + 4] for i in range(0, len(images), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for img_data, col in zip(row, cols):
            path = img_data.get("path")
            caption = img_data.get("caption", "Product image")
            page = img_data.get("page")
            full_caption = f"{caption} (Page {page})" if page else caption

            if os.path.exists(path):
                with col:
                    with st.expander(full_caption):
                        st.image(path, use_column_width=True)
            else:
                with col:
                    st.warning(f"Missing image: {full_caption}")


# ------------------- Sidebar -------------------
with st.sidebar:
    st.title("User Profile")
    user_name = "Lee Whieldon"
    user_email = "lwhieldon1@gmail.com"
    try:
        profile_pic = get_profile_picture(user_email)
        st.image(profile_pic, width=150)
    except Exception:
        st.warning("Could not load profile image.")
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** [{user_email}](mailto:{user_email})")

    st.markdown("---")
    st.markdown("### Export Chat History")
    if st.session_state["user_prompt_history"]:
        export_data = export_chat_history()
        st.download_button(
            label="📥 Download as JSON",
            data=export_data,
            file_name="chat_history.json",
            mime="application/json"
        )
    else:
        st.info("No chat history yet.")

    st.markdown("---")
    st.sidebar.checkbox("🔄 Show similar illustrations when exact match not found", key="allow_similar_images")


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
                "images": llm_response.get("images", [])
            }

            st.markdown("---")
            st.markdown("**Response:**")
            st.write(processed_response["answer"])

            display_images(processed_response["images"])

            st.session_state["user_prompt_history"].append({
                "text": prompt,
                "timestamp": timestamp
            })
            st.session_state["chat_answers_history"].append({
                "answer": processed_response["answer"],
                "images": processed_response["images"],
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


# ------------------- Footer -------------------
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>App Powered by LangChain and Streamlit</p>",
    unsafe_allow_html=True,
)
