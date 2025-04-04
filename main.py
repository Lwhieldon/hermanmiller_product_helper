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
from backend.core import run_llm, process_response


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
            "sources": bot["sources"]
        })
    return json.dumps(history, indent=2)


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


# ------------------- Main Interface -------------------
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 30px;'>HermanMiller Product Helper Bot</h1>",
    unsafe_allow_html=True,
)

# Use a safe structure for layout blocks
try:
    st.markdown("### Prompt")
    col1, col2 = st.columns([5, 1])
    with col1:
        prompt = st.text_input("Enter your message here...", label_visibility="collapsed")
    with col2:
        submit = st.button("Submit")
except st.runtime.scriptrunner.script_run_context.StopException:
    st.stop()


# ------------------- Handle Input -------------------
if 'prompt' in locals() and submit and prompt:
    timestamp = format_timestamp(get_local_time())

    with st.spinner("Generating response..."):
        try:
            llm_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
            processed_response = process_response(llm_response)

            st.markdown("---")
            st.markdown("**Response:**")
            st.write(processed_response["answer"])

            with st.expander("Show sources"):
                for source in processed_response["sources"]:
                    source_text = f"Page {source['page']}, from {source['source']}"
                    if source["prev_heading"]:
                        source_text += f", (Previous Heading: {source['prev_heading']})"
                    st.write(source_text)

            # Save chat with timestamp
            st.session_state["user_prompt_history"].append({
                "text": prompt,
                "timestamp": timestamp
            })
            st.session_state["chat_answers_history"].append({
                "answer": processed_response["answer"],
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

        if st.session_state["show_all_sources"]:
            with st.expander(f"Show sources for answer #{i + 1}"):
                for source in bot_msg["sources"]:
                    source_text = f"Page {source['page']}, from {source['source']}"
                    if source["prev_heading"]:
                        source_text += f", (Previous Heading: {source['prev_heading']})"
                    st.write(source_text)


# ------------------- Footer -------------------
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>App Powered by LangChain and Streamlit</p>",
    unsafe_allow_html=True,
)
