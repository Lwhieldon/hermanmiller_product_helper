from dotenv import load_dotenv

load_dotenv()
from typing import Set

import streamlit as st
from streamlit_chat import message
# from langchain.globals import set_debug 
# set_debug(True)  -- Uncomment this line to enable debug mode in terminal

from backend.core import run_llm

st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Add these imports
from PIL import Image
import requests
from io import BytesIO


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


# Add this function to get a profile picture
def get_profile_picture(email):
    # This uses Gravatar to get a profile picture based on email
    # You can replace this with a different service or use a default image
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img

# Sidebar user information
with st.sidebar:
    st.title("User Profile")

    # You can replace these with actual user data
    user_name = "Lee Whieldon"
    user_email = "lwhieldon1@gmail.com"

    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("Herman Miller Product Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Create two columns for a more modern layout
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"  # Default message if input is empty

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        sources = set(doc.metadata["source"] for doc in generated_response["source_documents"])
        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

# Display chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in reversed(list(zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ))):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)


# Add a footer
st.markdown("---")
st.markdown("App Powered by LangChain and Streamlit")