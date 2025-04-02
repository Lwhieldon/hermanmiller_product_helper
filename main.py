from dotenv import load_dotenv
load_dotenv()

from typing import List
import streamlit as st
from streamlit_chat import message
from PIL import Image
import requests
from io import BytesIO

from backend.core import run_llm

st.set_page_config(
    page_title="Your App Title",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def create_sources_string(source_docs: List[dict]) -> str:
    if not source_docs:
        return ""
    sources_string = "Sources:\n"
    seen = set()
    for doc in source_docs:
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page_number", "N/A")
        element_type = metadata.get("element_type", "").title()
        key = (source, page, element_type)
        if key not in seen:
            seen.add(key)
            sources_string += f"{len(seen)}. Page {page} [{element_type}] from {source}\n"
    return sources_string

def get_profile_picture(email):
    gravatar_url = f"https://www.gravatar.com/avatar/{hash(email)}?d=identicon&s=200"
    response = requests.get(gravatar_url)
    img = Image.open(BytesIO(response.content))
    return img

# Sidebar user profile
with st.sidebar:
    st.title("User Profile")
    user_name = "Lee Whieldon"
    user_email = "lwhieldon1@gmail.com"

    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

    if st.button("Clear chat"):
        st.session_state["chat_answers_history"] = []
        st.session_state["user_prompt_history"] = []
        st.session_state["chat_history"] = []

st.header("Herman Miller Product Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Chat UI
col1, col2 = st.columns([2, 1])
with col1:
    st.text_input("Prompt", key="user_prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = st.session_state["user_prompt"] or "Hello"
        st.session_state["submitted_prompt"] = prompt

if "submitted_prompt" in st.session_state:
    prompt = st.session_state.pop("submitted_prompt")
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n{create_sources_string(generated_response['source_documents'])}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append({
            "answer": generated_response["result"],
            "sources": generated_response["source_documents"]
        })
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

if st.session_state["chat_answers_history"]:
    for chat_data, user_query in reversed(list(zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ))):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(chat_data["answer"])
        with st.expander("Show sources and metadata"):
            for doc in chat_data["sources"]:
                metadata = doc.get("metadata", {})
                page = metadata.get("page_number", "None")
                source = metadata.get("source", "Unknown source")
                st.markdown(f"**Page {page}, From {source}")
                content = doc.get("page_content", "")
                st.code(content[:500] + "..." if len(content) > 500 else content)

st.markdown("---")
st.markdown("App Powered by LangChain and Streamlit")