import streamlit as st
import requests
import os 
from dotenv import load_dotenv
load_dotenv()


AWS_URL = os.environ.get("AWS_URL")


st.set_page_config(
    page_title="LLM Chat",
    page_icon="🤖",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.sidebar:
    st.title("⚙️ Controls")

    if st.button("🔄 Update Vector Store"):
        with st.spinner("Updating vector store..."):
            try:
                res = requests.get(AWS_URL + "/vectore_update")

                if res.status_code == 200:
                    st.success("Vector store updated successfully!")
                else:
                    st.error("Vector update failed.")

            except Exception as e:
                st.error(f"Error: {e}")


st.title("💬 LLM Chat Application")


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

def stream_data(prompt):

    payload = {
        "query": prompt,
        "message_history": st.session_state.chat_history[:10]
    }

    response = requests.post(
        AWS_URL + "/chat",
        json=payload,
        stream=True
    )

    for chunk in response.iter_content(chunk_size=1):

        if chunk:
            yield chunk.decode("utf-8")


if prompt := st.chat_input("Ask something..."):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response_text = st.write_stream(
                    stream_data(prompt)
                )
            except Exception as e:
                st.error(f"API Error: {e}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )

    st.session_state.chat_history.append(
        {
            "USER": prompt,
            "ASSISTANT": response_text
        }
    )