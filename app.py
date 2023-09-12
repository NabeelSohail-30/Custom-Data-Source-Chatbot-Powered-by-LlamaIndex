import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with the Custom docs, powered by LlamaIndex",
                   page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)


api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

# openai.api_key = st.secrets.openai_key
# openai.api_key = ""

if api_key:
    openai.api_key = api_key
else:
    st.sidebar.write("Please enter your OpenAI API key above")
    st.stop()


st.title("Chat with the Custom docs, powered by LlamaIndex 💬🦙")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about ..."}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(
            model="gpt-3.5-turbo", temperature=0.5, system_prompt="System Prompt will come here"))
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context)
        return index


index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# Prompt for user input and save to chat history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            # Add response to message history
            st.session_state.messages.append(message)
