#前端
import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

st.title("📑 AI智能PDF问答工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入您的OpenAI API密钥",type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")

#AI需要了解和用户之前的对话，需要对记忆进行初始化
if "memory" not in st.session_state:
    st.session_state["memort"] = ConversationBufferMemory(
        return_messages = True,
        memory_key = "chat_history",
        outpuy_key = "answer"
    )
#上传文件并提问
uploaded_file = st.file_uploader("请上传您的PDF文件：",type="pdf")
question = st.text_input("请对pdf的内容进行提问",disabled=not uploaded_file)
#情况判断
if uploaded_file and question and not openai_api_key:
    st.info("请输入你的OpenAI API密钥")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        response =qa_agent(openai_api_key,st.session_state["memory"],
                           uploaded_file,question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"]=response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                        st.divider()