import streamlit as st
import glob
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain import hub


# API KEY
load_dotenv()

st.title("성우봇 👍")

if "messages" not in st.session_state:
    # 대화기록 저장
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    prompt_files = glob.glob("prompts/*.yaml")

    selected_prompt = st.selectbox("프롬프트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화 출력
def print_messages():
    for chat_messages in st.session_state["messages"]:
        st.chat_message(chat_messages.role).write(chat_messages.content)


def creat_chain(prompt_file_path, task=None):

    prompt = load_prompt(prompt_file_path, encoding="utf-8")

    if task:
        prompt = prompt.partial(task=task)

    # if prompt_type == "SNS 게시글":
    #     prompt = load_prompt(path="prompts/sns.yaml", encoding="utf8")

    # elif prompt_type == "요약":
    #     prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


# 새로운 대화 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자의 입력이 들어오면
if user_input:
    # 사용자 입력
    st.chat_message("user").write(user_input)

    # chain 생성
    chain = creat_chain(selected_prompt, task=task_input)
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)를 만들어 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        answer = ""

        for token in response:
            answer += token
            container.markdown(answer)

    # 대화 기록 저장
    add_message("user", user_input)
    add_message("assistant", answer)
