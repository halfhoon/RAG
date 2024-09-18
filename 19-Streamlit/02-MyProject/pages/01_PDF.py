import streamlit as st
import glob
import os
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS


# API KEY
load_dotenv()

# langsmith 추적
os.environ["LANGCHAIN_PROJECT"] = "[Project] PDF RAG"

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("PDF 기반 QA 💬")

# 처음 한 번만 실행
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    selected_mdoel = st.selectbox(
        "Choose LLM", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 이전 대화 출력
def print_messages():
    for chat_messages in st.session_state["messages"]:
        st.chat_message(chat_messages.role).write(chat_messages.content)


# 파일을 캐시 저장(시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    retriever = vectorstore.as_retriever()

    return retriever


# 체인 생성
def creat_chain(retriever, model_name):

    # prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")   /// COD

    # launching은 main에서 이루어지므로, root는 main의 위치
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    llm = ChatOpenAI(model_name=model_name, temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file)
    chain = creat_chain(retriever, model_name=selected_mdoel)
    st.session_state["chain"] = chain


# 새로운 대화 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


if clear_btn:

    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메세지 출력
warning_msg = st.empty()


# 사용자의 입력이 들어오면
if user_input:

    # chain 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 입력
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)

        with st.chat_message("assistant"):
            container = st.empty()
            answer = ""

            for token in response:
                answer += token
                container.markdown(answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", answer)

    else:
        warning_msg.error("파일을 업로드 해 주세요 ⚠️")
