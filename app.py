import os
import vertexai
import streamlit as st

from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatVertexAI

from loguru import logger
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory

def get_text(docs):
    """
    사용자가 업로드한 문서들 각각에 적합한 document loader를 이용해서
    각 문서에서 text를 추출하여 추출한 텍스트 전체를 반환
    """
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    """
    document loader로 읽어온 문서를 청크로 분할반환
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Google Cloud VertexAI embedding 모델을 이용해서
    청크로 분할한 문서를 임베딩 시키고
    FAISS(Facebook AI Similarity Search) 라이브러리를 이용해서 
    유사성 검색을 할 수 있는 vector db를 만들어 반환
    """
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(model, vector_db):
    """
    새로운 질문이 검색을 거쳐 chat history를 기반으로 답변을 하는 체인반환
    """
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        chain_type="stuff", 
        retriever=vector_db.as_retriever(search_type = 'mmr', vervose = True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose = True
    )
    return conversation_chain

def main():
    ### TODO VertexAI API Setup
    """
    Google의 VertexAI API를 사용하기 위해서 아래의 두 가지가 필요하다.
    1. Google Cloud Project ID
    2. Application Default Credentials (ADC)
    Reference: https://yunwoong.tistory.com/297
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your/google/application/credentials.json"
    PROJECT_ID = "your_project_id"                     # Google Cloud 프로젝트 ID
    LOCATION = "your_google_cloud_service_location"    # Google Cloud 서비스 지역: 'asia-northeast3', 'europe-west1' ..
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    ### App title
    st.set_page_config(
        page_title="Palm2-QA-chat",
        page_icon=":palm_tree:", 
        layout="wide",
        initial_sidebar_state="expanded")
    
    ### Initialize Session
    if "conversation" not in st.session_state:      # Langchain의 ConversationalRetrievalChain
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:   # 업로드한 파일의 처리여부
        st.session_state.processComplete = None
    if "messages" not in st.session_state:          # assistant와 user의 대화내용 누적
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 궁금한 점은 무엇이든 물어보세요!"}]


    ### Sidebar
    with st.sidebar:
        st.title(':palm_tree: Palm-2 QA Chatbot')
        st.caption('이 챗봇은 :blue[G]:red[o]:orange[o]:blue[g]:green[l]:red[e]의 PaLM2와 함께합니다.')

        selected_model = st.sidebar.selectbox('사용할 모델을 선택하세요.', ['PaLM-2', 'Gemini (beta)'], key='selected_model')
        uploaded_files = st.file_uploader("업로드할 파일을 선택하세요.", type=['pdf','docx'], accept_multiple_files=True)
        
        ## 선택한 모델로드
        if selected_model == "PaLM-2":
            model = ChatVertexAI()
        elif selected_model == "Gemini (beta)":
            model = ChatVertexAI()  ## 수정필요

        ## 업로드한 파일처리
        if uploaded_files:
            if st.button("Process"):
                files_text = get_text(uploaded_files)
                text_chunks = get_text_chunks(files_text)
                vector_db = get_vector_store(text_chunks)

                st.session_state.conversation = get_conversation_chain(model, vector_db)
                st.session_state.processComplete = True

    ### Main Page
    # st.session_state
    ## display    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    ## 사용자 입력창
    if prompt := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    ## 사용자 입력에 대한 모델답변 생성
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner(f"{selected_model} 모델이 열심히 답변을 완성하고 있습니다! 잠시만 기다려주세요."):
                result = chain({"question": prompt})
                response = result['answer']
                source_documents = result['source_documents']
                chat_history = result['chat_history']

            st.write(response)
            with st.expander("참고문서 확인하기"):
                st.markdown(f"{source_documents[0].metadata['source']} page: {source_documents[0].metadata['page']}", help = source_documents[0].page_content)
                st.markdown(f"{source_documents[1].metadata['source']} page: {source_documents[1].metadata['page']}", help = source_documents[1].page_content)
                st.markdown(f"{source_documents[2].metadata['source']} page: {source_documents[2].metadata['page']}", help = source_documents[2].page_content)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history = chat_history

if __name__ == '__main__':
    main()