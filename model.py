import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.runnables import RunnableParallel , RunnablePassthrough, RunnableLambda
from youtube_transcript_api.proxies import WebshareProxyConfig

from dotenv import load_dotenv


load_dotenv()



if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "current_video" not in st.session_state:
    st.session_state.current_video = None


st.title("youtube transcript app")
video_input = st.text_input("Enter YouTube video ID")
#language = st.text_input("Enter language code (e.g., 'en' for English, 'hi' for Hindi)")
question = st.text_input("Enter your Query here")

def extract_video_id(url):
    pattern = re.search(r"v=([^&]+)",url)
    return pattern.group(1) if pattern else url




if video_input:
    video_id = extract_video_id(video_input)

    if st.session_state.current_video != video_id:
        st.session_state.vectorstore = None
        st.session_state.transcript = ""
        st.session_state.current_video = video_id

langs = ["en", "hi", "fr", "es", "de", "zh", "ja", "ru", "ar", "pt"]
if video_input :
        try:
            api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
        proxy_username="<proxy-username>",
        proxy_password="<proxy-password>",
    )
            )
            transcript_list = api.fetch(video_id , languages= langs)
            transcript = " ".join( chunk.text for chunk in transcript_list)
            st.session_state.transcript = transcript
            st.success("Transcript fetched successfully!")
          
        except Exception as e:
            st.write(f"error: {str(e)}")

transcript = st.session_state.get("transcript","")


splitter = RecursiveCharacterTextSplitter(chunk_size = 230 , chunk_overlap = 30)


@st.cache_resource
def load_model():
    return HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")


embedding_model = load_model()


if  st.session_state.vectorstore is None and transcript:
    chunks = splitter.create_documents([transcript])
    st.session_state.vectorstore = FAISS.from_documents(
            chunks, embedding_model
        )

vectorstore = st.session_state.vectorstore

if vectorstore is not None and question:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "conversational",
    temperature =0.5)

    model =   ChatHuggingFace(llm=llm)
    if any(word in question.lower() for word in ["summary", "summarize", "overview"]):
        prompt = f"""
        Summarize the following video transcript
        {transcript}
        """
        response = model.invoke(prompt)
        st.write(response.content)
    else:
       
        prompt = PromptTemplate(
                          template="""You are a helpful assistant,
    answer only from provided transcript context.
    if the contexxt is insufficient , just say you don't know.
    
    {context}
    Question :{question}""",

    input_variables=["context", "question"])
    final_prompt = prompt.format(context=context, question=question)
    
    response = model.invoke(final_prompt)
    st.write("Answer")
    st.write(response.content)