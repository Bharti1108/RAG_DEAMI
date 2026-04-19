from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from dotenv import load_dotenv
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableParallel , RunnablePassthrough, RunnableLambda




load_dotenv()




st.title("youtube transcript app")
video_input = st.text_input("Enter YouTube video ID")
language = st.text_input("Enter language code (e.g., 'en' for English, 'hi' for Hindi)")
question = st.text_input("Enter your Query here")
def extract_video_id(url):
    pattern = re.search(r"v=([^&]+)",url)
    return pattern.group(1) if pattern else url
transcript = " "
if video_input:
    video_id = extract_video_id(video_input)

    if language:
        langs = [language]
    else:
        langs = ["en", "hi", "fr", "es", "de", "zh", "ja", "ru", "ar", "pt"]
    
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id , languages= langs)
        transcript = " ".join( chunk.text for chunk in transcript_list)
        st.success("Transcript fetched successfully!")
    

    except TranscriptsDisabled:
        st.write("no transcript available for this video")

    except Exception as e:
        st.write(f"error: {str(e)}")



splitter = RecursiveCharacterTextSplitter(chunk_size = 230 , chunk_overlap = 30)

chunks = splitter.create_documents([transcript])
for i, doc in enumerate(chunks):
    st.write(f"chunk{i+1}")
    st.write(f"{doc.page_content}")



embedding_model = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
vectorstore=  Chroma(embedding_function=embedding_model,
                     collection_name="youtube_transcript",
                    persist_directory="./chroma_db")
if "db_initialized" not in st.session_state:
    vectorstore.add_documents(chunks)
    st.session_state.db_initialized = True
retriver = vectorstore.as_retriever(search_kwargs={"k":3})
#to talk in terms of meaning we use cosine as similarity search funstion
retriver_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3 , "fetch_k":10 , "lambda_mult":0.5})




# query = "what is svm?"
# result = retriver.invoke(query)
# result_mmr = retriver_mmr.invoke(query)

prompt = PromptTemplate(
    template="""You are a helpful tutor assistant,
    answer only from provided transcript context.
    if the contexxt is insufficient , just say you don't know.
    
    {context}
    Question :{question}""",

    input_variables=["context", "question"]
)




retrieved_docs = retriver_mmr.invoke(question)

def format_docs(d):
    context_text = "\n\n".join(doc.page_content for doc in d)
    return context_text


for i ,doc in enumerate(retrieved_docs):
    st.write(f"retrived_result_mmr {i+1}")
    st.write(doc.page_content)

# for i ,doc in enumerate(result):
#     st.write(f"result {i+1}")
#     st.write(doc.page_content)

# for i, doc in enumerate(result_multiquery):
#     st.write(f"result_multiquery {i+1}")
#     st.write(doc.page_content)  
      

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "conversational",
    temperature =0.5)

model =   ChatHuggingFace(llm=llm)

# llm_mqr =  HuggingFaceEnd
# model = ChatHuggingFace(llm=llm)
# result =model.invoke("what is svm?")
# st.write("rsult of llm")
# st.write(result.content)

# final_prompt = prompt.invoke({"context": context_Text , "question":question})


parallel_chain=RunnableParallel({
    'context': retriver_mmr | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# #answer = llm.invoke(final_prompt)
# parallel_chain.invoke(question)
if question:
    main_chain = parallel_chain | prompt | model
    result = main_chain.invoke(question)
    st.write(transcript)
    st.write("final Result")
    st.write(result.content)











