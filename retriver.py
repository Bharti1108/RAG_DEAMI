from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from 

load_dotenv()

transcript =  """Evaluation of a Retrieval-Augmented Generation (RAG) system involves assessing both its ability to retrieve relevant information and to generate accurate, meaningful responses based on that information. First, the retrieval component is evaluated using metrics like recall@k, precision@k, and mean reciprocal rank (MRR), which measure how effectively the system identifies and ranks relevant documents. Next, the generation component is assessed using metrics such as BLEU, ROUGE, BERTScore, or exact match to determine how closely the generated response aligns with a reference answer in terms of correctness and semantic similarity. More importantly, end-to-end evaluation focuses on overall answer quality, including faithfulness (whether the answer is grounded in retrieved documents), relevance (whether it answers the question properly), and coherence. Modern approaches often use large language models as evaluators to judge these qualities, while human evaluation remains the most reliable method for detecting subtle errors and hallucinations. Overall, effective RAG evaluation requires a combination of retrieval, generation, and holistic metrics to ensure both accurate information retrieval and high-quality response generation."""
splitter = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 10)
chunks = splitter.create_documents([transcript])

embedding_model = HuggingFaceEmbeddings( model_name ="sentence-transformers/all-MiniLM-L6-v2")
    
db = FAISS.from_documents(chunks, embedding_model)

retriver = db.as_retriever(search_kwargs={"k":3})
mmr_retriver = db.as_retriever(search_type ='mmr',search_kwargs={'k':3})


for i, 
ground_truth =    
{
  'question' : ''
}


