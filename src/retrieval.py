import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

def init_ensemble_retriever(persist_directory='chroma_db/db'):
    # 1. Khởi tạo Embedding             
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Load lại Database
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    
    # 3. Chuẩn bị BM25 
    all_docs = db.get()['documents'] 
    
    metadatas = db.get()['metadatas']
    langchain_docs = [Document(page_content=d, metadata=m) for d, m in zip(all_docs, metadatas)]
    
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = 10

    # 4. Chuẩn bị Vector Retriever
    vector_retriever = db.as_retriever(search_kwargs={'k': 10})

    # 5. Ensemble (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def get_reranked_documents(query, ensemble_retriever):
    # 1. Thu thập ứng viên (Candidates)
    relevant_docs = ensemble_retriever.invoke(query)
    
    # 2. Khởi tạo Reranker
    rerank = CohereRerank(
        model="rerank-multilingual-v3.0",
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        top_n=5
    )
    
    # 3. Tái xếp hạng (Rerank)
    rerank_docs = rerank.compress_documents(
        documents=relevant_docs,
        query=query
    )
    
    return rerank_docs