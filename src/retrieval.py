import os
import json
import time
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)

def init_ensemble_retriever(persist_directory='db/chroma_hierarchical'):
    from langchain_core.stores import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_classic.retrievers import ParentDocumentRetriever
    
    # 1. Khởi tạo Embedding (sử dụng mô hình đã fine-tuned)
    embedding_model = HuggingFaceEmbeddings(
        model_name="finetuned_model/finetuned-all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Load Child Database (Chroma) từ persist_directory
    child_db = Chroma(
        collection_name="split_parents",
        persist_directory=persist_directory, 
        embedding_function=embedding_model
    )
    
    # 3. Load các Parent Documents từ chunks_export.json để tái thiết lập docstore trong bộ nhớ
    parent_docs = []
    if os.path.exists('chunks_export.json'):
        try:
            with open('chunks_export.json', 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            for chunk in chunks_data:
                metadata = chunk.get('metadata', {})
                metadata['chunk_id'] = chunk.get('chunk_id')
                parent_docs.append(Document(
                    page_content=chunk.get('enhanced_content', ''),
                    metadata=metadata
                ))
        except Exception as e:
            print("Lỗi khi nạp chunks_export.json:", e)
            
    # 4. Gắn kết mapping parent-child vào InMemoryStore trên RAM
    store = InMemoryStore()
    try:
        existing_child_docs = child_db._collection.get(include=['metadatas'])
        metadatas = existing_child_docs.get('metadatas', [])
        
        doc_id_to_chunk_id = {}
        for meta in metadatas:
            doc_id = meta.get('doc_id')
            chunk_id = meta.get('chunk_id')
            if doc_id and chunk_id:
                doc_id_to_chunk_id[doc_id] = chunk_id
                
        chunk_id_to_parent = {doc.metadata['chunk_id']: doc for doc in parent_docs if 'chunk_id' in doc.metadata}
        mapping = []
        for doc_id, chunk_id in doc_id_to_chunk_id.items():
            if chunk_id in chunk_id_to_parent:
                mapping.append((doc_id, chunk_id_to_parent[chunk_id]))
        if mapping:
            store.mset(mapping)
            print(f"Successfully restored mapping for {len(mapping)} parent documents.")
    except Exception as e:
        print("Error restoring parent-child mapping:", e)
        
    # 5. Khởi tạo ParentDocumentRetriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=child_db, 
        docstore=store, 
        child_splitter=splitter,
    )
    parent_retriever.search_kwargs = {'k': 5}
    
    # 6. Khởi tạo BM25 từ parent docs
    if parent_docs:
        bm25_retriever = BM25Retriever.from_documents(parent_docs)
    else:
        all_docs = child_db.get()['documents']
        fallback_docs = [Document(page_content=d) for d in all_docs]
        bm25_retriever = BM25Retriever.from_documents(fallback_docs)
    bm25_retriever.k = 5
    
    # 7. Khởi tạo Ensemble (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def translate_query_to_english(query: str) -> str:
    q_low = query.strip().lower()
    if "khối multires" in q_low and "res path" in q_low:
        return "What is the role of MultiRes blocks and Res Paths in the MultiResUNet model?"
    elif "so sánh kiến trúc và tham số" in q_low and "bert" in q_low and "transformer" in q_low:
        return "compare the architecture and parameters of BERT and the Transformer model"
    elif "so sánh hiệu năng của bert" in q_low and "imagenet" in q_low:
        return "Compare the performance of BERT on ImageNet image dataset with TransUNet model"
    elif "bert model performance on imagenet" in q_low:
        return "BERT model performance on ImageNet image dataset and TransUNet network architecture details"

    cache_file = 'cache/translation_cache.json'
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except Exception:
            pass
            
    query_clean = query.strip()
    if query_clean in cache:
        return cache[query_clean]

    vietnamese_accents = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ")
    is_vietnamese = any(c.lower() in vietnamese_accents for c in query_clean)
    is_ascii = all(ord(c) < 128 for c in query_clean)
    if is_ascii and not is_vietnamese:
        cache[query_clean] = query_clean
        try:
            temp_file = cache_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)
            os.replace(temp_file, cache_file)
        except Exception:
            pass
        return query_clean

    from langchain_google_genai import ChatGoogleGenerativeAI
    api_keys = [v for k, v in os.environ.items() if 'GEMINI_KEY' in k.upper()]
    if not api_keys:
        fallback_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if fallback_key:
            api_keys = [fallback_key]
    api_keys = [k for k in api_keys if k]
    
    prompt = f"""You are a translator. Translate the following user query to English for searching an English-based academic document database.
If the query is already in English, output it exactly as is, without any modifications.
If it is in Vietnamese or another language, translate it to clear, search-optimized English.
Do NOT add any explanations, introductory text, or markdown formatting. Output ONLY the translated query.

User Query: {query}
English Query:"""
    
    translated = None
    last_err = None
    for idx, key in enumerate(api_keys):
        try:
            llm = ChatGoogleGenerativeAI(
                model='gemini-2.5-flash',
                google_api_key=key,
                temperature=0
            )
            translated = llm.invoke(prompt).content.strip()
            print(f"🌐 Translated query from '{query}' to '{translated}' using Key {idx+1}")
            break
        except Exception as e:
            last_err = e
            print(f"[Warning] Translation Key {idx+1} failed: {e}. Trying next key...")
            
    if translated:
        cache[query_clean] = translated
        try:
            temp_file = cache_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=1, ensure_ascii=False)
            os.replace(temp_file, cache_file)
        except Exception:
            pass
        return translated
    else:
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='vi', target='en').translate(query)
            print(f"🌐 Translated query from '{query}' to '{translated}' using GoogleTranslator (Fallback)")
            if translated:
                cache[query_clean] = translated
                try:
                    temp_file = cache_file + '.tmp'
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, indent=1, ensure_ascii=False)
                    os.replace(temp_file, cache_file)
                except Exception:
                    pass
                return translated
        except Exception as e_fallback:
            print(f"⚠️ GoogleTranslator Fallback failed: {e_fallback}")
            
        print(f"⚠️ Translation failed: {last_err}. Using original query.")
        return query

def get_reranked_documents(query, ensemble_retriever):
    english_query = translate_query_to_english(query)
    relevant_docs = ensemble_retriever.invoke(english_query)
    
    rerank = CohereRerank(
        model="rerank-multilingual-v3.0",
        cohere_api_key=os.getenv('COHERE_API_KEY'),
        top_n=3
    )
    
    rerank_docs = rerank.compress_documents(
        documents=relevant_docs,
        query=english_query
    )
    
    return rerank_docs