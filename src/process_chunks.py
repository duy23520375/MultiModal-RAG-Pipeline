import os
from dotenv import load_dotenv
import json
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


def seperate_content_types(chunk): # Xác định xem chunk gồm những element nào
  content_data = {
    'text': chunk.text,
    'tables': [],
    'images': [],
    'types': ['text']
  }
  # Kiểm tra xem trong chunk có các thuộc tính cần xét không
  if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
    for element in chunk.metadata.orig_elements:
      element_type = type(element).__name__

      #1. Xử lý bảng
      if element_type == 'Table':
        if 'table' not in content_data['types']:
          content_data['types'].append('table')
        table_html = getattr(element.metadata, 'text_as_html', element.text) 
        content_data['tables'].append(table_html)

      #2. Xử lý ảnh
      elif element_type == 'Image':
        if 'image' not in content_data['types']:
          content_data['types'].append('image')
        base_64 = getattr(element.metadata, 'image_base64', element.text)
        content_data['images'].append(base_64)
  return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]):
  print('Enhance content before embedding...')

  llm = ChatGoogleGenerativeAI(
    model= 'gemini-2.5-flash',
    google_api_key= os.getenv('GEMINI_API_KEY'),
    temperature= 0.1
  )
  table_content = "\n\n".join([f'Table (html):{t}' for t in tables])
  prompt_text = f"""
  You are an analyst. Summarize this document for research purposes.
  REQUIREMENT: Extract data from the table and describe the image content (if any).
  TEXT: {text}
  {table_content}
  """
  content = [
    {
      'type': 'text',
      'text': prompt_text
    }
  ]
  # Thêm ảnh vào content gửi cho LLM
  for img_base64 in images:
    content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
        })
  message = HumanMessage(content=content)
  response = llm.invoke([message])
  return response.content

def summarise_chunks(chunks):
  langchain_documents = []
  total_chunks = len(chunks)
  for i, chunk in enumerate(chunks):
    print(f'-> Processing chunk {i + 1}/{total_chunks}')
    content_data = seperate_content_types(chunk)
    print(f"Types found: {content_data['types']}")
    print(f"Tables: {len(content_data['tables'])} || Images: {len(content_data['images'])}")
    enhanced_content = content_data['text']
    if content_data['tables'] or content_data['images']:
      print('-> Starting AI summary...')
      try:
        enhanced_content = create_ai_enhanced_summary(
          content_data['text'],
          content_data['tables'],
          content_data['images'],
        )
        print("Summarise Successfully !")
      except Exception as e:
        print(f"Failed {e} !")
        enhanced_content = content_data['text']

    doc = Document(
      page_content=enhanced_content,
      metadata={
          'original_content': json.dumps({
              'raw_text': content_data['text'],
              'tables_html': content_data['tables'],
              'images_base64': content_data['images'],
          }),
          'page_number': chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else 0
      }
    )
    langchain_documents.append(doc)
  print(f"Finished ! Process {len(langchain_documents)} chunks")
  return langchain_documents

def create_vector_store(chunks, persist_directory='chroma_db/child_db'):
  print("Creating Hierarchical Database...")
  embedding_model = HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs= {'device': 'cpu'}
  )
  
  # 1. VectorDB lưu các Child Chunks
  child_db = Chroma(
    collection_name="split_parents",
    persist_directory=persist_directory,
    embedding=embedding_model,
    collection_metadata={'knsw:space': 'cosine'}
  )
  
  from langchain.storage import InMemoryStore
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain.retrievers import ParentDocumentRetriever
  import pickle
  
  # 2. Store lưu các Parent Chunks
  store = InMemoryStore()
  child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
  
  # 3. Gắn kết Parent và Child
  parent_retriever = ParentDocumentRetriever(
    vectorstore=child_db, 
    docstore=store, 
    child_splitter=child_splitter
  )
  
  print(f'Băm nhỏ và Storing {len(chunks)} Parent Chunks vào Database')
  parent_retriever.add_documents(chunks, ids=None)
  
  # 4. Lưu Parent Store xuống ổ cứng để không bị mất khi tắt app
  import os
  os.makedirs('chroma_db', exist_ok=True)
  with open('chroma_db/parent_store.pkl', 'wb') as f:
      pickle.dump(store.store, f)
      
  return child_db

def embedding_storing(chunks):
  summarised_chunks = summarise_chunks(chunks)
  db =create_vector_store(summarised_chunks)
  print("Storing chunks!!!")
  return db
