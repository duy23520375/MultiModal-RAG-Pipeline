import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
load_dotenv(override=True)

from src.process_pdf_files import process_pdf_files
from src.process_chunks import embedding_storing
from src.retrieval import init_ensemble_retriever, get_reranked_documents
from src.chain import contextualize_q_prompt, qa_prompt, reconstruct_context, llm


# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Multi-Modal RAG System", layout="wide", page_icon="🤖")
st.title("🤖 Multi-Modal RAG Chatbot")
st.markdown("---")

# --- SIDEBAR: UPLOAD & CONFIG ---
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    
    if uploaded_file:
        if st.button("🚀 Process & Indexing"):
            with st.status("Processing documents...", expanded=True) as status:
                # Lưu file tạm
                with open("temp_upload.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.write("1. Phân tách PDF (Hi-Res)...")
                chunks = process_pdf_files("temp_upload.pdf")
                
                st.write("2. AI Summarizing & Embedding (Gemini)...")
                # Hàm này sẽ gọi summarise_chunks và tạo ChromaDB
                db = embedding_storing(chunks)
                
                status.update(label="✅ Indexing finished", state="complete", expanded=False)
                st.success("The system is ready for chatting!")
                # Reset retriever trong session để load cái mới
                if 'ensemble_retriever' in st.session_state:
                    del st.session_state['ensemble_retriever']

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- KHỞI TẠO RETRIEVER ---
# Chỉ khởi tạo khi đã có DB trong folder
if os.path.exists("chroma_db/db"):
    if 'ensemble_retriever' not in st.session_state:
        with st.spinner("Connecting to the database..."):
            st.session_state.ensemble_retriever = init_ensemble_retriever()
else:
    st.warning("Please upload and process the PDF in the sidebar to begin.")

# --- QUẢN LÝ LỊCH SỬ CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị tin nhắn cũ
for message in st.session_state.chat_history:
    with st.chat_message("human" if isinstance(message, HumanMessage) else "ai"):
        st.markdown(message.content)

# --- KHUNG CHAT CHÍNH ---
if user_input := st.chat_input("Ask me anything about the document..."):
    # 1. Hiển thị câu hỏi
    with st.chat_message("human"):
        st.markdown(user_input)
    
    # 2. Xử lý câu trả lời
    if 'ensemble_retriever' not in st.session_state:
        st.error("No data available! Please upload the PDF first.")
    else:
        with st.chat_message("ai"):
            with st.spinner("Information is being extracted..."):
                # BƯỚC 1: Làm rõ câu hỏi (Contextualize)
                contextualize_chain = contextualize_q_prompt | llm
                refined_query = contextualize_chain.invoke({
                    'chat_history': st.session_state.chat_history,
                    'input': user_input
                }).content

                # BƯỚC 2: Tìm kiếm & Rerank 
                # Lấy 10-20 cái rồi Cohere lọc còn 5
                final_docs = get_reranked_documents(refined_query, st.session_state.ensemble_retriever)
                
                # BƯỚC 3: Tái cấu trúc context (Text + Table + Image)
                context_multi_modal = reconstruct_context(final_docs)
                
                # BƯỚC 4: Gemini trả lời
                qa_chain = qa_prompt | llm
                response = qa_chain.invoke({
                    'chat_history': st.session_state.chat_history,
                    'context': context_multi_modal,
                    'input': user_input
                })
                
                # Hiển thị câu trả lời
                st.markdown(response.content)
                
                # Hiển thị nguồn (Nâng cao)
                with st.expander("🔍 See the sources and images used."):
                    for i, doc in enumerate(final_docs):
                        st.write(f"**Source: {i+1} (Page: {doc.metadata.get('page_number')}):**")
                        st.text(doc.page_content[:300] + "...")

        # Lưu lịch sử
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response.content))