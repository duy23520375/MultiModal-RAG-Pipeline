# Agentic Multi-Modal RAG Assistant with Gemini 2.5 & LangGraph

[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-LangGraph--LangChain-green)](https://python.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-red)](https://aistudio.google.com/)

Hệ thống **Agentic Multi-Modal RAG** (Retrieval-Augmented Generation đa phương tiện dạng Agent) tiên tiến, tích hợp đồ thị trạng thái Agentic Workflow bằng LangGraph. Hệ thống cho phép người dùng hội thoại với các tài liệu PDF phức tạp (chứa văn bản, bảng biểu HTML, hình ảnh Base64), hỗ trợ chống ảo giác (Self-Verification) và tự động viết lại truy vấn (Query Rewriting).

---

## 🌟 Tính năng chính

- **Xử lý Đa Phương Tiện (Multi-Modal)**: Phân tách chi tiết cấu trúc bảng (HTML) và trích xuất hình ảnh (Base64) bằng thư viện `unstructured`.
- **Tập truy xuất phân cấp (Hierarchical RAG)**: Áp dụng `ParentDocumentRetriever` giúp lưu trữ các đoạn text con (Child Chunks) để tìm kiếm chính xác nhưng trả về ngữ cảnh lớn (Parent Chunks) cho LLM.
- **Tìm kiếm kết hợp (Hybrid Search)**: Phối hợp sức mạnh của BM25 (truy vấn từ khóa) và Vector Semantic Search thông qua `EnsembleRetriever`.
- **Tinh chỉnh Embedding (Fine-tuned Embedding)**: Tinh chỉnh mô hình `all-MiniLM-L6-v2` cho miền dữ liệu chuyên biệt để tối ưu hóa kết quả tìm kiếm ngữ nghĩa.
- **Tái xếp hạng (Cohere Rerank)**: Sử dụng mô hình `rerank-multilingual-v3.0` để tối ưu hóa độ liên quan của ngữ cảnh.
- **Đồ thị Agent chống ảo giác (LangGraph self-verification)**: Node Grader thẩm định ảo giác và Node Rewrite viết lại câu hỏi nếu phát hiện ảo giác.
- **Trích dẫn trực quan (Visual Citations)**: Hiển thị trích dẫn trang PDF trực quan trong hộp hội thoại Chat.

---

## 🏗️ Công nghệ sử dụng

- **Workflow Orchestration**: LangGraph & LangChain
- **LLM**: Google Gemini 2.5 Flash (hỗ trợ xoay vòng key thông minh)
- **Vector Database**: ChromaDB
- **Embedding Model**: Fine-tuned `all-MiniLM-L6-v2`
- **Reranker**: Cohere Rerank 3.0
- **Web Search API**: Tavily Search

---

## 🛠️ Hướng dẫn cài đặt & Chạy ứng dụng

### Cách 1: Chạy trực tiếp trên máy (Local)

1. **Clone repository:**
   ```bash
   git clone https://github.com/duy23520375/MultiModal-RAG-Pipeline.git
   cd MultiModal-RAG-Pipeline
   ```

2. **Khởi tạo môi trường ảo Python 3.11:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Cấu hình API Keys:**
   Tạo file `.env` dựa theo file `.env.example` và điền key của bạn:
   ```env
   COHERE_API_KEY=your_cohere_key
   TAVILY_API_KEY=your_tavily_key
   GEMINI_KEY_1=your_gemini_key_1
   GEMINI_KEY_2=your_gemini_key_2
   ```

5. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```

---

### Cách 2: Chạy bằng Docker (Khuyên dùng)

Dự án đã được đóng gói Docker hoàn chỉnh (tự động cài đặt poppler, tesseract-ocr, libmagic cho `unstructured`).

Khởi chạy bằng Docker Compose:
```bash
docker compose up --build -d
```
Ứng dụng sẽ chạy tại địa chỉ: `http://localhost:8501`.

---

## 📁 Cấu trúc thư mục dự án

```text
MultiModal-RAG-Pipeline/
├── app.py                  # Giao diện chính Streamlit và luồng chạy đồ thị
├── Dockerfile              # Docker đóng gói ứng dụng
├── docker-compose.yml      # Cấu hình container chạy Docker compose
├── .env.example            # Mẫu cấu hình biến môi trường
├── requirements.txt        # Danh sách thư viện Python
├── .gitignore              # Cấu hình các file bỏ qua khi git commit
├── src/                    # Thư mục mã nguồn logic hệ thống
│   ├── process_pdf_files.py # Cắt phân rã PDF bằng unstructured
│   ├── process_chunks.py    # Xử lý embedding, tóm tắt và lưu VectorDB
│   ├── retrieval.py         # Dịch thuật, phân rã và tìm kiếm Hybrid
│   ├── chain.py             # Prompt template và khởi tạo LLM chain
│   └── agent.py             # Cấu hình agent workflow phụ trợ
├── data/                   # Thư mục tài liệu PDF mẫu và file kết quả
├── finetuned_model/        # Thư mục mô hình embedding đã được fine-tune
└── multi_modal_rag.ipynb   # Notebook hướng dẫn và đánh giá Ragas
```
