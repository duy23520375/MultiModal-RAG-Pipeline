#  Multi-Modal RAG Assistant with Gemini 2.5 & LangChain

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-LangChain-green)](https://python.langchain.com/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-red)](https://aistudio.google.com/)

Hệ thống RAG (Retrieval-Augmented Generation) đa phương tiện , cho phép người dùng trò chuyện với tài liệu PDF phức tạp bao gồm văn bản, bảng biểu và hình ảnh. Sử dụng kiến trúc Hybrid Search kết hợp với Reranking để đạt độ chính xác tối ưu.



---

## 🌟 Key Features

- **Multi-Modal Processing**: Trích xuất và phân tích sâu sắc các Element (Text, Table, Image) từ PDF sử dụng thư viện `unstructured`.
- **Hybrid Retrieval**: Kết hợp sức mạnh của **BM25** (Keyword search) và **Vector Search** (Semantic search) thông qua `EnsembleRetriever`.
- **AI-Powered Summarization**: Tự động tóm tắt nội dung bảng biểu và mô tả hình ảnh bằng Gemini để tăng cường khả năng tìm kiếm (Indexing).
- **Reranking**: Sử dụng **Cohere Rerank** (`rerank-multilingual-v3.0`) để tái sắp xếp kết quả, đảm bảo thông tin chính xác nhất được đưa vào Context.
- **Contextual Query**: Tự động viết lại câu hỏi (Query Rewriting) dựa trên lịch sử trò chuyện để xử lý các câu hỏi phụ thuộc ngữ cảnh.
- **Streamlit UI**: Giao diện người dùng trực quan, hỗ trợ Upload file và Chat thời gian thực.

---

## 🏗️ Architecture Stack

- **Orchestration**: LangChain
- **LLM**: Google Gemini 2.5 Flash
- **Vector Database**: ChromaDB
- **Embedding**: HuggingFace (`all-MiniLM-L6-v2`)
- **PDF Partitioning**: Unstructured (Hi-Res Strategy)
- **Reranker**: Cohere AI

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/MultiModal-RAG-Gemini.git](https://github.com/your-username/MultiModal-RAG-Gemini.git)
cd MultiModal-RAG-Gemini
```
### 2. Create Virtual Environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Environment Configuration
GEMINI_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
