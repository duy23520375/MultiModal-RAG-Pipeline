import os
import json
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from src.retrieval import get_reranked_documents
from src.chain import reconstruct_context

load_dotenv(override=True)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent_workflow(ensemble_retriever):
    """
    Tạo LangGraph Agent tích hợp Multi-modal RAG, Web Search và Code Interpreter.
    """
    
    # Khởi tạo mô hình
    llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )

    # 1. TOOL: Tìm kiếm tài liệu nội bộ
    def internal_pdf_search(query: str) -> str:
        """
        Sử dụng công cụ này ĐẦU TIÊN để tra cứu thông tin chuyên sâu từ tài liệu PDF nội bộ của người dùng.
        Công cụ này trả về văn bản, và cả dữ liệu bảng (HTML/JSON) và ảnh.
        """
        try:
            docs = get_reranked_documents(query, ensemble_retriever)
            context = reconstruct_context(docs)
            # Chuyển đổi list context multi-modal thành chuỗi JSON để Tool trả về dạng Text, 
            # sau đó LLM tự render hoặc ta format cho LLM dễ đọc.
            # Với Gemini, ta có thể trả về string mô tả nội dung.
            text_response = "--- TRÍCH XUẤT TỪ TÀI LIỆU NỘI BỘ ---\n"
            for item in context:
                if item["type"] == "text":
                    text_response += item["text"] + "\n"
                elif item["type"] == "image_url":
                    text_response += "[HÌNH ẢNH ĐƯỢC TÌM THẤY TRONG TÀI LIỆU - Dạng Base64]\n"
            return text_response
        except Exception as e:
            return f"Lỗi khi tìm kiếm tài liệu nội bộ: {e}"

    internal_search_tool = StructuredTool.from_function(
        func=internal_pdf_search,
        name="internal_pdf_search",
        description="Sử dụng để tìm kiếm thông tin từ tài liệu PDF tải lên (chứa text, bảng biểu, đồ thị)."
    )

    # 2. TOOL: Tìm kiếm Web (Dự phòng)
    web_search_tool = TavilySearchResults(
        max_results=3,
        description="Chỉ sử dụng công cụ này NẾU tài liệu nội bộ không chứa câu trả lời, HOẶC cần thông tin cập nhật mới nhất từ Internet."
    )

    # 3. TOOL: Data Analyst (Code Interpreter)
    python_repl_tool = PythonREPLTool(
        description="""
        Một Python shell (REPL). Sử dụng công cụ này để thực thi mã Python (ví dụ: Pandas, Math). 
        ĐẶC BIỆT HỮU ÍCH: Khi người dùng hỏi các câu hỏi TÍNH TOÁN (tổng, trung bình, max, min) dựa trên dữ liệu Bảng (Table) được trích xuất từ PDF.
        Bạn có thể khởi tạo DataFrame từ dữ liệu HTML/JSON của bảng và tính toán, sau đó in ra kết quả bằng hàm print().
        """
    )

    tools = [internal_search_tool, web_search_tool, python_repl_tool]
    llm_with_tools = llm.bind_tools(tools)

    # ĐỊNH NGHĨA CÁC NODE TRONG ĐỒ THỊ
    def agent_node(state: AgentState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Hàm điều hướng (Routing)
    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if not last_message.tool_calls:
            return END
        return "tools"

    # KHỞI TẠO ĐỒ THỊ LANGGRAPH
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", agent_node)
    
    # ToolNode tự động map và chạy tool dựa trên tool_calls của LLM
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
