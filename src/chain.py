import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(override=True)


# 1. Prompt làm rõ câu hỏi
contextualize_q_system_prompt = """
Based on the conversation history and the most recent question, 
rewrite it as a standalone question. Do NOT answer, just rewrite the question.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "{input}")
])

# 2. Prompt trả lời chính (QA)
qa_system_prompt = """
You are a professional virtual assistant. 
Use ONLY the CONTEXT provided below (including text, tables, and images) to answer the user's question.

RULES:
1. If the information is NOT present in the CONTEXT, strictly answer: "Dữ liệu không đủ cung cấp để trả lời câu hỏi này."
2. Do not use your outside knowledge.
3. Prioritize data from tables.
4. Describe images to support your answer.

CONTEXT:
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ('system', qa_system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', "{input}")
])
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key= os.getenv("GEMINI_API_KEY")
)
def reconstruct_context(relevant_docs):
    final_content = []
    for i, doc in enumerate(relevant_docs):
        try:
            original = json.loads(doc.metadata.get('original_content', '{}'))
            raw_text = original.get('raw_text', '')
            tables = original.get('tables_html', [])
            images = original.get('images_base64', []) 

            text_part = f"--- Source {i+1} ---\n{raw_text}\n"
            if tables:
                text_part += "Table Data (HTML):\n" + "\n".join(tables)
            
            final_content.append({"type": "text", "text": text_part})
            
            for img_b64 in images:
                final_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        except:
            final_content.append({"type": "text", "text": doc.page_content})
    return final_content