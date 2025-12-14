import streamlit as st
from pinecone import Pinecone
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from docx import Document
import time

# --- CẤU HÌNH ---
# Lấy từ Secrets (trên Cloud) hoặc điền trực tiếp nếu chạy local
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "123456") # Mật khẩu mặc định là 123456
except:
    # Fallback cho chạy local nếu chưa setup secrets
    GOOGLE_API_KEY = "ĐIỀN_KEY_GOOGLE_CUA_BAN"
    PINECONE_API_KEY = "ĐIỀN_KEY_PINECONE_CUA_BAN"
    ADMIN_PASSWORD = "123456" 

PINECONE_INDEX_NAME = "chatbot-demo"

# Setup
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

st.set_page_config(page_title="Chatbot Tài Liệu", page_icon="🤖", layout="wide")

# --- PHẦN 1: ADMIN PANEL (NẠP DỮ LIỆU) ---
with st.sidebar:
    st.header("⚙️ Quản trị viên")
    password = st.text_input("Nhập mật khẩu Admin", type="password")
    
    if password == ADMIN_PASSWORD:
        st.success("Đã mở khóa tính năng nạp dữ liệu!")
        uploaded_files = st.file_uploader("Upload tài liệu (PDF, DOCX, TXT)", accept_multiple_files=True)
        
        if st.button("Xử lý & Nạp vào AI"):
            if not uploaded_files:
                st.warning("Vui lòng chọn file trước!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                vectors_to_upsert = []
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Đang đọc file: {file.name}...")
                    
                    # Đọc nội dung file
                    text = ""
                    try:
                        if file.name.endswith('.pdf'):
                            pdf = PdfReader(file)
                            for page in pdf.pages: text += page.extract_text() or ""
                        elif file.name.endswith('.docx'):
                            doc = Document(file)
                            for para in doc.paragraphs: text += para.text + "\n"
                        elif file.name.endswith('.txt'):
                            text = file.read().decode("utf-8")
                    except Exception as e:
                        st.error(f"Lỗi đọc file {file.name}: {e}")
                        continue
                        
                    # Chia nhỏ & Embedding
                    chunks = text_splitter.split_text(text)
                    for chunk_id, chunk_text in enumerate(chunks):
                        try:
                            embedding = genai.embed_content(
                                model="models/text-embedding-004",
                                content=chunk_text,
                                task_type="retrieval_document"
                            )['embedding']
                            
                            vector_id = f"{file.name}_{chunk_id}"
                            metadata = {"text": chunk_text, "source": file.name}
                            vectors_to_upsert.append((vector_id, embedding, metadata))
                        except Exception as e:
                            pass # Bỏ qua lỗi nhỏ để chạy tiếp
                    
                    # Cập nhật thanh tiến trình
                    progress_bar.progress((i + 1) / total_files)

                # Đẩy lên Pinecone
                status_text.text("Đang đẩy dữ liệu lên Cloud...")
                batch_size = 50
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    index.upsert(vectors=batch)
                    time.sleep(1) # Tránh rate limit
                
                status_text.text("✅ Hoàn tất! Dữ liệu mới đã sẵn sàng.")
                st.balloons()
    elif password:
        st.error("Sai mật khẩu!")

# --- PHẦN 2: GIAO DIỆN CHAT (CHO NGƯỜI DÙNG) ---
st.title("🤖 Trợ lý Tra Cứu Tài Liệu")
st.caption("Hỏi đáp miễn phí dựa trên 500 tài liệu đã cung cấp.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_relevant_context(query):
    try:
        query_embedding = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")['embedding']
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        context_text = ""
        for match in results['matches']:
            context_text += f"\n[Nguồn: {match['metadata'].get('source', 'Unknown')}]: {match['metadata'].get('text', '')}\n---\n"
        return context_text
    except:
        return ""

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm..."):
            context = get_relevant_context(prompt)
            if not context:
                response_text = "Tôi chưa có dữ liệu về vấn đề này, hoặc hệ thống dữ liệu đang trống."
            else:
                full_prompt = f"Thông tin: {context}\nCâu hỏi: {prompt}\nHãy trả lời dựa trên thông tin trên."
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(full_prompt)
                    response_text = response.text
                except Exception as e:
                    response_text = f"Xin lỗi, có lỗi xảy ra: {e}"
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})