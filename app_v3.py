import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import pypdf
import docx
import re
import os
import io
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DENSO Multimodal CASAC-RAG System v3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for DENSO Professional Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #E60012;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.2rem;
    }
    .badge-casac {
        background-color: #EBF8FF;
        color: #2B6CB0;
        border: 1px solid #BEE3F8;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-sufficient {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-expanded {
        background-color: #FEECDC;
        color: #9C4221;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CASAC RAG ENGINE CLASS
# -----------------------------------------------------------------------------

class RealCASACRAGSystem:
    """
    Complete CASAC RAG Implementation:
    1. Structure-Aware Document Ingestion & Chunking
    2. Contextualization (Heading Hierarchy prepending)
    3. Hybrid TF-IDF / Keyword Retrieval & Reranker
    4. Evidence Sufficiency Check & Selective Context Expansion
    5. Grounded Synthesis & Citation Generator
    """
    
    def __init__(self):
        self.chunks = []  # List of chunk dictionaries
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_indexed = False
        
        # Load default fallback data (Denso Prototype) if no user file uploaded yet
        self._load_default_knowledge_base()

    def _load_default_knowledge_base(self):
        default_docs = [
            {
                "doc_name": "Denso-Hack_NexaCrew_Prototype1.PDF",
                "page": 1,
                "parent_heading": "1. Quy trình Vận hành Máy Quay DENSO",
                "section_title": "1.1 Thông số Áp suất Tiêu chuẩn",
                "content": "Áp suất vận hành tiêu chuẩn của hệ thống máy quay DENSO nằm trong khoảng 120 bar đến 150 bar. Trong điều kiện hoạt động bình thường, áp suất duy trì ổn định ở 135 bar."
            },
            {
                "doc_name": "Denso-Hack_NexaCrew_Prototype1.PDF",
                "page": 2,
                "parent_heading": "1. Quy trình Vận hành Máy Quay DENSO",
                "section_title": "1.2 Quy định Báo động Quá áp & Khẩn cấp",
                "content": "Nếu áp suất vượt quá 180 bar, cảm biến tự động sẽ kích hoạt còi báo động quá áp. Người vận hành phải nhấn nút ngắt khẩn cấp ngay lập tức để bảo vệ thiết bị. Nếu áp suất dưới 100 bar, phải tiến hành bổ sung dầu thủy lực khẩn cấp."
            },
            {
                "doc_name": "Denso-Hack_NexaCrew_Prototype1.PDF",
                "page": 3,
                "parent_heading": "2. Kiến trúc Giải pháp CASAC RAG",
                "section_title": "2.1 Bối cảnh & Thuật toán Ingest",
                "content": "CASAC RAG (Context-assembling and structure-aware chunking RAG) là giải pháp giúp bảo toàn cấu trúc tài liệu bằng cách bổ sung tiêu đề cha và ngữ cảnh phân cấp vào từng chunk văn bản trước khi đưa vào vector DB."
            }
        ]
        self.chunks = []
        for idx, doc in enumerate(default_docs):
            doc_name = doc['doc_name']
            page = doc['page']
            p_head = doc['parent_heading']
            s_title = doc['section_title']
            content = doc['content']
            ctx_str = f"[{doc_name} | Trang {page} | {p_head} > {s_title}]\n{content}"
            chunk_obj = {
                "chunk_id": f"default_chunk_{idx}",
                "doc_name": doc_name,
                "page": page,
                "parent_heading": p_head,
                "section_title": s_title,
                "content": content,
                "neighbor_before": f"default_chunk_{idx-1}" if idx > 0 else None,
                "neighbor_after": f"default_chunk_{idx+1}" if idx < len(default_docs)-1 else None,
                "contextualized_text": ctx_str
            }
            self.chunks.append(chunk_obj)
        self._build_index()

    def clear_database(self):
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_indexed = False

    def ingest_file(self, file_bytes, file_name):
        """
        Extracts content from PDF/DOCX/TXT and performs structure-aware chunking.
        """
        ext = os.path.splitext(file_name)[1].lower()
        extracted_pages = []

        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        extracted_pages.append((page_idx + 1, text))
            except Exception as e:
                st.error(f"Lỗi đọc PDF '{file_name}': {e}")
                return 0
                
        elif ext in [".docx", ".doc"]:
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                extracted_pages.append((1, full_text))
            except Exception as e:
                st.error(f"Lỗi đọc DOCX '{file_name}': {e}")
                return 0

        elif ext in [".txt", ".md"]:
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
                extracted_pages.append((1, text))
            except Exception as e:
                st.error(f"Lỗi đọc file văn bản '{file_name}': {e}")
                return 0
        else:
            st.warning(f"Định dạng file '{file_name}' chưa được hỗ trợ bóc tách văn bản.")
            return 0

        # Structure-Aware Chunking
        added_chunks_count = 0
        current_doc_chunks = []

        for page_num, raw_text in extracted_pages:
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            
            parent_heading = f"Tài liệu {file_name}"
            current_section = f"Trang {page_num}"
            current_paragraph_lines = []

            for line in lines:
                if re.match(r'^(Chương|Chapter|PHẦN|SECTION|MỤC|CHAPTER|CHƯƠNG|\d+\.|\d+\.\d+)\s+', line, re.IGNORECASE) or (len(line) < 60 and line.isupper()):
                    if current_paragraph_lines:
                        para_text = " ".join(current_paragraph_lines)
                        if len(para_text) > 25:
                            current_doc_chunks.append({
                                "file_name": file_name,
                                "page": page_num,
                                "parent_heading": parent_heading,
                                "section_title": current_section,
                                "content": para_text
                            })
                            current_paragraph_lines = []
                    
                    if line.isupper() or len(line) < 40:
                        parent_heading = line
                    else:
                        current_section = line
                else:
                    current_paragraph_lines.append(line)
                    if sum(len(l) for l in current_paragraph_lines) > 350:
                        para_text = " ".join(current_paragraph_lines)
                        current_doc_chunks.append({
                            "file_name": file_name,
                            "page": page_num,
                            "parent_heading": parent_heading,
                            "section_title": current_section,
                            "content": para_text
                        })
                        current_paragraph_lines = []

            if current_paragraph_lines:
                para_text = " ".join(current_paragraph_lines)
                if len(para_text) > 15:
                    current_doc_chunks.append({
                        "file_name": file_name,
                        "page": page_num,
                        "parent_heading": parent_heading,
                        "section_title": current_section,
                        "content": para_text
                    })

        start_idx = len(self.chunks)
        for idx, item in enumerate(current_doc_chunks):
            cid = f"chunk_{start_idx + idx}"
            prev_id = f"chunk_{start_idx + idx - 1}" if idx > 0 else None
            next_id = f"chunk_{start_idx + idx + 1}" if idx < len(current_doc_chunks) - 1 else None

            fname = item['file_name']
            pg = item['page']
            phead = item['parent_heading']
            stitle = item['section_title']
            cnt = item['content']
            ctx_text = f"[{fname} | Trang {pg} | {phead} > {stitle}]\n{cnt}"

            chunk_obj = {
                "chunk_id": cid,
                "doc_name": fname,
                "page": pg,
                "parent_heading": phead,
                "section_title": stitle,
                "content": cnt,
                "neighbor_before": prev_id,
                "neighbor_after": next_id,
                "contextualized_text": ctx_text
            }
            self.chunks.append(chunk_obj)
            added_chunks_count += 1

        self._build_index()
        return added_chunks_count

    def _build_index(self):
        if not self.chunks:
            self.is_indexed = False
            return
        
        corpus = [c["contextualized_text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

    def retrieve_and_answer(self, query, sufficiency_threshold=0.35, openai_api_key=""):
        if not self.chunks or not self.is_indexed:
            return {
                "answer": "Chưa có tài liệu nào được nạp vào hệ thống. Vui lòng tải file lên ở cột bên trái.",
                "status": "NO_DATA",
                "retrieved_chunks": [],
                "expanded": False,
                "best_score": 0.0
            }

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        query_words = set(re.findall(r'\w+', query.lower()))
        scored_candidates = []

        for idx, chunk in enumerate(self.chunks):
            base_sim = similarities[idx]
            content_words = set(re.findall(r'\w+', chunk["contextualized_text"].lower()))
            overlap = len(query_words.intersection(content_words)) / (len(query_words) + 1e-5)
            
            final_score = (base_sim * 0.6) + (overlap * 0.4)
            scored_candidates.append((final_score, chunk))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:3]

        best_score = top_candidates[0][0] if top_candidates else 0.0

        is_sufficient = best_score >= sufficiency_threshold
        status_text = "SUFFICIENT" if is_sufficient else "EXPANDED"

        retrieved_contexts = []
        chunk_map = {c["chunk_id"]: c for c in self.chunks}

        for score, chunk in top_candidates:
            assembled_block = {
                "score": round(score, 4),
                "chunk_id": chunk["chunk_id"],
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "parent_heading": chunk["parent_heading"],
                "section_title": chunk["section_title"],
                "main_content": chunk["content"],
                "expanded_text": ""
            }

            if not is_sufficient:
                expansion_parts = []
                if chunk["neighbor_before"] and chunk["neighbor_before"] in chunk_map:
                    before_c = chunk_map[chunk["neighbor_before"]]
                    expansion_parts.append(f"[Khối trước]: {before_c['content']}")
                
                expansion_parts.append(f"[Khối chính]: {chunk['content']}")

                if chunk["neighbor_after"] and chunk["neighbor_after"] in chunk_map:
                    after_c = chunk_map[chunk["neighbor_after"]]
                    expansion_parts.append(f"[Khối sau]: {after_c['content']}")

                assembled_block["expanded_text"] = "\n".join(expansion_parts)
            else:
                assembled_block["expanded_text"] = chunk["content"]

            retrieved_contexts.append(assembled_block)

        if openai_api_key.strip():
            try:
                import openai
                client = openai.OpenAI(api_key=openai_api_key.strip())
                
                context_prompt = "\n\n".join([
                    f"--- NGUỒN [{c['doc_name']} | Trang {c['page']} | {c['section_title']}] ---\n{c['expanded_text']}"
                    for c in retrieved_contexts
                ])
                
                prompt = f"""Bạn là Chatbot Quản trị Tri thức DENSO. Hãy trả lời câu hỏi dựa TRỰC TIẾP và CHỈ dựa vào nguồn thông tin dưới đây. 
Mỗi thông tin trả lời CẦN phải ghi rõ trích dẫn nguồn [File, Trang, Section].

NGUỒN THÔNG TIN:
{context_prompt}

CÂU HỎI: {query}

CÂU TRẢ LỜI:"""
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                generated_answer = response.choices[0].message.content
            except Exception as e:
                generated_answer = f"Lỗi khi gọi OpenAI API: {e}\n\nChuyển sang chế độ tổng hợp tự động nội bộ."
                generated_answer += "\n\n" + self._synthesize_extractive_answer(query, retrieved_contexts, is_sufficient)
        else:
            generated_answer = self._synthesize_extractive_answer(query, retrieved_contexts, is_sufficient)

        return {
            "answer": generated_answer,
            "status": status_text,
            "retrieved_chunks": retrieved_contexts,
            "best_score": round(best_score, 4),
            "expanded": not is_sufficient
        }

    def _synthesize_extractive_answer(self, query, retrieved_contexts, is_sufficient):
        """Generates a structured, grounded Vietnamese response from extracted passages."""
        if not retrieved_contexts or retrieved_contexts[0]["score"] < 0.02:
            return "Không tìm thấy thông tin phù hợp trong các tài liệu đã tải lên để trả lời câu hỏi này. Bạn vui lòng kiểm tra lại file đã upload hoặc điều chỉnh câu hỏi."

        sentences_found = []
        citations = []

        query_words = set(re.findall(r'\w+', query.lower()))

        for ctx in retrieved_contexts:
            doc_ref = f"[Nguồn: {ctx['doc_name']}, Trang {ctx['page']} - {ctx['section_title']}]"
            citations.append(doc_ref)
            
            raw_text = ctx["expanded_text"]
            lines = [s.strip() for s in re.split(r'[\.\n]+', raw_text) if len(s.strip()) > 10]

            for line in lines:
                line_words = set(re.findall(r'\w+', line.lower()))
                overlap = len(query_words.intersection(line_words))
                if overlap > 0:
                    sentences_found.append((overlap, line, doc_ref))

        sentences_found.sort(key=lambda x: x[0], reverse=True)

        ans_lines = [
            f"Dựa trên dữ liệu trích xuất từ tài liệu **({retrieved_contexts[0]['doc_name']})**:"
        ]

        if not is_sufficient:
            ans_lines.append("⚠️ *Lưu ý: Ngữ cảnh ban đầu chưa đủ ngưỡng, hệ thống đã kích hoạt **Selective Context Expansion** để tự động kéo thêm các phần lân cận.*")

        ans_lines.append("")

        unique_sents = []
        seen = set()
        for overlap, sent, ref in sentences_found:
            if sent.lower() not in seen:
                seen.add(sent.lower())
                unique_sents.append((sent, ref))

        if unique_sents:
            for s, r in unique_sents[:5]:
                ans_lines.append(f"- **{s}** {r}")
        else:
            top_ctx = retrieved_contexts[0]
            ans_lines.append(f"- {top_ctx['main_content']} [{top_ctx['doc_name']} | Trang {top_ctx['page']}]")

        return "\n".join(ans_lines)

# -----------------------------------------------------------------------------
# STREAMLIT UI & INTERACTION
# -----------------------------------------------------------------------------

if "rag_system" not in st.session_state:
    st.session_state.rag_system = RealCASACRAGSystem()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

rag = st.session_state.rag_system

# Sidebar - Document Ingestion & Configuration
with st.sidebar:
    st.title("🔴 DENSO Knowledge Base")
    st.subheader("📂 Quản lý Nguồn Tài liệu")
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Tải lên tài liệu mới (PDF, DOCX, TXT):",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Ingest & Index", type="primary", use_container_width=True):
            if uploaded_files:
                rag.clear_database()
                st.session_state.uploaded_file_names = []
                
                total_chunks = 0
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    file_bytes = file.read()
                    chunks_added = rag.ingest_file(file_bytes, file.name)
                    total_chunks += chunks_added
                    st.session_state.uploaded_file_names.append(file.name)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"✅ Đã xử lý {len(uploaded_files)} file! Đã tạo **{total_chunks}** Structure-Aware Chunks.")
            else:
                st.warning("Vui lòng chọn ít nhất 1 file trước khi bấm Ingest.")

    with col_btn2:
        if st.button("🔄 Reset Mặc định", use_container_width=True):
            rag._load_default_knowledge_base()
            st.session_state.uploaded_file_names = []
            st.info("Đã khôi phục dữ liệu mẫu DENSO Prototype.")

    st.markdown("---")
    
    st.subheader("📊 Trạng thái Kho Kiến thức")
    st.write(f"- **Tổng số Chunks:** `{len(rag.chunks)}`")
    if st.session_state.uploaded_file_names:
        st.write("**File người dùng tải lên:**")
        for fn in st.session_state.uploaded_file_names:
            st.caption(f"📄 {fn}")
    else:
        st.caption("📌 Đang dùng dữ liệu mẫu mặc định (Denso-Hack_NexaCrew_Prototype1.PDF)")

    st.markdown("---")
    st.subheader("⚙️ Cấu hình CASAC RAG")
    
    sufficiency_threshold = st.slider(
        "Threshold Evidence Sufficiency Check:",
        min_value=0.1,
        max_value=0.8,
        value=0.35,
        step=0.05,
        help="Nếu điểm Rerank thấp hơn ngưỡng này, hệ thống tự động kích hoạt Selective Context Expansion."
    )
    
    # Auto-load API Key from .env environment variable if available
    env_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if env_api_key:
        st.success("🟢 Đã tự động nạp OPENAI_API_KEY từ file .env")
        default_key = env_api_key
    else:
        default_key = ""
        st.info("💡 Chưa tìm thấy file .env hoặc chưa khai báo OPENAI_API_KEY")

    openai_key = st.text_input(
        "OpenAI API Key (Tùy chọn):",
        value=default_key,
        type="password",
        help="Nhập API Key để dùng GPT-4o trả lời. Hệ thống ưu tiên dùng Key từ file .env nếu có."
    )

# Main Screen - Header & Chatbot Interface
st.markdown('<div class="main-header">DENSO Multimodal CASAC-RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hệ thống Trợ lý AI Quản trị & Truy xuất Tri thức Nội bộ <span class="badge-casac">Document-Aware RAG</span></div>',
    unsafe_allow_html=True
)

st.markdown("##### 💡 Câu hỏi gợi ý thử nghiệm:")
col_q1, col_q2, col_q3 = st.columns(3)

user_query = None
if col_q1.button("📋 Áp suất an toàn máy DENSO là bao nhiêu?", use_container_width=True):
    user_query = "Mức áp suất an toàn của máy sản xuất DENSO là bao nhiêu và làm gì khi quá áp?"

if col_q2.button("⚠️ Xử lý thế nào khi áp suất < 100 bar?", use_container_width=True):
    user_query = "Xử lý thế nào khi áp suất máy quay giảm xuống dưới 100 bar?"

if col_q3.button("🔍 Tóm tắt tài liệu đã tải lên", use_container_width=True):
    user_query = "Tóm tắt các nội dung và thông số chính trong tài liệu đã nạp."

input_query = st.chat_input("Nhập câu hỏi kỹ thuật hoặc tra cứu tri thức tại đây...")
if input_query:
    user_query = input_query

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander("🔍 Chi tiết Bằng chứng & Trích dẫn Nguồn (CASAC RAG Inspector)"):
                st.write(f"**Trạng thái Sufficiency Check:** `{msg['details']['status']}`")
                st.write(f"**Rerank Score cao nhất:** `{msg['details']['best_score']}`")
                for c in msg['details']['retrieved_chunks']:
                    st.markdown(f"---")
                    st.markdown(f"**📄 Source File:** `{c['doc_name']}` | **Trang:** `{c['page']}`")
                    st.markdown(f"**📌 Section:** `{c['parent_heading']} > {c['section_title']}`")
                    st.markdown(f"**Score:** `{c['score']}`")
                    st.text_area("Văn bản ngữ cảnh:", c["expanded_text"], height=100, key=f"hist_{c['chunk_id']}_{time.time()}")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Đang tìm kiếm vector, tái xếp hạng & kiểm tra độ đầy đủ bằng chứng..."):
            result = rag.retrieve_and_answer(
                query=user_query,
                sufficiency_threshold=sufficiency_threshold,
                openai_api_key=openai_key
            )
            
            st.markdown(result["answer"])
            
            if result["status"] == "SUFFICIENT":
                st.markdown('<span class="status-sufficient">✅ Evidence Sufficiency: ĐỦ (Sufficient)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-expanded">⚡ Evidence Sufficiency: THIẾU -> Context Expanded</span>', unsafe_allow_html=True)
            
            with st.expander("🔍 Chi tiết Bằng chứng & Trích dẫn Nguồn (CASAC RAG Inspector)"):
                st.write(f"**Trạng thái Sufficiency Check:** `{result['status']}`")
                st.write(f"**Rerank Score cao nhất:** `{result['best_score']}`")
                for c in result['retrieved_chunks']:
                    st.markdown(f"---")
                    st.markdown(f"**📄 Source File:** `{c['doc_name']}` | **Trang:** `{c['page']}`")
                    st.markdown(f"**📌 Section:** `{c['parent_heading']} > {c['section_title']}`")
                    st.markdown(f"**Score:** `{c['score']}`")
                    st.text_area("Ngữ cảnh được lắp ráp:", c["expanded_text"], height=100, key=f"new_{c['chunk_id']}_{time.time()}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "details": result
            })
