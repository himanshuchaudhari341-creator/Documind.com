import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import docx
import chromadb
from google import genai

# Load environment variables silently
load_dotenv()

# ----------------- App Config & UI Setup -----------------
st.set_page_config(
    page_title="DocuMind AI - Notes Q&A Engine",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind: Smart Notes Q&A Engine")
st.caption("Upload PDFs or Word docs, search across content, and get verified summarized answers strictly from your notes.")

# ----------------- Secure Background Client Setup -----------------
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error("⚠️ GEMINI_API_KEY .env file mein nahi mili! Project folder mein .env file banakar key add karein.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ----------------- In-Memory ChromaDB Setup -----------------
@st.cache_resource
def get_chroma_collection():
    chroma = chromadb.Client()
    return chroma.get_or_create_collection(name="pdf_notes_collection")

collection = get_chroma_collection()

# ----------------- Session State -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ----------------- Sidebar (Vault & 10 Languages) -----------------
with st.sidebar:
    st.header("📂 Document Vault")
    uploaded_files = st.file_uploader(
        "Upload Notes (PDF / DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.header("🌐 Response Language")
    languages = [
        "English",
        "Hindi (हिंदी)",
        "Hinglish (Hindi in Roman script)",
        "Marwadi (मारवाड़ी)",
        "Marathi (मराठी)",
        "Gujarati (ગુજરાતી)",
        "Tamil (தமிழ்)",
        "Telugu (తెలుగు)",
        "Bengali (বাংলা)",
        "Kannada (ಕನ್ನಡ)"
    ]
    selected_language = st.selectbox(
        "Answer kis bhasha mein chahiye?",
        options=languages,
        index=0
    )

# ----------------- Document Parser -----------------
def process_documents(files):
    """PDF aur DOCX files se text extract karta hai."""
    all_chunks = []
    all_metadatas = []
    chunk_size = 500
    overlap = 80

    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                text = " ".join(raw_text.split())
                if not text:
                    continue
                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size].strip()
                    if len(chunk) > 30:
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "file_name": file.name,
                            "page": page_idx + 1
                        })
                        
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            full_text = " ".join([p.text for p in doc.paragraphs if p.text.strip()])
            for i in range(0, len(full_text), chunk_size - overlap):
                chunk = full_text[i:i + chunk_size].strip()
                if len(chunk) > 30:
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "file_name": file.name,
                        "page": 1
                    })
                    
    return all_chunks, all_metadatas

def generate_with_fallback(prompt: str) -> str:
    """Active Gemini models se structured answer generate karta hai."""
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]
    last_err = ""
    for model_name in candidate_models:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            if res and res.text:
                return res.text
        except Exception as e:
            last_err = str(e)
            continue
    return f"AI Error: {last_err}"

# ----------------- Ingestion Pipeline -----------------
if uploaded_files:
    current_names = [f.name for f in uploaded_files]
    if current_names != st.session_state.processed_files:
        with st.spinner("Documents parse aur vector store mein index ho rahe hain..."):
            existing_ids = collection.get()["ids"]
            if existing_ids:
                collection.delete(ids=existing_ids)

            chunks, metadatas = process_documents(uploaded_files)

            if chunks:
                try:
                    ids = [f"doc_{i}" for i in range(len(chunks))]
                    collection.add(
                        documents=chunks,
                        metadatas=metadatas,
                        ids=ids
                    )

                    st.session_state.processed_files = current_names
                    st.session_state.chat_history = []
                    st.sidebar.success(f"Indexed {len(chunks)} chunks across {len(uploaded_files)} file(s)!")
                except Exception as e:
                    st.sidebar.error(f"Indexing Error: {str(e)}")
            else:
                st.sidebar.warning("Uploaded files mein readable text nahi mila.")

# ----------------- Chat Interface -----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Verified Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 **{s['file']} (Page {s['page']})**: {s['text']}")

user_query = st.chat_input("Apne uploaded notes se sawal puchein...")

if user_query:
    if not st.session_state.processed_files:
        st.warning("Pehle left sidebar se document upload karein.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        search_results = collection.query(query_texts=[user_query], n_results=4)

        retrieved_docs = search_results["documents"][0]
        retrieved_meta = search_results["metadatas"][0]

        context_blocks = []
        sources = []
        for doc, meta in zip(retrieved_docs, retrieved_meta):
            context_blocks.append(f"[{meta['file_name']} | Page {meta['page']}]: {doc}")
            sources.append({"file": meta["file_name"], "page": meta["page"], "text": doc})

        context_text = "\n\n---\n\n".join(context_blocks)

        # Dynamic language rules and localized headings
        lang_config = {
            "English": {
                "desc": "Write the response strictly in formal English.",
                "h1": "📌 Summary & Overview",
                "h2": "🔑 Key Concepts & Highlights"
            },
            "Hindi (हिंदी)": {
                "desc": "उत्तर पूरी तरह शुद्ध हिंदी (Devanagari script) में ही लिखें। Technical terms को छोड़कर बाकी वाक्य रचना केवल हिंदी होनी चाहिए।",
                "h1": "📌 सारांश और विवरण",
                "h2": "🔑 मुख्य बिंदु और विशेषताएँ"
            },
            "Hinglish (Hindi in Roman script)": {
                "desc": "Poora response Hinglish (Hindi written in English alphabets) mein naturally likhein. Pure English sentences mat banayein.",
                "h1": "📌 Summary & Overview",
                "h2": "🔑 Main Points aur Features"
            },
            "Marwadi (मारवाड़ी)": {
                "desc": "पूरो जवाब मारवाड़ी बोली (Devanagari script / राजस्थानी मारवाड़ी) में लिखो। सरल मारवाड़ी में साफ-साफ समझाओ।",
                "h1": "📌 सारांश अर विवरण",
                "h2": "🔑 खास बातां अर मुख्य अंग"
            },
            "Marathi (मराठी)": {
                "desc": "संपूर्ण उत्तर शुद्ध मराठी भाषेत (Devanagari script) स्पष्टपणे लिहा.",
                "h1": "📌 सारांश आणि ओळख",
                "h2": "🔑 मुख्य संकल्पना आणि वैशिष्ट्ये"
            },
            "Gujarati (ગુજરાતી)": {
                "desc": "સંપૂર્ણ જવાબ ગુજરાતી લિપિમાં (Gujarati script) મુદ્દાસર લખો.",
                "h1": "📌 સારાંશ અને વિહંગાવલોકન",
                "h2": "🔑 મુખ્ય મુદ્દાઓ અને વિશેષતાઓ"
            },
            "Tamil (தமிழ்)": {
                "desc": "முழு பதிலையும் தூய தமிழ் எழுத்துக்களில் (Tamil script) தெளிவாக எழுதவும்.",
                "h1": "📌 சுருக்கம் மற்றும் கண்ணோட்டம்",
                "h2": "🔑 முக்கிய கருத்துக்கள் மற்றும் சிறப்பம்சங்கள்"
            },
            "Telugu (తెలుగు)": {
                "desc": "పూర్తి సమాధానాన్ని స్పష్టమైన తెలుగు లిపిలో (Telugu script) రాయండి.",
                "h1": "📌 సారాంశం మరియు అవలోకనం",
                "h2": "🔑 ముఖ్యమైన అంశాలు మరియు ముఖ్యాంశాలు"
            },
            "Bengali (বাংলা)": {
                "desc": "সম্পূর্ণ উত্তরটি স্পষ্ট বাংলা লিপিতে (Bengali script) পয়েন্ট আকারে লিখুন।",
                "h1": "📌 সারসংক্ষেপ এবং বিবরণ",
                "h2": "🔑 মূল ধারণা এবং বৈশিষ্ট্য"
            },
            "Kannada (ಕನ್ನಡ)": {
                "desc": "ಸಂಪೂರ್ಣ ಉತ್ತರವನ್ನು ಸ್ಪಷ್ಟ ಕನ್ನಡ ಲಿಪಿಯಲ್ಲಿ (Kannada script) ಬರೆಯಿರಿ.",
                "h1": "📌 ಸಾರಾಂಶ ಮತ್ತು ಅವಲೋಕನ",
                "h2": "🔑 ಪ್ರಮುಖ ಅಂಶಗಳು ಮತ್ತು ಮುಖ್ಯಾಂಶಗಳು"
            }
        }

        cfg = lang_config.get(selected_language, lang_config["English"])

        grounded_prompt = f"""
You are an expert tutor. Analyze the provided context and answer the user query accurately.

CRITICAL INSTRUCTION - TARGET LANGUAGE:
Target Language: {selected_language}
Rule: {cfg['desc']}

Strict Output Structure:
### {cfg['h1']}
(Provide a clear 2-3 line summary in {selected_language}. Do not directly copy-paste raw English sentences; translate and synthesize the meaning into {selected_language}.)

### {cfg['h2']}
(Explain the core points and components in bullet points in {selected_language}.)

Important:
- Use only facts from the context.
- Keep technical terms (e.g., APIs, framework names, software names) recognizable, but all explanation and sentence structure must strictly follow {selected_language}.

Context:
{context_text}

User Question: {user_query}
"""

        with st.chat_message("assistant"):
            with st.spinner(f"Generating summary in {selected_language}..."):
                bot_reply = generate_with_fallback(grounded_prompt)
                st.markdown(bot_reply)

                if sources:
                    with st.expander("🔍 View Verified Sources"):
                        for s in sources:
                            st.caption(f"📄 **{s['file']} (Page {s['page']})**: {s['text']}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_reply,
            "sources": sources
        })
    except Exception as e:
        st.error(f"Search Query Error: {str(e)}")