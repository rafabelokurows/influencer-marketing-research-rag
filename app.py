import streamlit as st
from rag import query

st.set_page_config(
    page_title="Influencer Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 860px;
    }

    /* Source citation boxes */
    .source-box {
        background-color: #161b22;
        border-left: 3px solid #4CAF50;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.85em;
        font-family: monospace;
    }

    /* Retrieved chunk boxes */
    .chunk-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.8em;
        color: #8b949e;
        margin: 5px 0;
        font-family: monospace;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        border: 1px solid #30363d;
    }

    /* Sidebar */
    .stSidebar {
        border-right: 1px solid #30363d;
    }

    /* Example question buttons */
    .stButton > button {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
        border-radius: 6px;
        font-size: 0.8em;
        text-align: left;
        transition: border-color 0.2s;
    }
    .stButton > button:hover {
        border-color: #4CAF50;
        color: #4CAF50;
    }

    /* Title styling */
    h1 {
        font-family: monospace;
        letter-spacing: -0.5px;
    }

    /* Divider */
    hr {
        border-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📚 Influencer Research Assistant")
    st.caption("Ask questions across your scientific paper library — answers include citations.")
with col2:
    st.caption("🟢 Model loaded")
    st.caption("🗄️ DB connected")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Passages to retrieve (top_k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Higher = more context for the model, but slower answers"
    )

    st.divider()
    st.header("💡 Example questions")
    examples = [
        "What are the key predictors of purchase intention?",
        "How does follower count affect engagement?",
        "Which theories explain influencer credibility?",
        "What is the role of authenticity in influencer marketing?",
        "How does sponsorship disclosure affect consumer behaviour?",
        "What content styles drive the most engagement?",
        "What is the difference between micro and macro influencers?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["queued_question"] = ex

    st.divider()
    st.header("🤖 Model info")
    st.caption("**LLM:** Qwen2.5 1.5B Instruct")
    st.caption("**Embeddings:** all-MiniLM-L6-v2")
    st.caption("**Vector DB:** ChromaDB (local)")
    st.caption("**Runs:** 100% offline")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

# ── Conversation history ──────────────────────────────────────────────────────
for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander(f"📎 {len(item['sources'])} source(s) retrieved"):
            st.markdown("**Sources:**")
            for s in item["sources"]:
                st.markdown(
                    f'<div class="source-box">{s}</div>',
                    unsafe_allow_html=True
                )
            st.divider()
            st.markdown("**Retrieved passages:**")
            for i, (chunk, meta) in enumerate(item["chunks"]):
                st.markdown(f"**[{i+1}] {meta['paper']} — p.{meta['page']}**")
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                st.markdown(
                    f'<div class="chunk-box">{preview}</div>',
                    unsafe_allow_html=True
                )

# ── Input handling ────────────────────────────────────────────────────────────
question = st.chat_input("Ask a research question...")

if "queued_question" in st.session_state:
    question = st.session_state.pop("queued_question")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            result = query(question, top_k=top_k)

        st.write(result["answer"])

        with st.expander(f"📎 {len(result['sources'])} source(s) retrieved"):
            st.markdown("**Sources:**")
            for s in result["sources"]:
                st.markdown(
                    f'<div class="source-box">{s}</div>',
                    unsafe_allow_html=True
                )
            st.divider()
            st.markdown("**Retrieved passages:**")
            for i, (chunk, meta) in enumerate(result["chunks"]):
                st.markdown(f"**[{i+1}] {meta['paper']} — p.{meta['page']}**")
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                st.markdown(
                    f'<div class="chunk-box">{preview}</div>',
                    unsafe_allow_html=True
                )

        st.session_state["history"].append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"]
        })
