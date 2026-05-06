import streamlit as st
from rag import query

st.set_page_config(
    page_title="Influencer Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #4CAF50;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        font-size: 0.85em;
    }
    .chunk-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.8em;
        color: #555;
        margin: 5px 0;
    }
    .stChatMessage {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📚 Influencer Marketing Research Assistant")
st.caption("Ask questions across your scientific paper library — answers include citations from the source papers.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider(
        "Passages to retrieve (top_k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Higher = more context but slower answers"
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
    st.caption("LLM: Qwen2.5 1.5B Instruct (local)")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector DB: ChromaDB (local)")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# Session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Display conversation history
for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        with st.expander(f"📎 {len(item['sources'])} source(s) retrieved"):
            for s in item["sources"]:
                st.markdown(f'<div class="source-box">{s}</div>', unsafe_allow_html=True)
            st.divider()
            for i, (chunk, meta) in enumerate(item["chunks"]):
                st.markdown(f"**[{i+1}] {meta['paper']} — p.{meta['page']}**")
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                st.markdown(f'<div class="chunk-box">{preview}</div>', unsafe_allow_html=True)

# Handle input — either from chat box or example button
question = st.chat_input("Ask a research question...")
if "queued_question" in st.session_state:
    question = st.session_state.pop("queued_question")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            result = query(question, top_k=top_k)

        st.write(result["answer"])

        with st.expander(f"📎 {len(result['sources'])} source(s) retrieved"):
            for s in result["sources"]:
                st.markdown(f'<div class="source-box">{s}</div>', unsafe_allow_html=True)
            st.divider()
            for i, (chunk, meta) in enumerate(result["chunks"]):
                st.markdown(f"**[{i+1}] {meta['paper']} — p.{meta['page']}**")
                preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
                st.markdown(f'<div class="chunk-box">{preview}</div>', unsafe_allow_html=True)

        st.session_state["history"].append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"]
        })