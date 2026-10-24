"""
app.py — Streamlit UI for Hybrid Knowledge Graph RAG Movie Assistant
"""

import time
import streamlit as st

# ---------------------------------------------------------------------------
# Page config MUST be the very first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Movie Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helper: import our modules (safe, env‑var–guarded)
# ---------------------------------------------------------------------------
def _import_modules():
    """Import the core modules after ensuring env vars are loaded."""
    import config  # noqa: F401 — loads .env and validates keys
    from rag_chain import run_hybrid_rag
    from graph_retriever import graph_retrieve
    from vector_store import similarity_search, get_vector_store
    from router import route
    return config, run_hybrid_rag, graph_retrieve, similarity_search, get_vector_store, route

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"query": str, "answer": str, "retriever": str, "time": float}

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

# ---------------------------------------------------------------------------
# Load modules & pre‑warm retrievers (runs once at startup)
# ---------------------------------------------------------------------------
try:
    config_module, run_hybrid_rag, graph_retrieve, similarity_search, get_vector_store, route = _import_modules()
    # Force vector store build once so the UI is ready immediately
    vs = get_vector_store()
    st.session_state.vector_ready = True
except Exception as e:
    st.error(f"⚠️ Initialization error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: system status & sample questions
# ---------------------------------------------------------------------------
st.sidebar.title("🔧 System Status")
st.sidebar.write(f"**Neo4j URI**: {config_module.NEO4J_URI}")
st.sidebar.write(f"**OpenAI Key**: {'*' * 8}…{config_module.OPENAI_API_KEY[-4:] if config_module.OPENAI_API_KEY else 'N/A'}")

if st.sidebar.button("🔄 Clear Chat"):
    st.session_state.chat_history = []
    # Also clear the text_input value from session state
    if "text_input" in st.session_state:
        del st.session_state.text_input
    st.rerun()

st.sidebar.write("")
st.sidebar.write("### Sample Questions")
sample_questions = [
    "Who acted in The Matrix?",
    "Who directed Inception?",
    "Who has Tom Hanks worked with?",
    "Recommend a sci-fi movie about dreams.",
    "Find a film about survival on a deserted island.",
    "Who directed The Dark Knight?",
    "Recommend a romantic movie to watch.",
    "Which actors starred in Titanic?",
    "Find a mind-bending thriller.",
    "Who has Keanu Reeves worked with?",
]
for sq in sample_questions:
    if st.sidebar.button(sq, key=f"sample_{sq}"):
        # Set the query in session state so the text input pre-fills on rerun
        st.session_state.text_input = sq
        st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🎬 Hybrid Knowledge Graph RAG Movie Assistant")
st.caption("Ask me about movies, actors, directors, or get recommendations. I'll route your question to the best retriever (Neo4j graph or FAISS vector) and generate a grounded answer.")

# Chat area
st.write("**Chat**")
chat_container = st.container()

with chat_container:
    for i, entry in enumerate(st.session_state.chat_history):
        # User question
        st.markdown(f"**You:** {entry['query']}")
        # Assistant answer
        st.markdown(f"**Assistant:** {entry['answer']}")
        # Retriever badge
        st.caption(f"🔍 Retrieval method: {entry['retriever'].upper()} retrieval  |  ⏱️ {entry['time']:.2f}s")
        st.divider()

# Question input — use session state value if available (e.g. from sample button)
query = st.text_input("Your question", placeholder="e.g. Who acted in The Matrix?")

if submit := st.button("Send", use_container_width=True):
    user_query = query.strip()
    if not user_query:
        st.warning("Please enter a question.")
        st.rerun()

    # Clear sample query from session state after using it
    if "text_input" in st.session_state:
        del st.session_state.text_input

    # Record start time
    start_time = time.time()

    # Run the hybrid RAG pipeline
    try:
        result = run_hybrid_rag(user_query, verbose=False)
        elapsed = time.time() - start_time

        # Store in chat history
        st.session_state.chat_history.append({
            "query": user_query,
            "answer": result["answer"],
            "retriever": result["retriever"],
            "time": elapsed,
        })
    except Exception as e:
        st.error(f"❌ Error: {e}")

    st.rerun()

# ---------------------------------------------------------------------------
# Footer / expandable context
# ---------------------------------------------------------------------------
if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]
    with st.expander("📡 Show retrieved context & details"):
        st.write(f"**Query:** {last['query']}")
        st.write(f"**Retriever used:** {last['retriever'].upper()}")

        # Re‑run retrieval to show context (graph or vector)
        try:
            if last["retriever"] == "graph":
                from graph_retriever import graph_retrieve as gr
                ctx = gr(last["query"])
                st.write("**Graph query results:**")
                st.code(ctx or "No results returned", language="text")
            else:
                from vector_store import similarity_search as ss
                docs = ss(last["query"], k=4)
                formatted = "\n\n---\n\n".join(doc.page_content for doc in docs)
                st.write("**Vector search results (top 4):**")
                st.code(formatted or "No results returned", language="text")
        except Exception as e:
            st.info(f"Could not load context: {e}")

        st.write(f"**Generated answer time:** {last['time']:.2f}s")

# ---------------------------------------------------------------------------
# Run on import if called directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pass  # Handled by Streamlit runtime