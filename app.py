import os
import sys
import yaml
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Internal modules
from src.cloning import clone_repository, walk_repository, REPO_DIR
from src.parsing import load_and_chunk_repository
from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore
from agents.orchestrator_agent import OrchestratorAgent

# =====================================================================
# Page Configuration & Clean Styling
# =====================================================================
st.set_page_config(
    page_title="GitHub Repo Analyzer - Agentic RAG",
    page_icon="👍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a clean, modern aesthetic without sidebar
st.markdown("""
<style>
    /* Hide Streamlit Sidebar elements */
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Main Container Styling */
    .main .block-container {
        max-width: 860px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }

    /* Status & Metric Cards */
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.1);
        color: #4f46e5;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-right: 0.5rem;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* QA Box styling */
    .qa-box {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        background: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10b981;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Configuration & Resource Caching
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


config = get_config()
LLM_MODEL = config["llm"]["model"]
COLLECTION_NAME = config["vector_db"]["collection_name"]


@st.cache_resource(show_spinner="Loading Embedding Model (Jina Code)...")
def get_embedder():
    return EmbeddingPipe()


@st.cache_resource(show_spinner="Connecting to Vector DB (Qdrant)...")
def get_vector_db():
    return QdrantStore(collection_name=COLLECTION_NAME)


# Initialize Session State
if "active_repo_path" not in st.session_state:
    st.session_state.active_repo_path = None
if "active_repo_name" not in st.session_state:
    st.session_state.active_repo_name = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Auto-detect existing repos in repos/ directory
existing_repos = []
if os.path.exists(REPO_DIR):
    existing_repos = [
        d for d in os.listdir(REPO_DIR)
        if os.path.isdir(os.path.join(REPO_DIR, d)) and not d.startswith(".")
    ]

# If an existing repo is present and no active repo is set, default to it
if not st.session_state.active_repo_name and existing_repos:
    default_repo = existing_repos[0]
    st.session_state.active_repo_name = default_repo
    st.session_state.active_repo_path = os.path.join(REPO_DIR, default_repo)


# Helper function to instantiate orchestrator
def get_or_create_orchestrator(repo_path: str):
    embedder = get_embedder()
    vector_db = get_vector_db()
    return OrchestratorAgent(
        repo_path=repo_path,
        embedder=embedder,
        vector_db=vector_db,
        model=LLM_MODEL,
        max_revision_attempts=1,
    )


# =====================================================================
# Header Section
# =====================================================================
st.markdown("""
<div class="app-header">
    <div class="app-title">⚡ GitHub Repo Analyzer</div>
    <div class="app-subtitle">Agentic RAG for Deep Codebase Understanding & Question Answering</div>
</div>
""", unsafe_allow_html=True)

# Status bar
col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    if st.session_state.active_repo_name:
        st.success(f"📦 Repo: **{st.session_state.active_repo_name}**")
    else:
        st.warning("📦 Repo: None (Please ingest)")

with col_stat2:
    try:
        v_db = get_vector_db()
        count = v_db.count()
        st.info(f"🔢 Vectors in DB: **{count}**")
    except Exception:
        st.info("🔢 Vectors in DB: **0**")

# with col_stat3:
#     st.info(f"🤖 Model: **{LLM_MODEL.split('/')[-1]}**")

st.divider()

# =====================================================================
# 1. Ingestion Section
# =====================================================================
st.subheader("1. Ingest Repository")

col_url, col_btn = st.columns([4, 1.2])

with col_url:
    repo_url_input = st.text_input(
        "GitHub Repository URL or Local Name",
        placeholder="https://github.com/owner/repo",
        value="" if not st.session_state.active_repo_name else st.session_state.active_repo_name,
        label_visibility="collapsed",
    )

with col_btn:
    ingest_button = st.button("🚀 Ingest & Index", type="primary", use_container_width=True)

col_opt1, col_opt2 = st.columns([2, 2])
with col_opt1:
    reindex_toggle = st.checkbox("Re-parse & Re-index Embeddings", value=True, help="Uncheck if the repository is already indexed in Qdrant")
with col_opt2:
    if existing_repos:
        selected_existing = st.selectbox(
            "Or switch to existing cloned repo:",
            options=["-- Select --"] + existing_repos,
            index=0,
            label_visibility="collapsed",
        )
        if selected_existing != "-- Select --" and selected_existing != st.session_state.active_repo_name:
            st.session_state.active_repo_name = selected_existing
            st.session_state.active_repo_path = os.path.join(REPO_DIR, selected_existing)
            st.session_state.orchestrator = get_or_create_orchestrator(st.session_state.active_repo_path)
            st.success(f"Switched active repository to **{selected_existing}**")
            st.rerun()

# Handle Ingestion Logic
if ingest_button:
    target = repo_url_input.strip()
    if not target:
        st.error("Please enter a valid GitHub URL or local repository name.")
    else:
        status_container = st.container()
        with status_container:
            progress_bar = st.progress(0, text="Starting repository ingestion...")

            try:
                # Step 1: Clone or resolve local path
                progress_bar.progress(15, text="Cloning repository...")
                if target.startswith("http://") or target.startswith("https://"):
                    local_path = clone_repository(target)
                    if not local_path:
                        st.error(f"Failed to clone repository from '{target}'.")
                        st.stop()
                    repo_name = os.path.basename(os.path.normpath(local_path))
                else:
                    if os.path.isdir(os.path.join(REPO_DIR, target)):
                        local_path = os.path.join(REPO_DIR, target)
                        repo_name = target
                    elif os.path.isdir(target):
                        local_path = os.path.abspath(target)
                        repo_name = os.path.basename(os.path.normpath(local_path))
                    else:
                        st.error(f"Local repository path '{target}' not found.")
                        st.stop()

                # Step 2: Parse & Chunk
                embedder = get_embedder()
                vector_db = get_vector_db()

                if reindex_toggle:
                    progress_bar.progress(35, text="Parsing and chunking code files via AST...")
                    chunks = load_and_chunk_repository(local_path)
                    
                    if not chunks:
                        st.warning("No parseable code files or chunks found in this repository.")
                    else:
                        st.info(f"Extracted **{len(chunks)}** AST code chunks from repository files.")

                        # Step 3: Embed Chunks
                        progress_bar.progress(60, text=f"Generating embeddings for {len(chunks)} chunks...")
                        vector_db.ensure_collection(recreate=True)
                        enriched_chunks = embedder.embed_chunks(chunks)

                        # Step 4: Upsert to Vector DB
                        progress_bar.progress(85, text="Storing embeddings into Vector DB (Qdrant)...")
                        upserted_count = vector_db.upsert(enriched_chunks)
                        progress_bar.progress(100, text="Ingestion complete!")
                        st.success(f"✅ Successfully indexed **{upserted_count}** chunks into Qdrant for **{repo_name}**!")
                else:
                    vector_db.ensure_collection(recreate=False)
                    progress_bar.progress(100, text="Using existing vector index.")
                    st.success(f"✅ Active repository set to **{repo_name}** using existing vector index.")

                # Set active session state
                st.session_state.active_repo_path = local_path
                st.session_state.active_repo_name = repo_name
                st.session_state.orchestrator = get_or_create_orchestrator(local_path)
                st.rerun()

            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")

st.divider()

# =====================================================================
# 2. User Query Input & QA Box
# =====================================================================
st.subheader("2. Codebase Question Answering (Agentic RAG)")

# Quick suggestion chips
# st.markdown("<small style='color: gray;'>💡 Suggested Questions:</small>", unsafe_allow_html=True)
# suggest_cols = st.columns(3)

# suggested_queries = [
#     "What is the overall architecture and main entry point?",
#     "Explain how data/request flow is handled in this codebase.",
#     "List the key modules and their main responsibilities.",
# ]

# clicked_suggestion = None
# for idx, (col, q_text) in enumerate(zip(suggest_cols, suggested_queries)):
#     with col:
#         if st.button(f"📌 {q_text[:32]}...", key=f"sug_{idx}", use_container_width=True):
#             clicked_suggestion = q_text

# Query input form
with st.form(key="qa_form", clear_on_submit=False):
    default_text = clicked_suggestion if clicked_suggestion else st.session_state.get("query_input", "")
    user_query = st.text_input(
        "Ask a question about the repository:",
        value=default_text,
        placeholder="e.g., How does authentication work? Where is the database configured?",
        key="qa_text_input"
    )
    col_submit, col_clear = st.columns([4, 1])
    with col_submit:
        submit_query = st.form_submit_button(" Ask Agent", type="primary", use_container_width=True)
    with col_clear:
        clear_btn = st.form_submit_button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.last_result = None
    st.session_state.last_query = ""
    st.session_state.query_input = ""
    st.session_state.qa_history = []
    st.rerun()

# Handle Query Execution (either form submit or suggestion click)
query_to_run = clicked_suggestion if clicked_suggestion else (user_query.strip() if submit_query else None)

if query_to_run:
    q = query_to_run.strip()
    if not q:
        st.warning("Please enter a question.")
    elif not st.session_state.active_repo_path:
        st.error("No active repository loaded. Please ingest or select a repository first above.")
    else:
        if st.session_state.orchestrator is None:
            st.session_state.orchestrator = get_or_create_orchestrator(st.session_state.active_repo_path)

        with st.spinner("Orchestrator Agent is analyzing codebase & gathering evidence..."):
            try:
                result = st.session_state.orchestrator.answer(q)
                st.session_state.last_query = q
                st.session_state.last_result = result
                st.session_state.qa_history.append({"query": q, "result": result})
            except Exception as e:
                st.error(f"Error answering query: {str(e)}")

# Display QA Box
if st.session_state.last_result:
    res = st.session_state.last_result
    st.markdown("### 💬 Answer")
    
    # Render final answer
    st.markdown(res.get("answer", "No answer generated."))

    # Render Evidence / Tool Calls in an expandable accordion
    evidence = res.get("evidence", "")
    if evidence and evidence != "No tool calls were made.":
        with st.expander("🔍 View Agent Tool Calls & Retrieved Evidence", expanded=False):
            st.markdown(f"```text\n{evidence}\n```")
    
    # Previous queries in session
    if len(st.session_state.qa_history) > 1:
        with st.expander(f"📜 Session Q&A History ({len(st.session_state.qa_history)} queries)", expanded=False):
            for i, item in enumerate(reversed(st.session_state.qa_history[:-1]), start=1):
                st.markdown(f"**Q: {item['query']}**")
                st.markdown(item['result'].get('answer', ''))
                st.divider()

# Footer
st.markdown("<br><br><center><small style='color: gray;'>Agentic RAG GitHub Repoitory Analyzer </small></center>", unsafe_allow_html=True)
