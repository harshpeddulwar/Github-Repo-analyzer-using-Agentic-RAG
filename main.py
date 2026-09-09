import os
import sys
import yaml
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables (.env)
load_dotenv()

# Internal project modules
from src.cloning import clone_repository, walk_repository, REPO_DIR
from src.parsing import load_and_chunk_repository
from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore
from agents.orchestrator_agent import OrchestratorAgent


# =====================================================================
# Configuration Loader
# =====================================================================
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config("config.yaml")
LLM_MODEL = config["llm"]["model"]
COLLECTION_NAME = config["vector_db"]["collection_name"]


# =====================================================================
# Application State & Lifespan Management
# =====================================================================
class AppState:
    embedder: Optional[EmbeddingPipe] = None
    vector_db: Optional[QdrantStore] = None
    active_repo_path: Optional[str] = None
    active_repo_name: Optional[str] = None
    orchestrator: Optional[OrchestratorAgent] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy assets (Embedding model & Vector DB client) once at startup.
    """
    print("[STARTUP] Initializing Embedding Model & Qdrant Connection...")
    try:
        state.embedder = EmbeddingPipe()
        state.vector_db = QdrantStore(collection_name=COLLECTION_NAME)
        state.vector_db.ensure_collection(recreate=False)
        print("[STARTUP] Embedding Pipe & Vector DB successfully initialized.")
    except Exception as e:
        print(f"[STARTUP ERROR] Failed to initialize core resources: {e}")

    yield

    print("[SHUTDOWN] Cleaning up server resources...")


# =====================================================================
# FastAPI Application Initialization
# =====================================================================
app = FastAPI(
    title="Agentic RAG GitHub Repo Analyzer API",
    description="REST API for parsing, embedding, and answering codebase queries using Agentic RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration (React, Vue, Streamlit, HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Pydantic Request / Response Models
# =====================================================================
class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL (e.g. https://github.com/user/repo) or local folder name")
    reindex: bool = Field(default=True, description="Whether to re-parse and recreate embeddings in Qdrant")


class IngestResponse(BaseModel):
    status: str
    message: str
    repo_name: str
    repo_path: str
    chunks_indexed: int
    total_vectors: int


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question about the repository codebase")
    repo_name: Optional[str] = Field(default=None, description="Target repository name (optional, defaults to active repo)")


class QueryResponse(BaseModel):
    query: str
    repo_name: str
    answer: str
    evidence: str


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    collection_name: str
    active_repo: Optional[str]
    vector_count: int


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Returns API health status, loaded configurations, active repository, and vector count.
    """
    vector_count = 0
    if state.vector_db:
        try:
            vector_count = state.vector_db.count()
        except Exception:
            vector_count = 0

    return HealthResponse(
        status="healthy",
        llm_model=LLM_MODEL,
        collection_name=COLLECTION_NAME,
        active_repo=state.active_repo_name,
        vector_count=vector_count,
    )


@app.get("/repos", tags=["Repository"])
def list_repositories():
    """
    List all repositories currently cloned or available locally in the `repos/` folder.
    """
    if not os.path.exists(REPO_DIR):
        return {"repositories": []}

    repos = [
        d for d in os.listdir(REPO_DIR)
        if os.path.isdir(os.path.join(REPO_DIR, d)) and not d.startswith(".")
    ]
    return {
        "active_repo": state.active_repo_name,
        "repositories": repos
    }


@app.get("/files", tags=["Repository"])
def get_repository_files(repo_name: Optional[str] = None):
    """
    List all indexed source files in the active or specified repository.
    """
    target_repo = repo_name or state.active_repo_name
    if not target_repo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active repository. Ingest a repository first or provide 'repo_name'."
        )

    target_path = os.path.join(REPO_DIR, target_repo) if not os.path.isabs(target_repo) else target_repo
    if not os.path.isdir(target_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{target_repo}' not found at '{target_path}'."
        )

    files = walk_repository(target_path)
    rel_files = [os.path.relpath(f, target_path).replace("\\", "/") for f in files]

    return {
        "repo_name": target_repo,
        "total_files": len(rel_files),
        "files": rel_files
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Repository"])
def ingest_repository(req: IngestRequest):
    """
    Clones a GitHub repository (if remote), parses AST code chunks,
    generates embeddings, and upserts them into Qdrant.
    """
    if not state.embedder or not state.vector_db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedder or Vector DB not initialized."
        )

    repo_url = req.repo_url.strip()
    if not repo_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Repository URL cannot be empty.")

    # 1. Resolve Local Path & Clone if needed
    if repo_url.startswith("http://") or repo_url.startswith("https://"):
        local_path = clone_repository(repo_url)
        if not local_path:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to clone repository.")
        repo_name = os.path.basename(os.path.normpath(local_path))
    else:
        # Local repository path
        if os.path.isdir(os.path.join(REPO_DIR, repo_url)):
            local_path = os.path.join(REPO_DIR, repo_url)
            repo_name = repo_url
        elif os.path.isdir(repo_url):
            local_path = os.path.abspath(repo_url)
            repo_name = os.path.basename(os.path.normpath(local_path))
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Local repository '{repo_url}' not found.")

    chunks_indexed = 0

    # 2. Chunking & Indexing into Qdrant
    if req.reindex:
        print(f"[INGEST] Parsing codebase at '{local_path}'...")
        chunks = load_and_chunk_repository(local_path)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No parseable source code chunks found in the repository."
            )

        print(f"[INGEST] Recreating Qdrant collection '{COLLECTION_NAME}'...")
        state.vector_db.ensure_collection(recreate=True)

        print(f"[INGEST] Embedding {len(chunks)} chunks...")
        enriched_chunks = state.embedder.embed_chunks(chunks)

        print(f"[INGEST] Upserting to Qdrant...")
        state.vector_db.upsert(enriched_chunks)
        chunks_indexed = len(enriched_chunks)
    else:
        state.vector_db.ensure_collection(recreate=False)

    # 3. Update Active State
    state.active_repo_path = local_path
    state.active_repo_name = repo_name
    state.orchestrator = OrchestratorAgent(
        repo_path=local_path,
        embedder=state.embedder,
        vector_db=state.vector_db,
        model=LLM_MODEL,
        max_revision_attempts=1,
    )

    total_vectors = state.vector_db.count()

    return IngestResponse(
        status="success",
        message="Repository ingested and indexed successfully.",
        repo_name=repo_name,
        repo_path=local_path,
        chunks_indexed=chunks_indexed,
        total_vectors=total_vectors,
    )


@app.post("/query", response_model=QueryResponse, tags=["Agentic RAG"])
def query_repository(req: QueryRequest):
    """
    Executes an Agentic RAG retrieval and reasoning loop using the Orchestrator
    and RetrievalAgent to answer questions about the codebase with full tool evidence.
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty.")

    # Determine target repo & agent
    target_repo = req.repo_name or state.active_repo_name
    if not target_repo:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active repository loaded. Please ingest a repository first via /ingest."
        )

    # Setup orchestrator if target repo differs or not instantiated yet
    if state.orchestrator is None or (req.repo_name and req.repo_name != state.active_repo_name):
        target_path = os.path.join(REPO_DIR, target_repo) if not os.path.isabs(target_repo) else target_repo
        if not os.path.isdir(target_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target repository '{target_repo}' not found."
            )
        state.active_repo_path = target_path
        state.active_repo_name = target_repo
        state.orchestrator = OrchestratorAgent(
            repo_path=target_path,
            embedder=state.embedder,
            vector_db=state.vector_db,
            model=LLM_MODEL,
            max_revision_attempts=1,
        )

    try:
        result = state.orchestrator.answer(query_text)
        return QueryResponse(
            query=query_text,
            repo_name=state.active_repo_name,
            answer=result.get("answer", "No answer generated."),
            evidence=result.get("evidence", "No evidence recorded."),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agentic query: {str(e)}"
        )


# =====================================================================
# Main Execution Entrypoint
# =====================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
