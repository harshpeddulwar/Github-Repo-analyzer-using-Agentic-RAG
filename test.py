import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config_loader import load_config
from src.cloning import clone_repository, repo_id_from_path
from src.parsing import load_and_chunk_repository
from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore
from src.logger import get_logger
from agents.orchestrator_agent import OrchestratorAgent

logger = get_logger("cli")

def main():
    # 1. Load configuration
    config = load_config()
    
    llm_model = config["llm"]["model"]
    collection_name = config["vector_db"]["collection_name"]

    print("=" * 80)
    print(" GITHUB REPOSITORY RAG PIPELINE ")
    print("=" * 80)

    # 2. Get GitHub URL from user
    github_url = input("Enter GitHub Repository URL: ").strip()
    if not github_url:
        print("[ERROR] GitHub URL cannot be empty.")
        return

    # 3. Clone repository
    print(f"\n[STEP 1] Cloning repository: {github_url} ...")
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join("repos", repo_name)
    already_exists = os.path.exists(local_path)

    local_path = clone_repository(github_url)
    if not local_path:
        print("[ERROR] Cloning failed. Exiting.")
        return

    # Initialize Embedder and Vector DB
    embedder = EmbeddingPipe(
        model_name=config["embedding"]["model"],
        batch_size=config["embedding"]["batch_size"],
    )
    vcfg = config["vector_db"]
    vector_db = QdrantStore(
        collection_name=collection_name,
        vector_size=vcfg["vector_size"],
        url=vcfg["url"],
        path=vcfg["path"],
    )
    repo_id = repo_id_from_path(local_path)

    # 4. Handle indexing choice if repository already exists
    reindex = True
    if already_exists:
        choice = input("\nRepository already cloned. Do you want to re-index it? (y/n) [y]: ").strip().lower()
        if choice == "n":
            reindex = False

    if reindex:
        # 5. Parse/Chunk repository
        print("\n[STEP 2] Parsing and chunking repository source files...")
        chunks = load_and_chunk_repository(local_path)
        print(f"[INFO] Generated {len(chunks)} chunks from source files.")

        if not chunks:
            print("[WARN] No chunks extracted. Cannot index repository.")
            return

        # 6. Embed first (DB untouched), then upsert and drop only this repo's stale points
        print("\n[STEP 3] Embedding chunks, then updating the index for this repo...")
        enriched_chunks = embedder.embed_chunks(chunks, repo_id=repo_id)
        vector_db.ensure_collection()
        vector_db.upsert(enriched_chunks)
        vector_db.delete_stale(repo_id, keep_ids=[c["chunk_id"] for c in enriched_chunks])
        print(f"[INFO] Indexing completed. {vector_db.count(repo_id)} points stored for '{repo_id}'.")
    else:
        print("\n[STEP 2 & 3] Skipping parsing and re-indexing. Using existing vector index.")
        vector_db.ensure_collection()
        if not vector_db.has_repo(repo_id):
            print(f"[WARN] No vectors found for '{repo_id}'. Re-run and choose to re-index.")

    # 7. Initialize Orchestrator Agent
    print("\n[STEP 4] Initializing Orchestrator Agent...")
    agent = OrchestratorAgent(
        repo_path=local_path,
        embedder=embedder,
        vector_db=vector_db,
        model=llm_model,
        repo_id=repo_id,
        config=config,
    )
    print("\nAgent ready! You can now query the repository.")
    print("Type 'exit' to quit.")

    # 8. Query Loop
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            if not query:
                continue
            if query.lower() == "exit":
                print("Exiting RAG QA pipeline. Goodbye!")
                break

            print("\n" + "-" * 50)
            print(f"Processing query: '{query}'")
            print("-" * 50)

            result = agent.answer(query)

            print("\n" + "=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)
            print(result.get("answer", "No answer generated."))

            plan = result.get("plan", {})
            ver = result.get("verification", {})
            print("\n" + "-" * 80)
            print(f"[plan] intent={plan.get('intent')} sub_questions={plan.get('sub_questions')}")
            print(f"[verify] passed={ver.get('passed')} revisions={ver.get('revisions')} issues={ver.get('issues')}")
            
            print("\n" + "=" * 80)
            print("EVIDENCE / TOOL CALLS")
            print("=" * 80)
            print(result.get("evidence", "No evidence recorded."))
            print("=" * 80)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred while answering: {e}")

if __name__ == "__main__":
    main()
