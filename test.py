import os
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.cloning import clone_repository
from src.parsing import load_and_chunk_repository
from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore
from agents.orchestrator_agent import OrchestratorAgent

def main():
    # 1. Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file '{config_path}' not found.")
        return

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    llm_model = config["llm"]["model"]
    temperature = config["llm"]["temperature"]
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
    # clone_repository returns local_path
    repo_name = github_url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join("repos", repo_name)
    already_exists = os.path.exists(local_path)

    local_path = clone_repository(github_url)
    if not local_path:
        print("[ERROR] Cloning failed. Exiting.")
        return

    # Initialize Embedder and Vector DB
    embedder = EmbeddingPipe()
    vector_db = QdrantStore(collection_name=collection_name)

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

        # 6. Ensure Collection and Upsert Embeddings
        print("\n[STEP 3] Recreating Qdrant collection and indexing embeddings...")
        vector_db.ensure_collection(recreate=True)
        
        # Embed chunks
        enriched_chunks = embedder.embed_chunks(chunks)
        
        # Store in Vector DB
        vector_db.upsert(enriched_chunks)
        print(f"[INFO] Indexing completed. {vector_db.count()} points stored.")
    else:
        print("\n[STEP 2 & 3] Skipping parsing and re-indexing. Using existing vector index.")
        # Ensure the collection exists even if we skip re-indexing, without deleting existing data
        vector_db.ensure_collection(recreate=False)

    # 7. Initialize Orchestrator Agent
    print("\n[STEP 4] Initializing Orchestrator Agent...")
    agent = OrchestratorAgent(
        repo_path=local_path,
        embedder=embedder,
        vector_db=vector_db,
        model=llm_model,
        max_revision_attempts=1,
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
