import os
import sys

from dotenv import load_dotenv


from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore

COLLECTION_NAME = "rag_repo_collection"
TOP_K = 5
load_dotenv()


def get_groq_api_key():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        print("\n[WARNING] GROQ_API_KEY environment variable not found.")
        groq_key = input("Please enter your Groq API Key: ").strip()
        if not groq_key:
            print("[ERROR] Groq API Key is required.")
            sys.exit(1)
        os.environ["GROQ_API_KEY"] = groq_key
    else:
        print("\n[OK] GROQ_API_KEY found in environment.")


def format_context(results):
    context_parts = []
    print("\n" + "-" * 40)
    print("RETRIEVED CONTEXT CHUNKS:")
    print("-" * 40)

    for i, res in enumerate(results, 1):
        score = res.get("score", 0.0)
        payload = res.get("payload", {})
        file_path = payload.get("file_path", "N/A")
        start_line = payload.get("start_line", "N/A")
        end_line = payload.get("end_line", "N/A")
        chunk_type = payload.get("type", "N/A")
        chunk_name = payload.get("name", "")
        text = payload.get("text", "")

        meta_str = (
            f"[{i}] File: {file_path} | Lines: {start_line}-{end_line} | "
            f"Type: {chunk_type}"
        )
        if chunk_name:
            meta_str += f" | Name: {chunk_name}"
        meta_str += f" | Score: {score:.4f}"
        print(meta_str)

        block = f"Source: {file_path} (Lines {start_line}-{end_line})\n"
        if chunk_name:
            block += f"Entity: {chunk_name} ({chunk_type})\n"
        block += f"Content:\n{text}\n"
        context_parts.append(block)

    print("-" * 40)
    return "\n---\n".join(context_parts)


def generate_answer(context, question):
    print("\n[LLM] Connecting to Groq and generating answer...")
    try:
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.0,
        )

        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert technical assistant. Answer the user's "
                "question about the codebase using only the provided context "
                "chunks. Cite the specific file name(s) and lines when "
                "referencing code details. If the context does not contain "
                "enough information, explain what is missing.",
            ),
            (
                "human",
                "Context Chunks:\n{context}\n\nQuestion: {question}",
            ),
        ])

        chain = prompt_template | llm
        response = chain.invoke({"context": context, "question": question})

        print("\n" + "=" * 60)
        print("LLM RESPONSE:")
        print("=" * 60)
        print(response.content)
        print("=" * 60)
    except Exception as e:
        print(f"[ERROR] LLM generation failed: {e}")


def main():
    print("=" * 80)
    print("              AGENTIC RAG - RETRIEVAL TEST RUNNER")
    print("=" * 80)

    get_groq_api_key()

    print("\n[1] Loading embedding model...")
    embedder = EmbeddingPipe()
    print("[OK] Embedding model loaded.")

    print("\n[2] Connecting to Qdrant...")
    vector_db = QdrantStore(
        collection_name=COLLECTION_NAME,
        vector_size=768,
    )
    print("[OK] Connected to Qdrant.")

    print("\n[3] Checking collection...")
    vector_db.ensure_collection()
    count = vector_db.count()
    print(f"[OK] Collection: {COLLECTION_NAME}")
    print(f"[INFO] Total vectors: {count}")

    if count == 0:
        print("[WARNING] Collection contains no vectors.")
        print("[INFO] Run your ingestion/embedding pipeline first.")
        return

    print("\n" + "=" * 50)
    print("Ready for queries! Enter 'exit' or 'quit' to stop.")
    print("=" * 50)

    while True:
        try:
            query = input("\nEnter your search query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        print("\n[4] Embedding query...")
        try:
            query_vector = embedder.embed_query(query)
        except Exception as e:
            print(f"[ERROR] Query embedding failed: {e}")
            continue

        print("[OK] Query embedded.")
        print(f"[INFO] Vector dimension: {len(query_vector)}")

        print("\n[5] Searching Qdrant...")
        try:
            results = vector_db.search(query_vector=query_vector, limit=TOP_K)
        except Exception as e:
            print(f"[ERROR] Qdrant search failed: {e}")
            continue

        print("\n" + "=" * 70)
        print("RETRIEVAL RESULTS")
        print("=" * 70)

        if not results:
            print("No results found.")
            continue

        for i, result in enumerate(results, start=1):
            payload = result.get("payload", {})
            print(f"\n--- Result {i} ---")
            print(f"Score     : {result.get('score')}")
            print(f"File      : {payload.get('file_path', 'N/A')}")
            print(f"Type      : {payload.get('type', 'N/A')}")
            print(f"Name      : {payload.get('name', 'N/A')}")
            print(
                f"Lines     : {payload.get('start_line', 'N/A')}-"
                f"{payload.get('end_line', 'N/A')}"
            )
            print("\nContent:")
            print(payload.get("text", "No content available."))

        context_str = format_context(results)
        generate_answer(context_str, query)


if __name__ == "__main__":
    main()
