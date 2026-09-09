from agents.orchestrator_agent import OrchestratorAgent
from src.embedding import EmbeddingPipe
from src.VectorDB import QdrantStore
import os

from dotenv import load_dotenv
load_dotenv()


import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]
# Use the same objects you already use to initialize your project.
# Replace these with your actual repo path / embedder / vector DB setup.

repo_path = os.path.join("repos", "RAG")  # Replace with your actual repo path

embedder = EmbeddingPipe()
vector_db = QdrantStore(collection_name="rag_repo_collection")  # Replace with your actual vector DB setup

agent = OrchestratorAgent(
    repo_path=repo_path,
    embedder=embedder,
    vector_db=vector_db,
    model=model,
    max_revision_attempts=1,
)

while True:
    query = input("Enter your query :")

    if query.lower() == "exit":
        break   

    result = agent.answer(query)
    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("TOOL CALLS")
    print("=" * 80)