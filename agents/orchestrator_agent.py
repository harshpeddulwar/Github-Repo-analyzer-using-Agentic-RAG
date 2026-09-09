from typing import Any, Dict, List

from agents.retrieval_agent import RetrievalAgent

import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]


class OrchestratorAgent:
    """
    Coordinates RetrievalAgent to produce an answer to a question about a
    codebase.

    Flow:
      1. RetrievalAgent gathers evidence via its tools and drafts an answer.
      2. The answer and the underlying tool-call trace are returned as-is.
    """

    def __init__(
        self,
        repo_path: str,
        embedder,
        vector_db,
        model: str = model,
        max_revision_attempts: int = 1,
    ):
        self.retrieval_agent = RetrievalAgent(
            repo_path=repo_path, embedder=embedder, vector_db=vector_db, model=model
        )
        self.max_revision_attempts = max_revision_attempts

    def answer(self, query: str) -> Dict[str, Any]:
        result = self.retrieval_agent.ask(query)
        draft = result["answer"]
        evidence = self._format_evidence(result["steps"])

        return {
            "answer": draft,
            "evidence": evidence,
        }

    @staticmethod
    def _format_evidence(steps: List[Dict[str, Any]]) -> str:
        if not steps:
            return "No tool calls were made."

        blocks = []
        for i, step in enumerate(steps, start=1):
            blocks.append(
                f"[Tool call {i}] {step.get('tool')}({step.get('args')})\n"
                f"Result:\n{step.get('result', '(no result recorded)')}"
            )
        return "\n\n".join(blocks)