import os
from typing import Optional

from src.cloning import ALLOWED_EXTENSIONS, walk_repository
from tools.retrieval_optimizer_tool import RetrievalOptimizer


DEFAULT_MAX_READ_LINES = 200


class RetrievalTools:
    """
    Holds repo-scoped retrieval logic (semantic search, file listing, file reading).

    NOTE: These are plain instance methods, not `@tool`-decorated functions.
    `@tool` from langchain_core.tools is meant for free functions - decorating a
    bound method with it makes LangChain infer a schema that includes `self`,
    which breaks tool-calling. Instead the bound methods are wrapped with
    `StructuredTool.from_function(...)` in retrieval_agent.py.
    """

    def __init__(
        self,
        repo_path: str,
        embedder,
        vector_db,
        repo_id: Optional[str] = None,
        optimizer: Optional[RetrievalOptimizer] = None,
        candidate_limit: int = 20,
        final_result_count: int = 4,
        max_chunks_per_file: int = 2,
        min_rerank_score: float = -2.0,
        max_context_tokens: int = 3000,
    ):
        # realpath (not abspath) so symlinks inside a repo cannot escape it.
        self.repo_path = os.path.realpath(repo_path)
        self.repo_id = repo_id or os.path.basename(os.path.normpath(repo_path))
        self.embedder = embedder
        self.vector_db = vector_db
        self.candidate_limit = candidate_limit
        self.max_context_tokens = max_context_tokens
        self.optimizer = optimizer or RetrievalOptimizer(
            final_result_count=final_result_count,
            max_chunks_per_file=max_chunks_per_file,
            min_rerank_score=min_rerank_score,
        )

    def semantic_code_search(self, query: str) -> str:
        """
        Search the repository for code chunks that are semantically related to
        a natural-language query (e.g. "where do we validate user input",
        "retry logic for API calls"). Use this when you don't know exact file
        names or line numbers and want to find conceptually relevant code.
        Returns the few best chunks with file path, line range, and content.
        """
        try:
            query_vector = self.embedder.embed_query(query)

            # Scoped to THIS repository only (repo_id filter).
            raw_results = self.vector_db.search(
                query_vector=query_vector,
                limit=self.candidate_limit,
                repo_id=self.repo_id,
            )

            if not raw_results:
                return (
                    "No semantically matching code chunks found. "
                    f"(If this repo was never indexed, index '{self.repo_id}' first.)"
                )

            candidates = [
                {
                    "payload": res.get("payload", {}),
                    "_vector_score": res.get("score", 0),
                }
                for res in raw_results
            ]

            return self.optimizer.optimize(
                query=query,
                candidates=candidates,
                max_tokens=self.max_context_tokens,
            )

        except Exception as e:
            return f"Error during semantic search: {str(e)}"

    def list_repo_files(self) -> str:
        """
        List all source code and markdown files present in the repository.
        Helps map out the project layout and find filenames.
        """
        try:
            files = walk_repository(self.repo_path)

            rel_files = [
                os.path.relpath(f, self.repo_path).replace("\\", "/")
                for f in files
            ]

            if not rel_files:
                return "No source files found in this repository."

            return "\n".join(rel_files)

        except Exception as e:
            return f"Error listing repository files: {str(e)}"

    def read_file_content(
        self,
        file_path: str,
        start_line: Optional[int] = 1,
        end_line: Optional[int] = None
    ) -> str:
        """
        Read the contents of a specific file inside the repository.

        Optionally specify start_line and end_line (1-indexed, inclusive) to
        inspect only part of the file. Every returned line is prefixed with its
        real line number ("  42| code"), so cite lines exactly as shown.
        """
        try:
            full_path = os.path.realpath(os.path.join(self.repo_path, file_path))

            # Prevent path traversal / symlink escape outside the repository
            if os.path.commonpath([self.repo_path, full_path]) != self.repo_path:
                return "Error: File path is outside the repository."

            if not os.path.isfile(full_path):
                return f"Error: File '{file_path}' does not exist in this repository."

            # Only the same file types the indexer accepts (keeps .env, keys, binaries out).
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return (
                    f"Error: '{file_path}' is not a readable source/doc file type "
                    f"(allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))})."
                )

            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)

            start = max(1, start_line or 1)
            if end_line is None:
                end = min(start + DEFAULT_MAX_READ_LINES - 1, total_lines)
            else:
                end = min(end_line, total_lines)

            if start > total_lines:
                return (
                    f"Error: start_line {start} is beyond the end "
                    f"of the file ({total_lines} lines)."
                )

            if end < start:
                return "Error: end_line must be greater than or equal to start_line."

            width = len(str(end))
            numbered = "".join(
                f"{n:>{width}}| {line}"
                if line.endswith("\n")
                else f"{n:>{width}}| {line}\n"
                for n, line in enumerate(lines[start - 1:end], start=start)
            )

            return (
                f"--- File: {file_path} "
                f"(Lines {start}-{end} of {total_lines}) ---\n"
                f"{numbered}"
            )

        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"
