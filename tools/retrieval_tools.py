import os
from typing import Optional

from src.cloning import walk_repository
from tools.retrieval_optimizer_tool import RetrievalOptimizer


DEFAULT_MAX_READ_LINES = 200

class RetrievalTools:
    """
    Holds repo-scoped retrieval logic (semantic search, file listing, file reading).

    NOTE: These are plain instance methods, not `@tool`-decorated functions.
    `@tool` from langchain_core.tools is meant for free functions - decorating a
    bound method with it causes LangChain to try to infer a schema that includes
    `self` as a parameter, which breaks tool-calling. Instead, wrap the bound
    methods with `langchain_core.tools.StructuredTool.from_function(...)` after
    instantiating this class (see retrieval_agent.py).
    """

    def __init__(self, repo_path: str, embedder, vector_db, optimizer: Optional[RetrievalOptimizer] = None,):
        self.repo_path = os.path.abspath(repo_path)
        self.embedder = embedder
        self.vector_db = vector_db
        self.optimizer = optimizer or RetrievalOptimizer(
             final_result_count=4,
             max_chunks_per_file=2,
             min_rerank_score=-2.0,
        )

    def semantic_code_search(self, query: str) -> str:
        """
        Search the repository for code chunks that are semantically related to
        a natural-language query (e.g. "where do we validate user input",
        "retry logic for API calls"). Use this when you don't know exact file
        names or line numbers and want to find conceptually relevant code.
        Returns up to 6 matching chunks with file path, line range, and content.
        """
        try:
            query_vector = self.embedder.embed_query(query)

            raw_results = self.vector_db.search(
                query_vector=query_vector,
                limit=20
            )

            if not raw_results:
                return "No semantically matching code chunks found."

            candidates = [
                {
                    "payload":res.get("payload",{}),
                    "_vector_score": res.get("score", 0),
                }
                for res in raw_results
            ]

            # formatted_chunks = []

            # for i, res in enumerate(results, start=1):
            #     payload = res.get("payload", {})

            #     formatted_chunks.append(
            #         f"[{i}] File: {payload.get('file_path')}\n"
            #         f"Lines: {payload.get('start_line')}-{payload.get('end_line')}\n"
            #         f"Type: {payload.get('type')} | Name: {payload.get('name')}\n"
            #         f"Content:\n{payload.get('text')}\n"
            #         f"---"
            #     )

            return self.optimizer.optimize(
                query=query,
                candidates=candidates,
                max_tokens=3000
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

            # Return paths relative to repo root
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

        Optionally specify start_line and end_line
        (1-indexed, inclusive) to inspect only part of the file.
        """
        try:
            # Resolve the requested path
            full_path = os.path.abspath(
                os.path.join(self.repo_path, file_path)
            )

            # Prevent path traversal outside the repository
            if os.path.commonpath(
                [self.repo_path, full_path]
            ) != self.repo_path:
                return "Error: File path is outside the repository."

            if not os.path.isfile(full_path):
                return f"Error: File '{file_path}' does not exist in this repository."

            with open(
                full_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:
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

            slice_lines = lines[start - 1:end]
            content = "".join(slice_lines)

            return (
                f"--- File: {file_path} "
                f"(Lines {start}-{end} of {total_lines}) ---\n"
                f"{content}"
            )

        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"