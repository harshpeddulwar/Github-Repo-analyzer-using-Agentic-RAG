from typing import Optional, List, Dict, Any


# Default cross-encoder used for reranking. General-purpose, not code-specific;
# swap via the constructor if you find a better fit for code search.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Minimum cross-encoder score a chunk must clear to be considered relevant.
# Cross-encoder logits are unbounded; this is a conservative floor tuned for
# ms-marco-MiniLM-style models. Tune per-model if you swap the checkpoint.
DEFAULT_MIN_RERANK_SCORE = -2.0

# Max chunks allowed from any single file in the final result set, so one
# large/heavily-matched file doesn't crowd out everything else.
DEFAULT_MAX_CHUNKS_PER_FILE = 2

# Final number of chunks kept after the full pipeline runs.
DEFAULT_FINAL_RESULT_COUNT = 6

# Rough chars-per-token estimate used for budgeting context assembly.
# Avoids a hard dependency on a tokenizer library; good enough for budgeting.
CHARS_PER_TOKEN_ESTIMATE = 4

# Default token budget for the assembled context string.
DEFAULT_MAX_TOKENS = 4000


class RetrievalOptimizer:
    """
    Turns a raw pool of vector-search candidates into a single, ready-to-use
    context string for an agent.

    Pipeline (in order): deduplicate -> rerank -> filter by relevance ->
    diversify (per-file cap) -> assemble into a token-budgeted context block.

    This is an internal collaborator for RetrievalTools, not an agent-facing
    tool. It is never wrapped with StructuredTool / exposed to the LLM
    directly -- RetrievalTools.semantic_code_search calls `optimize()` and
    hands the agent the resulting string.
    """

    def __init__(
        self,
        reranker_model: Optional[str] = DEFAULT_RERANKER_MODEL,
        min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
        max_chunks_per_file: int = DEFAULT_MAX_CHUNKS_PER_FILE,
        final_result_count: int = DEFAULT_FINAL_RESULT_COUNT,
    ):
        self._reranker_model_name = reranker_model
        self._reranker = None  # lazily loaded; False = "tried and failed"

        self.min_rerank_score = min_rerank_score
        self.max_chunks_per_file = max_chunks_per_file
        self.final_result_count = final_result_count

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Runs the full pipeline and returns a formatted, token-budgeted
        context string ready to hand back to the agent.

        `candidates` is a list of dicts shaped like:
            {"payload": {...chunk fields...}, "_vector_score": float}
        as produced directly from a vector_db.search() call, where payload
        contains at least file_path, start_line, end_line, type, name, text.

        Returns a message string (not an exception) if nothing survives the
        pipeline, so callers can return it straight to the agent.
        """
        if not candidates:
            return "No semantically matching code chunks found."

        working = [dict(c) for c in candidates]  # don't mutate caller's list

        working = self._deduplicate(working)
        working = self._rerank(query, working)
        working = self._filter_relevance(working)
        working = self._diversify(working)

        working = working[: self.final_result_count]

        if not working:
            return "No sufficiently relevant code chunks found for this query."

        return self._assemble_context(working, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Stage 1: Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a <= end_b and start_b <= end_a

    def _deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collapses candidates that point at overlapping line ranges in the same
        file (common when a chunker produces sliding-window chunks). Keeps the
        highest vector-similarity candidate from each overlapping group.
        """
        candidates = sorted(candidates, key=lambda c: c.get("_vector_score", 0), reverse=True)

        deduped: List[Dict[str, Any]] = []
        for cand in candidates:
            payload = cand["payload"]
            file_path = payload.get("file_path")
            start = payload.get("start_line") or 0
            end = payload.get("end_line") or 0

            is_duplicate = False
            for kept in deduped:
                kept_payload = kept["payload"]
                if kept_payload.get("file_path") != file_path:
                    continue
                if self._ranges_overlap(
                    start, end,
                    kept_payload.get("start_line") or 0,
                    kept_payload.get("end_line") or 0,
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduped.append(cand)

        return deduped

    # ------------------------------------------------------------------
    # Stage 2: Reranking
    # ------------------------------------------------------------------

    def _get_reranker(self):
        """
        Lazily loads and caches the cross-encoder reranker model.
        Returns None if reranking is disabled (reranker_model=None) or if the
        model fails to load, in which case callers fall back to vector-order.
        """
        if self._reranker_model_name is None:
            return None

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(self._reranker_model_name)
            except Exception:
                # Missing dependency, no network to fetch weights, etc. --
                # degrade gracefully rather than breaking search entirely.
                self._reranker = False

        return self._reranker or None

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores each candidate with a cross-encoder over (query, chunk_text)
        pairs and sorts best-first. Falls back to existing vector-similarity
        order if no reranker is available.
        """
        if not candidates:
            return candidates

        reranker = self._get_reranker()

        if reranker is None:
            for cand in candidates:
                cand["_rerank_score"] = cand.get("_vector_score", 0)
            return sorted(candidates, key=lambda c: c["_rerank_score"], reverse=True)

        pairs = [(query, cand["payload"].get("text") or "") for cand in candidates]
        scores = reranker.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["_rerank_score"] = float(score)

        return sorted(candidates, key=lambda c: c["_rerank_score"], reverse=True)

    # ------------------------------------------------------------------
    # Stage 3: Relevance filtering
    # ------------------------------------------------------------------

    def _filter_relevance(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Drops candidates below the minimum rerank score. Only applied when a
        reranker actually ran -- if we fell back to raw vector scores, those
        aren't on a comparable scale to the configured threshold, so we skip
        filtering rather than filter on a meaningless comparison.
        """
        if self._get_reranker() is None:
            return candidates

        return [c for c in candidates if c.get("_rerank_score", float("-inf")) >= self.min_rerank_score]

    # ------------------------------------------------------------------
    # Stage 4: Diversification
    # ------------------------------------------------------------------

    def _diversify(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assumes `candidates` is already sorted best-first. Walks the list and
        drops any candidate that would exceed max_chunks_per_file for its
        file, so a single heavily-matched file can't crowd out other results.
        """
        counts: Dict[str, int] = {}
        capped: List[Dict[str, Any]] = []

        for cand in candidates:
            file_path = cand["payload"].get("file_path")
            counts.setdefault(file_path, 0)
            if counts[file_path] >= self.max_chunks_per_file:
                continue
            counts[file_path] += 1
            capped.append(cand)

        return capped

    # ------------------------------------------------------------------
    # Stage 5: Context assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunk(index: int, payload: Dict[str, Any]) -> str:
        return (
            f"[{index}] File: {payload.get('file_path')}\n"
            f"Lines: {payload.get('start_line')}-{payload.get('end_line')}\n"
            f"Type: {payload.get('type')} | Name: {payload.get('name')}\n"
            f"Content:\n{payload.get('text')}\n"
            f"---"
        )

    def _assemble_context(
        self,
        candidates: List[Dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Formats the final candidate set into text blocks and packs them into
        a single token-budgeted context string. Candidates are already
        best-first, so lowest-ranked blocks are dropped first if the budget
        would be exceeded. Token count is approximated from character count
        (~CHARS_PER_TOKEN_ESTIMATE chars/token) to avoid a hard tokenizer
        dependency -- treat max_tokens as a soft target, not an exact limit.
        """
        blocks = [
            self._format_chunk(i, cand["payload"])
            for i, cand in enumerate(candidates, start=1)
        ]

        char_budget = max_tokens * CHARS_PER_TOKEN_ESTIMATE
        separator = "\n\n"

        included: List[str] = []
        used_chars = 0
        dropped = 0

        for block in blocks:
            block_cost = len(block) + (len(separator) if included else 0)
            if used_chars + block_cost > char_budget:
                dropped += 1
                continue
            included.append(block)
            used_chars += block_cost

        if not included:
            # Even the first block alone exceeds budget; truncate rather
            # than returning nothing.
            truncated = blocks[0][:char_budget]
            return truncated + "\n\n[... truncated to fit context budget ...]"

        assembled = separator.join(included)
        if dropped:
            assembled += f"\n\n[... {dropped} additional block(s) omitted to fit context budget ...]"

        return assembled
