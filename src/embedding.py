import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
import uuid

class EmbeddingPipe:
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: Optional[int] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading model '{model_name}' on device: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=self.device,
        )

        # Jina v2 code model supports up to 8192 tokens; make sure we're not
        # silently truncating at a smaller library default.
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
            print(f"[INFO] max_seq_length set to {self.model.max_seq_length}")

        self.batch_size = batch_size

    # Chunk ID

    def make_chunk_id(self, chunk: Dict[str, Any]) -> str:
        """
        Build a stable ID for a chunk so embeddings can be mapped back to
        chunks and re-embedded/upserted deterministically when a file changes.
        """
        file_path = chunk.get("file_path", "unknown")
        start_line = chunk.get("start_line", "")
        end_line = chunk.get("end_line", "")
        name = chunk.get("name", "")
        chunk_type = chunk.get("type", "")

        raw = f"{file_path}:{chunk_type}:{name}:{start_line}-{end_line}"
        # return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] --> this makes hexIDs which qdrant use.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))  #--> this makes uuid

    # Text preparation
    def prepare_chunk(self, chunk: Dict[str, Any]) -> str:
        """
        Convert a structured parser chunk into an enriched string
        for embedding.
        """
        chunk_type = chunk.get("type", "unknown")
        name = chunk.get("name", "")
        file_path = chunk.get("file_path", "unknown")
        language = chunk.get("language", "unknown")
        start_line = chunk.get("start_line", "")
        end_line = chunk.get("end_line", "")
        code = chunk.get("text", "")

        parts = [
            f"File: {file_path}",
            f"Language: {language}",
            f"Type: {chunk_type}",
        ]
        if name:
            parts.append(f"Name: {name}")
        if start_line and end_line:
            parts.append(f"Lines: {start_line}-{end_line}")
        parts.append("")
        parts.append("Content:")
        parts.append(code)
        return "\n".join(parts)

    def _filter_valid_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Drop chunks with empty/whitespace-only text (or missing text field)
        so they don't produce noise vectors. Returns (valid, skipped).
        """
        valid, skipped = [], []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text and text.strip():
                valid.append(chunk)
            else:
                skipped.append(chunk)
        return valid, skipped

    # Embedding

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embeds chunks and returns a list of dicts, each the original chunk
        plus 'chunk_id' and 'embedding' keys, so embeddings stay reliably
        aligned to their source chunk regardless of downstream filtering,
        sorting, or dedup.
        """
        if not chunks:
            return []

        valid_chunks, skipped_chunks = self._filter_valid_chunks(chunks)

        if skipped_chunks:
            print(
                f"[INFO] Skipping {len(skipped_chunks)} empty/whitespace-only "
                f"chunk(s) before embedding."
            )

        if not valid_chunks:
            print("[WARN] No valid (non-empty) chunks to embed.")
            return []

        # Assign stable IDs up front
        for chunk in valid_chunks:
            chunk["chunk_id"] = self.make_chunk_id(chunk)

        texts = [self.prepare_chunk(chunk) for chunk in valid_chunks]
        effective_batch_size = batch_size or self.batch_size

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        results: List[Dict[str, Any]] = []

        # Embed in explicit batches (rather than one giant model.encode call)
        # so a bad chunk in one batch doesn't take down the whole run.
        for start in range(0, len(texts), effective_batch_size):
            end = start + effective_batch_size
            batch_texts = texts[start:end]
            batch_chunks = valid_chunks[start:end]

            try:
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=effective_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            except Exception as exc:
                print(
                    f"[WARN] Batch {start}-{end} failed to embed as a group "
                    f"({exc}). Retrying chunks individually..."
                )
                batch_embeddings = []
                for i, text in enumerate(batch_texts):
                    try:
                        emb = self.model.encode(
                            [text],
                            batch_size=1,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )[0]
                        batch_embeddings.append(emb)
                    except Exception as inner_exc:
                        chunk_id = batch_chunks[i].get("chunk_id", "unknown")
                        file_path = batch_chunks[i].get("file_path", "unknown")
                        print(
                            f"[ERROR] Skipping chunk {chunk_id} in {file_path}: "
                            f"{inner_exc}"
                        )
                        batch_embeddings.append(None)

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                if embedding is None:
                    continue
                enriched = dict(chunk)
                enriched["embedding"] = embedding
                results.append(enriched)

            print(f"[INFO] Embedded {min(end, len(texts))}/{len(texts)} chunks")

        print(f"[INFO] Successfully embedded {len(results)}/{len(chunks)} input chunks")
        return results

    
    
    def embed_query(self, query: str) -> List[float]:
      """
      Convert a user's natural-language query into an embedding
      using the same model and normalization as the stored chunks.
      """
      if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

      embedding = self.model.encode(
        query.strip(),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
      )

      return embedding.tolist()


   

    def embed_chunks_matrix(
        self, chunks: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Convenience wrapper for callers that want a chunk list plus a plain
        (N, dim) embedding matrix, still guaranteed aligned by construction.
        """
        enriched = self.embed_chunks(chunks, batch_size=batch_size)
        if not enriched:
            return [], np.empty((0, 0))
        matrix = np.stack([c["embedding"] for c in enriched])
        return enriched, matrix



