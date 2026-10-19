import warnings
from typing import Any, Dict, Iterable, List, Optional, Set

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class QdrantStore:
    """
    Thin wrapper around one Qdrant collection shared by many repositories.

    Every point carries a `repo_id` payload field. All reads (search/count)
    and cleanups (delete_stale/delete_by_repo) are scoped by it, so repos can
    never see or wipe each other's vectors.
    """

    REPO_FIELD = "repo_id"

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 768,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        distance: qm.Distance = qm.Distance.COSINE,
        path: Optional[str] = None,
        client: Optional[QdrantClient] = None,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        if client is not None:
            self.client = client
        elif path:
            # Embedded mode: no server needed, data lives in `path`.
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if missing (plus a payload index on repo_id).

        recreate=True drops EVERY repo's data -- only use it when the
        embedding model/dimension changes. Normal re-indexing should use
        upsert() + delete_stale(), which never leaves a gap.
        """
        exists = self.client.collection_exists(self.collection_name)

        if exists and recreate:
            print(f"[INFO] Recreating collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            print(
                f"[INFO] Creating collection '{self.collection_name}' "
                f"(dim={self.vector_size}, distance={self.distance})"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(
                    size=self.vector_size, distance=self.distance
                ),
            )
        else:
            self._check_dimension()

        self._ensure_repo_index()

    def _check_dimension(self) -> None:
        try:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            size = getattr(vectors, "size", None)
            if size is not None and size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' has dim={size} but the "
                    f"configured vector_size is {self.vector_size}. Change the config "
                    f"or recreate the collection (ensure_collection(recreate=True))."
                )
        except ValueError:
            raise
        except Exception:
            pass  # informational check only

    def _ensure_repo_index(self) -> None:
        """Keyword index so repo_id filters are fast on a server. No-op in embedded mode."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # embedded mode warns that indexes are no-ops
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=self.REPO_FIELD,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _repo_filter(self, repo_id: Optional[str]) -> Optional[qm.Filter]:
        if not repo_id:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key=self.REPO_FIELD, match=qm.MatchValue(value=repo_id))]
        )

    @staticmethod
    def _point_id(chunk: Dict[str, Any]) -> str:
        """chunk_id is a UUID string from EmbeddingPipe (seeded with repo_id too)."""
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("Chunk is missing 'chunk_id'; run embed_chunks() first.")
        return str(chunk_id)

    @staticmethod
    def _payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Everything except the embedding vector itself, safe for JSON."""
        return {k: v for k, v in chunk.items() if k != "embedding"}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert(self, enriched_chunks: List[Dict[str, Any]], batch_size: int = 256) -> int:
        """Upsert enriched chunks into Qdrant in batches. Returns points upserted."""
        if not enriched_chunks:
            print("[WARN] No chunks to upsert.")
            return 0

        total = len(enriched_chunks)
        upserted = 0

        for start in range(0, total, batch_size):
            batch = enriched_chunks[start : start + batch_size]
            points = [
                qm.PointStruct(
                    id=self._point_id(chunk),
                    vector=[float(x) for x in chunk["embedding"]],
                    payload=self._payload(chunk),
                )
                for chunk in batch
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            upserted += len(points)
            print(f"[INFO] Upserted {min(upserted, total)}/{total} points")

        print(f"[INFO] Done. {upserted} points stored in '{self.collection_name}'")
        return upserted

    def delete_stale(self, repo_id: str, keep_ids: Iterable[str]) -> int:
        """
        Delete this repo's points whose id is NOT in keep_ids. Call it AFTER a
        successful upsert: new data is already live, so there is never a window
        where the repo has no index. Other repos are untouched.
        """
        keep: Set[str] = {str(i) for i in keep_ids}
        stale: List[Any] = []
        offset = None
        flt = self._repo_filter(repo_id)
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            stale.extend(p.id for p in points if str(p.id) not in keep)
            if offset is None:
                break

        for i in range(0, len(stale), 512):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.PointIdsList(points=stale[i : i + 512]),
            )
        if stale:
            print(f"[INFO] Removed {len(stale)} stale point(s) for repo '{repo_id}'")
        return len(stale)

    def delete_by_repo(self, repo_id: str) -> None:
        """Remove every point of one repository (other repos untouched)."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(filter=self._repo_filter(repo_id)),
        )
        print(f"[INFO] Deleted all points for repo: {repo_id}")

    def delete_by_file(self, file_path: str, repo_id: Optional[str] = None) -> None:
        """Delete all points of one source file (optionally within one repo)."""
        must = [qm.FieldCondition(key="file_path", match=qm.MatchValue(value=file_path))]
        if repo_id:
            must.append(
                qm.FieldCondition(key=self.REPO_FIELD, match=qm.MatchValue(value=repo_id))
            )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=must)),
        )
        print(f"[INFO] Deleted existing points for file: {file_path}")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def count(self, repo_id: Optional[str] = None) -> int:
        return self.client.count(
            self.collection_name, count_filter=self._repo_filter(repo_id), exact=True
        ).count

    def has_repo(self, repo_id: str) -> bool:
        try:
            return self.count(repo_id) > 0
        except Exception:
            return False

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        repo_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for the most similar chunks, restricted to `repo_id` when given."""
        if not query_vector:
            raise ValueError("Query vector cannot be empty.")
        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"expected dimension {self.vector_size}."
            )
        if limit <= 0:
            raise ValueError("limit must be greater than 0.")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._repo_filter(repo_id),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {"id": str(p.id), "score": p.score, "payload": p.payload}
            for p in response.points
        ]
