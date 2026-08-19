from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class QdrantStore:
    def __init__(
        self,
        collection_name: str,
        vector_size: int = 768,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        distance: qm.Distance = qm.Distance.COSINE,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance
        self.client = QdrantClient(url=url, api_key=api_key)

   
    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if it doesn't exist. If recreate=True, drop
        and recreate it (useful when the embedding model/dim changes).
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
            print(f"[INFO] Collection '{self.collection_name}' already exists")


    @staticmethod
    def _point_id(chunk: Dict[str, Any]) -> str:
        """
        chunk_id is now a UUID string from EmbeddingPipe. Qdrant accepts
        UUID strings directly as point IDs, so no conversion is needed --
        same chunk always maps to the same point, enabling clean upserts
        on re-embedding.
        """
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            raise ValueError("Chunk is missing 'chunk_id'; run embed_chunks() first.")
        return str(chunk_id)

    @staticmethod
    def _payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Everything except the embedding vector itself, safe for JSON."""
        return {k: v for k, v in chunk.items() if k != "embedding"}

 
    def upsert(
        self,
        enriched_chunks: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        """
        Upsert enriched chunks (dicts with 'chunk_id' and 'embedding') into
        Qdrant in batches. Returns the number of points upserted.
        """
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
                    vector=list(chunk["embedding"]),
                    payload=self._payload(chunk),
                )
                for chunk in batch
            ]

            self.client.upsert(collection_name=self.collection_name, points=points)
            upserted += len(points)
            print(f"[INFO] Upserted {min(upserted, total)}/{total} points")

        print(f"[INFO] Done. {upserted} points stored in '{self.collection_name}'")
        return upserted


    def delete_by_file(self, file_path: str) -> None:
        """Delete all points belonging to a given source file (for re-indexing)."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="file_path", match=qm.MatchValue(value=file_path)
                        )
                    ]
                )
            ),
        )
        print(f"[INFO] Deleted existing points for file: {file_path}")

    def count(self) -> int:
        return self.client.count(self.collection_name, exact=True).count


    def search(self,query_vector: List[float],limit: int = 5,) -> List[Dict[str, Any]]:

        """
     Search Qdrant for the most similar chunks.
        """

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
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
        )

        results = []

        for point in response.points:
          results.append(
             {
                 "id": str(point.id),
                 "score": point.score,
                 "payload": point.payload,
             }
          )

        return results


