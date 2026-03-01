import logging
from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch, helpers
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """Client for interacting with OpenSearch vector database."""

    def __init__(self, embedding_dimension: int = 768):
        self.embedding_dimension = embedding_dimension
        self.client = OpenSearch(
            hosts=[{'host': settings.opensearch_host, 'port': settings.opensearch_port, 'scheme': 'https'}],
            http_auth=(settings.opensearch_user, settings.opensearch_password),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_name = settings.opensearch_index_name
        self._ensure_index_exists()
        logger.info(f"OpenSearch client initialized (dimension={self.embedding_dimension})")

    def _ensure_index_exists(self):
        """Creates the index with k-NN vector settings if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "document_id": {"type": "keyword"},
                        "file_name": {"type": "keyword"},
                        "file_path": {"type": "keyword"},
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dimension,  # Dynamic: matches selected embedding model
                            "method": {
                                "name": "hnsw",
                                "space_type": "l2",
                                "engine": "nmslib"
                            }
                        },
                        "metadata": {"type": "object"}
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"Created OpenSearch index: {self.index_name}")

    def index_document(
        self,
        document_id: str,
        file_name: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Indexes a full document with its chunks and embeddings using the bulk API."""
        actions = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            action = {
                "_index": self.index_name,
                "_id": f"{document_id}_chunk_{i}",
                "_source": {
                    "chunk_id": i,
                    "document_id": document_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "text": chunk["text"],
                    "embedding": embedding,
                    "metadata": {**(chunk.get("metadata", {})), **(metadata or {})}
                }
            }
            actions.append(action)
            
        success, failed = helpers.bulk(self.client, actions)
        logger.info(f"Indexed {success} chunks into OpenSearch. Failed: {failed}")
        return success

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Performs a strict k-NN visual similarity search on embeddings."""
        query = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            },
            "_source": ["chunk_id", "document_id", "file_name", "text", "metadata"]
        }
        
        response = self.client.search(
            body=query,
            index=self.index_name
        )
        
        hits = response["hits"]["hits"]
        results = []
        for hit in hits:
            source = hit["_source"]
            results.append({
                "score": hit["_score"],
                "chunk_id": source.get("chunk_id"),
                "document_id": source.get("document_id"),
                "file_name": source.get("file_name"),
                "text": source.get("text"),
                "metadata": source.get("metadata")
            })
        return results

    def get_document_count(self) -> int:
        """Counts the total amount of indexed chunks."""
        self.client.indices.refresh(index=self.index_name)
        return self.client.count(index=self.index_name)["count"]

    def get_index_dimension(self) -> Optional[int]:
        """Get the embedding dimension of the current index."""
        try:
            if self.client.indices.exists(index=self.index_name):
                mapping = self.client.indices.get_mapping(index=self.index_name)
                props = mapping[self.index_name]["mappings"]["properties"]
                return props.get("embedding", {}).get("dimension")
        except Exception as e:
            logger.warning(f"Could not read index dimension: {e}")
        return None

    def is_dimension_compatible(self, new_dimension: int) -> bool:
        """Check if a new embedding dimension is compatible with the existing index."""
        current = self.get_index_dimension()
        if current is None:
            return True  # No index exists yet
        return current == new_dimension
