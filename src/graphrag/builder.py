import logging
from typing import Dict, Any, List
import json
from src.graphrag.neo4j_client import Neo4jClient
from src.rag.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds the knowledge graph from documents and chunks."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.llm = OllamaClient()

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Uses the LLM to extract structured entities from raw text."""
        # Simple extraction prompt for the small granite model
        prompt = (
            "Extract the most important entities (Person, Organization, Location, Technology) "
            "from the following text. Return the result strictly as a JSON array of objects, "
            "where each object has a 'name' and 'type' key. Do not output anything else.\n\n"
            f"Text: {text}\n\nJSON:"
        )

        try:
            # We bypass semantic search here and just execute a generation task
            import requests
            url = f"{self.llm.host}/api/generate"
            payload = {
                "model": self.llm.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "temperature": 0.1
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            entities = json.loads(data["response"])
            if isinstance(entities, dict):
                 # Handle cases where the LLM returned an object instead of array
                 entities = [entities] 
                 
            return entities
            
        except Exception as e:
            logger.warning(f"Failed to extract entities from text chunk: {str(e)}")
            return []

    def build_document_graph(
        self,
        document_id: str,
        file_name: str,
        file_path: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
    ) -> None:
        """Processes a newly added document into the Knowledge Graph."""
        logger.info(f"Building knowledge graph for document {document_id}")
        
        # 1. Create top-level Document Node
        self.neo4j.create_document_node(document_id, file_name, file_path, metadata)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            text = chunk["text"]
            
            # 2. Create Chunk Node and Link to Document (`HAS_CHUNK`)
            self.neo4j.create_chunk_node(document_id, chunk_id, text, chunk.get("metadata"))
            
            # 3. Extract key entities from chunk text using LLM
            entities = self._extract_entities(text)
            
            for entity in entities:
                name = entity.get("name")
                type_ = entity.get("type", "Unknown")
                
                if name:
                    # 4. Create Entity Node 
                    self.neo4j.create_entity_node(name, type_)
                    
                    # 5. Link Chunk to Entity (`MENTIONS`)
                    self.neo4j.create_relationship(
                        from_node_id=chunk_id,
                        to_node_id=name,
                        relationship_type="MENTIONS"
                    )

        logger.info(f"Finished building knowledge graph for document {document_id}")

    def get_graph_summary(self) -> Dict[str, Any]:
         return self.neo4j.get_statistics()
