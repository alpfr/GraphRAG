import logging
from typing import Dict, Any, List, Optional
import json
from src.graphrag.neo4j_client import Neo4jClient
from src.llm.factory import LLMFactory
from src.llm.prompts import get_extraction_prompt
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds the knowledge graph from documents and chunks."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.neo4j = neo4j_client
        self.provider = provider
        self.llm = LLMFactory.get_chat_model(provider=provider, model_name=model_name)

    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Uses the LLM to extract structured entities from raw text."""
        prompt = get_extraction_prompt(self.provider or "ollama", text)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # remove ```json if it wrapped it
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            entities = json.loads(content)
            if isinstance(entities, dict):
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
