from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from loguru import logger

from config.settings import settings

class Neo4jClient:
    """Client for interacting with Neo4j graph database."""
    
    def __init__(self):
        """Initialize Neo4j client."""
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self._verify_connectivity()
        logger.info("Neo4j client initialized")
    
    def _verify_connectivity(self):
        """Verify connection to Neo4j."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            raise
    
    def close(self):
        """Close the driver connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def _flatten_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Flatten nested metadata dictionaries for Neo4j compatibility.
        Neo4j only accepts primitive types and arrays as property values.
        """
        if not metadata:
            return {}
        
        flattened = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flat_key = f"{key}_{nested_key}"
                    if isinstance(nested_value, (str, int, float, bool)):
                        flattened[flat_key] = nested_value
                    else:
                        flattened[flat_key] = str(nested_value)
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, list):
                flattened[key] = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in value]
            else:
                flattened[key] = str(value)
        
        return flattened
    
    def create_document_node(
        self, document_id: str, file_name: str, file_path: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        flat_metadata = self._flatten_metadata(metadata)
        with self.driver.session() as session:
            query = """
            MERGE (d:Document {id: $document_id})
            SET d.file_name = $file_name,
                d.file_path = $file_path,
                d += $metadata,
                d.created_at = datetime()
            RETURN d
            """
            result = session.run(query, document_id=document_id, file_name=file_name, file_path=file_path, metadata=flat_metadata)
            return dict(result.single()["d"])
    
    def create_chunk_node(
        self, document_id: str, chunk_id: int, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        flat_metadata = self._flatten_metadata(metadata)
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {id: $document_id})
            CREATE (c:Chunk {id: $chunk_id, document_id: $document_id})
            SET c.text = $text,
                c += $metadata,
                c.created_at = datetime()
            CREATE (d)-[:HAS_CHUNK]->(c)
            RETURN c
            """
            result = session.run(query, document_id=document_id, chunk_id=chunk_id, text=text, metadata=flat_metadata)
            return dict(result.single()["c"])
    
    def create_entity_node(
        self, entity_name: str, entity_type: str, properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        with self.driver.session() as session:
            query = f"""
            MERGE (e:Entity:{entity_type} {{name: $entity_name}})
            SET e += $properties,
                e.created_at = coalesce(e.created_at, datetime())
            RETURN e
            """
            result = session.run(query, entity_name=entity_name, properties=properties or {})
            return dict(result.single()["e"])
    
    def create_relationship(
        self, from_node_id: str, to_node_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None
    ):
        with self.driver.session() as session:
            def convert_id(id_val):
                try:
                    return int(id_val)
                except (ValueError, TypeError):
                    return id_val
            
            from_id_int = convert_id(from_node_id)
            to_id_int = convert_id(to_node_id)
            
            query = f"""
            MATCH (a)
            WHERE a.id = $from_id OR a.id = $from_id_int OR a.name = $from_id
            MATCH (b)
            WHERE b.id = $to_id OR b.id = $to_id_int OR b.name = $to_id
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            RETURN r
            """
            result = session.run(
                query, from_id=from_node_id, from_id_int=from_id_int, to_id=to_node_id, to_id_int=to_id_int, properties=properties or {}
            )
            if not result.single():
                logger.warning(f"Could not create relationship {relationship_type} from {from_node_id} to {to_node_id}")
