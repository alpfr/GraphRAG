from typing import List, Dict, Any
from langchain_core.tools import tool
from config.settings import settings

# Global references to avoid circular imports or passing through LangChain's strict tool decorators.
# These will be set by the GraphragAgent during initialization.
_opensearch_client = None
_neo4j_client = None
_embedding_model = None

def set_clients(opensearch_client, neo4j_client, embedding_model):
    global _opensearch_client, _neo4j_client, _embedding_model
    _opensearch_client = opensearch_client
    _neo4j_client = neo4j_client
    _embedding_model = embedding_model

@tool
def vector_search(query: str, k: int = 5) -> str:
    """
    Search the vector database for document chunks matching the query conceptually. 
    Use this tool to read the underlying text of the documents to answer specific questions or retrieve raw context.
    
    Args:
        query: The natural language search query.
        k: The number of results to return (default 5).
    """
    if not _opensearch_client or not _embedding_model:
        return "Error: Tools are not initialized with clients."
        
    try:
        # Generate embedding for the query
        query_vector = _embedding_model.embed_query(query)
        
        # Search OpenSearch
        results = _opensearch_client.search(query_embedding=query_vector, k=k)
        
        if not results:
            return "No relevant documents found."
            
        formatted_results = []
        for i, hit in enumerate(results):
            text = hit.get('text', '')
            file_name = hit.get('file_name', 'Unknown')
            formatted_results.append(f"[Source: {file_name}, Chunk: {i+1}]\n{text}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error executing vector search: {str(e)}"

@tool
def graph_traversal(entity_name: str) -> str:
    """
    Query the Neo4j Knowledge Graph to find connections and relationships for a specific entity.
    Use this tool to find out "who is connected to who", "what concepts relate to X", or to traverse the web of concepts.
    
    Args:
        entity_name: The name of the entity to search for (e.g., "Bob", "Python", "GraphRAG").
    """
    if not _neo4j_client:
        return "Error: Graph client not initialized."
        
    try:
        with _neo4j_client.driver.session() as session:
            # Simple Cypher query to retrieve immediate connections to the entity
            query = """
                MATCH (e:Entity)-[r]-(n)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e.name as primary_entity, type(r) as relationship, labels(n)[0] as connected_type, n.name as connected_entity, n.file_name as file_source
                LIMIT 50
            """
            result = session.run(query, name=entity_name)
            
            connections = []
            for record in result:
                primary = record["primary_entity"]
                rel = record["relationship"]
                conn_type = record["connected_type"]
                
                # Depending on what node it connected to, handle the display
                if conn_type == "Document":
                    conn_name = record["file_source"]
                elif conn_type == "Chunk":
                    conn_name = "Text Chunk"
                else:
                    conn_name = record["connected_entity"] or "Unknown"
                    
                connections.append(f"({primary}) -[{rel}]-> ({conn_type}: {conn_name})")
            
            if not connections:
                return f"No graph relationships found for entity '{entity_name}'."
                
            return "\n".join(connections)
    except Exception as e:
        return f"Error executing graph traversal: {str(e)}"
