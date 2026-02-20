import logging
from typing import Optional
from pyvis.network import Network
from src.graphrag.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Visualizes the Neo4j Knowledge Graph using PyVis."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    def _create_base_network(self) -> Network:
        """Initializes a PyVis network with standard physics configuration."""
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
        net.force_atlas_2based()
        return net

    def visualize_entity_graph(self, entity_name: str, max_depth: int = 2, max_nodes: int = 50) -> str:
        """Generates an HTML visualization of an entity and its connections."""
        net = self._create_base_network()
        
        with self.neo4j.driver.session() as session:
            query = f"""
            MATCH path = (e:Entity {{name: $entity_name}})-[*1..{max_depth}]-(related)
            RETURN path
            LIMIT $max_nodes
            """
            result = session.run(query, entity_name=entity_name, max_nodes=max_nodes)
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    labels = list(node.labels)
                    label = labels[0] if labels else "Node"
                    
                    if label == "Document":
                        net.add_node(node.element_id, label=node.get("file_name", "Document"), color="#3b82f6", shape="box")
                    elif label == "Chunk":
                         net.add_node(node.element_id, label=f"Chunk {node.get('id', '')}", color="#10b981", shape="dot")
                    else:
                        net.add_node(node.element_id, label=node.get("name", "Unknown"), color="#ef4444", shape="star")
                        
                for rel in path.relationships:
                    net.add_edge(rel.start_node.element_id, rel.end_node.element_id, title=rel.type)

        return net.generate_html()

    def visualize_document_graph(self, document_id: Optional[str] = None, max_nodes: int = 100) -> str:
        """Generates an HTML visualization of a document hierarchy."""
        net = self._create_base_network()
        
        with self.neo4j.driver.session() as session:
            if document_id:
                query = """
                MATCH path = (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                RETURN path
                LIMIT $max_nodes
                """
                result = session.run(query, document_id=document_id, max_nodes=max_nodes)
            else:
                 query = """
                 MATCH path = (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                 RETURN path
                 LIMIT $max_nodes
                 """
                 result = session.run(query, max_nodes=max_nodes)

            for record in result:
                path = record["path"]
                for node in path.nodes:
                    labels = list(node.labels)
                    label = labels[0] if labels else "Node"
                    
                    if label == "Document":
                        net.add_node(node.element_id, label=node.get("file_name", "Document"), color="#3b82f6", shape="box")
                    elif label == "Chunk":
                         net.add_node(node.element_id, label=f"Chunk {node.get('id', '')}", color="#10b981", shape="dot")
                    else:
                        net.add_node(node.element_id, label=node.get("name", "Unknown"), color="#ef4444", shape="star")
                        
                for rel in path.relationships:
                    net.add_edge(rel.start_node.element_id, rel.end_node.element_id, title=rel.type)
            
        return net.generate_html()

    def render_graph(self, html_string: str):
        """Helper to render PyVis HTML in Streamlit environments."""
        import streamlit.components.v1 as components
        components.html(html_string, height=600)
