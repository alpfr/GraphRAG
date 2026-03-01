import streamlit as st
from streamlit_option_menu import option_menu
import os
from pathlib import Path
from datetime import datetime
import json
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/app_{time}.log", rotation="1 day", retention="7 days", level="DEBUG")

# Import application modules
from src.processors import DoclingProcessor
from src.rag.opensearch_client import OpenSearchClient
from src.rag.ollama_client import OllamaClient
from src.graphrag.neo4j_client import Neo4jClient
from src.graphrag.builder import GraphBuilder
from src.graphrag.visualizer import GraphVisualizer
from src.agents.graphrag_agent import GraphragAgent
from src.llm.factory import LLMFactory
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="OPSSIGHT OpenSearch",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'opensearch_client' not in st.session_state:
    st.session_state.opensearch_client = None
if 'neo4j_client' not in st.session_state:
    st.session_state.neo4j_client = None
if 'graph_builder' not in st.session_state:
    st.session_state.graph_builder = None
if 'graph_visualizer' not in st.session_state:
    st.session_state.graph_visualizer = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False


import traceback
def initialize_clients():
    """Initialize all clients."""
    try:
        with st.spinner("Initializing clients..."):
            if not st.session_state.initialized:
                st.session_state.processor = DoclingProcessor()
                st.session_state.opensearch_client = OpenSearchClient()
                st.session_state.neo4j_client = Neo4jClient()
                st.session_state.embedding_model = LLMFactory.get_embeddings(settings.llm_provider)
                st.session_state.graph_builder = GraphBuilder(st.session_state.neo4j_client)
                st.session_state.graph_visualizer = GraphVisualizer(st.session_state.neo4j_client)
                st.session_state.agent = GraphragAgent(st.session_state.opensearch_client, st.session_state.neo4j_client)
                st.session_state.initialized = True
                st.success("✅ All clients initialized successfully!")
                logger.info("All clients initialized")
    except Exception as e:
        error_msg = repr(e)
        logger.error(f"Initialization error structure: {traceback.format_exc()}")
        if "Connection refused" in error_msg and settings.llm_provider == "ollama":
            st.error("❌ Error initializing clients: Cannot connect to local Ollama daemon (localhost:11434). Please select a Cloud provider (OpenAI/Anthropic/Gemini) since this is a cloud deployment.")
        else:
            st.error(f"❌ Error initializing clients: {error_msg}")



def process_single_file(uploaded_file):
    """Process a single uploaded file."""
    try:
        # Save uploaded file temporarily
        Path(settings.input_dir).mkdir(parents=True, exist_ok=True)
        temp_path = Path(settings.input_dir) / uploaded_file.name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner(f"Processing {uploaded_file.name}..."):
            doc_data = st.session_state.processor.process_document(str(temp_path))
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in doc_data['chunks']]
            embeddings = st.session_state.embedding_model.embed_documents(texts)
            
            # Index in OpenSearch
            document_id = f"{Path(uploaded_file.name).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.opensearch_client.index_document(
                document_id=document_id,
                file_name=uploaded_file.name,
                file_path=str(temp_path),
                chunks=doc_data['chunks'],
                embeddings=embeddings,
                metadata=doc_data['metadata']
            )
            
            # Build knowledge graph
            st.session_state.graph_builder.build_document_graph(
                document_id=document_id,
                file_name=uploaded_file.name,
                file_path=str(temp_path),
                chunks=doc_data['chunks'],
                metadata=doc_data['metadata']
            )
            
            # Save output
            output_file = st.session_state.processor.save_output(doc_data, settings.output_dir)
            
            return {
                'success': True,
                'document_id': document_id,
                'output_file': output_file,
                'chunks': len(doc_data['chunks'])
            }
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {'success': False, 'error': str(e)}


def process_batch_files():
    """Process all files in the input directory."""
    input_dir = Path(settings.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob('*'))
    files = [f for f in files if f.is_file() and not f.name.startswith('.')]
    
    if not files:
        st.warning("No files found in input directory")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = []
    
    for i, file_path in enumerate(files):
        status_text.text(f"Processing {file_path.name} ({i+1}/{len(files)})")
        
        try:
            # Process document
            doc_data = st.session_state.processor.process_document(str(file_path))
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in doc_data['chunks']]
            embeddings = st.session_state.embedding_model.embed_documents(texts)
            
            # Index in OpenSearch
            document_id = f"{file_path.stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.opensearch_client.index_document(
                document_id=document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                chunks=doc_data['chunks'],
                embeddings=embeddings,
                metadata=doc_data['metadata']
            )
            
            # Build knowledge graph
            st.session_state.graph_builder.build_document_graph(
                document_id=document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                chunks=doc_data['chunks'],
                metadata=doc_data['metadata']
            )
            
            # Save output
            output_file = st.session_state.processor.save_output(doc_data, settings.output_dir)
            
            results.append({
                'file': file_path.name,
                'status': 'success',
                'document_id': document_id,
                'chunks': len(doc_data['chunks'])
            })
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            results.append({
                'file': file_path.name,
                'status': 'failed',
                'error': str(e)
            })
        
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text("Batch processing complete!")
    return results


def main():
    """Main application."""
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Document RAG System")
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Upload", "Batch Process", "Search", "Graph Explorer", "Settings"],
            icons=["house", "cloud-upload", "files", "search", "diagram-3", "gear"],
            default_index=0,
        )
        
        st.divider()
        

        st.divider()
        
        # Display Mode Toggle
        st.subheader("📱 Display Mode")
        display_mode = st.radio(
            "View Mode",
            options=["Desktop", "Mobile"],
            horizontal=True
        )
        
        if display_mode == "Mobile":
            # Inject CSS to simulate a mobile screen width
            st.markdown("""
                <style>
                /* Constrain the main block container to mobile width */
                .block-container {
                    max-width: 480px !important;
                    margin: 0 auto !important;
                    padding-top: 2rem !important;
                }
                /* Hide sidebar completely on mobile simulation if needed, but Streamlit has a built-in collapser */
                </style>
            """, unsafe_allow_html=True)
        else:
            # Force wide layout for desktop
            st.markdown("""
                <style>
                .block-container {
                    max-width: 100% !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
        st.divider()
        # System status
        st.subheader("System Status")
        if st.session_state.initialized:
            st.success("✅ System Ready")
            
            # Show statistics
            try:
                doc_count = st.session_state.opensearch_client.get_document_count()
                st.metric("Indexed Chunks", doc_count)
                
                graph_stats = st.session_state.graph_builder.get_graph_summary()
                st.metric("Documents in Graph", graph_stats.get('documents', 0))
                st.metric("Entities", graph_stats.get('entities', 0))
            except:
                pass
        else:
            st.warning("⚠️ System Not Initialized")
            if st.button("Initialize System"):
                initialize_clients()
    
    # Main content
    if selected == "Home":
        st.title("🏠 Welcome to OPSSIGHT OpenSearch")
        st.markdown("""
        ### A Comprehensive Document Processing and RAG System
        
        This application combines:
        - **Docling**: Advanced document processing
        - **OpenSearch**: Vector search and retrieval
        - **Neo4j**: Knowledge graph construction
        - **Ollama**: Local LLM for embeddings and generation
        
        #### Features:
        - 📄 Process various document formats (PDF, DOCX, PPTX, etc.)
        - 🔍 Semantic search with vector embeddings
        - 🕸️ Knowledge graph construction and exploration
        - 💬 RAG-based question answering
        - 📊 Batch processing capabilities
        """)
        
        if not st.session_state.initialized:
            st.info("👈 Please initialize the system from the sidebar to get started")
    
    elif selected == "Upload":
        st.title("📤 Upload Documents")
        
        if not st.session_state.initialized:
            st.warning("Please initialize the system first")
            return
        
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'xlsx', 'xls', 'txt', 'md', 'html']
        )
        
        if uploaded_file:
            st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if st.button("Process Document"):
                result = process_single_file(uploaded_file)
                
                if result['success']:
                    st.success(f"✅ Document processed successfully!")
                    st.json({
                        'Document ID': result['document_id'],
                        'Chunks Created': result['chunks'],
                        'Output File': result['output_file']
                    })
                else:
                    st.error(f"❌ Error: {result['error']}")
    
    elif selected == "Batch Process":
        st.title("📁 Batch Process Documents")
        
        if not st.session_state.initialized:
            st.warning("Please initialize the system first")
            return
        
        st.info(f"Input directory: {settings.input_dir}")
        
        input_dir = Path(settings.input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        files = list(input_dir.glob('*'))
        files = [f for f in files if f.is_file() and not f.name.startswith('.')]
        
        st.write(f"Found {len(files)} files:")
        for f in files:
            st.text(f"  • {f.name}")
        
        if st.button("Process All Files"):
            if files:
                results = process_batch_files()
                
                # Display results
                if results:
                    st.subheader("Processing Results")
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    st.metric("Successfully Processed", f"{success_count}/{len(results)}")
                    
                    # Show details
                    for result in results:
                        if result['status'] == 'success':
                            st.success(f"✅ {result['file']} - {result['chunks']} chunks")
                        else:
                            st.error(f"❌ {result['file']} - {result['error']}")
            else:
                st.warning("No files to process")
    
    elif selected == "Search":
        st.title("🤖 Agentic GraphRAG Search")
        
        if not st.session_state.initialized:
            st.warning("Please initialize the system first")
            return
        
        st.markdown(f"**Current LLM Backend:** `{settings.llm_provider}`")
        
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            st.chat_message("user").write(query)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("Agent is analyzing request..."):
                    try:
                        for chunk in st.session_state.agent.stream_steps(query):
                            full_response += chunk + "\n\n"
                            message_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f"Error during agent execution: {str(e)}")
    
    elif selected == "Graph Explorer":
        st.title("🕸️ Knowledge Graph Explorer")
        
        if not st.session_state.initialized:
            st.warning("Please initialize the system first")
            return
        
        viz_tab, search_tab, stats_tab = st.tabs(["📊 Visualize", "🔍 Search", "📈 Statistics"])
        
        with viz_tab:
            st.subheader("Interactive Graph Visualization")
            
            viz_type = st.radio(
                "Select visualization type:",
                ["Entity Graph", "Document Structure", "Full Graph"],
                horizontal=True
            )
            
            if viz_type == "Entity Graph":
                st.markdown("**Visualize an entity and its connections**")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    entity_name = st.text_input(
                        "Enter entity name:",
                        placeholder="e.g., Bob, Python, AI",
                        key="entity_viz"
                    )
                with col2:
                    max_depth = st.number_input("Max depth:", min_value=1, max_value=5, value=2)
                
                if st.button("🎨 Visualize Entity", type="primary"):
                    if entity_name:
                        try:
                            with st.spinner(f"Generating graph for '{entity_name}'..."):
                                html = st.session_state.graph_visualizer.visualize_entity_graph(
                                    entity_name=entity_name,
                                    max_depth=max_depth,
                                    max_nodes=50
                                )
                                st.session_state.graph_visualizer.render_graph(html)
                                st.success(f"✅ Graph generated for '{entity_name}'")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.warning("⚠️ Please enter an entity name")
            
            elif viz_type == "Document Structure":
                st.markdown("**Visualize document structure with chunks and entities**")
                try:
                    with st.session_state.neo4j_client.driver.session() as session:
                        result = session.run("MATCH (d:Document) RETURN d.id as id, d.file_name as name")
                        documents = [(r['id'], r['name']) for r in result]
                    
                    if documents:
                        doc_options = ["All Documents"] + [f"{name} ({id[:8]}...)" for id, name in documents]
                        selected_doc = st.selectbox("Select document:", doc_options)
                        
                        if st.button("🎨 Visualize Document", type="primary"):
                            try:
                                with st.spinner("Generating document graph..."):
                                    if selected_doc == "All Documents":
                                        html = st.session_state.graph_visualizer.visualize_document_graph(
                                            document_id=None,
                                            max_nodes=100
                                        )
                                    else:
                                        doc_id = [id for id, name in documents if name in selected_doc][0]
                                        html = st.session_state.graph_visualizer.visualize_document_graph(
                                            document_id=doc_id,
                                            max_nodes=100
                                        )
                                    st.session_state.graph_visualizer.render_graph(html)
                                    st.success("✅ Document graph generated")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    else:
                        st.info("📄 No documents found. Upload and process documents first.")
                except Exception as e:
                    st.error(f"❌ Error fetching documents: {str(e)}")
            
            else:  # Full Graph
                st.markdown("**Visualize the entire knowledge graph**")
                max_nodes = st.slider("Maximum nodes to display:", 10, 200, 50)
                
                if st.button("🎨 Visualize Full Graph", type="primary"):
                    try:
                        with st.spinner("Generating full graph..."):
                            html = st.session_state.graph_visualizer.visualize_document_graph(
                                document_id=None,
                                max_nodes=max_nodes
                            )
                            st.session_state.graph_visualizer.render_graph(html)
                            st.success("✅ Full graph generated")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with search_tab:
            st.subheader("🔍 Search Entities")
            entity_name = st.text_input("Search for entity:", key="search_entity")
            
            if st.button("Search", key="search_btn"):
                if entity_name:
                    try:
                        with st.session_state.neo4j_client.driver.session() as session:
                            result = session.run("""
                                MATCH (e:Entity)
                                WHERE toLower(e.name) CONTAINS toLower($name)
                                RETURN e.name as name, id(e) as id
                                LIMIT 20
                            """, name=entity_name)
                            
                            entities = list(result)
                            if entities:
                                st.success(f"Found {len(entities)} entities:")
                                for entity in entities:
                                    with st.expander(f"🔴 {entity['name']}"):
                                        conn_result = session.run("""
                                            MATCH (e:Entity)-[r]-(n)
                                            WHERE id(e) = $id
                                            RETURN type(r) as rel_type, labels(n) as labels, count(*) as count
                                        """, id=entity['id'])
                                        
                                        connections = list(conn_result)
                                        if connections:
                                            st.markdown("**Connections:**")
                                            for conn in connections:
                                                st.text(f"  • {conn['rel_type']} → {conn['labels'][0]}: {conn['count']}")
                            else:
                                st.info(f"No entities found matching '{entity_name}'")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.warning("Please enter an entity name")
        
        with stats_tab:
            st.subheader("📈 Graph Statistics")
            try:
                stats = st.session_state.neo4j_client.get_statistics()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Documents", stats.get('documents', 0))
                with col2:
                    st.metric("Chunks", stats.get('chunks', 0))
                with col3:
                    st.metric("Entities", stats.get('entities', 0))
                with col4:
                    st.metric("Relationships", stats.get('relationships', 0))
            except Exception as e:
                st.error(f"❌ Error fetching statistics: {str(e)}")
    
    elif selected == "Settings":
        st.title("⚙️ Settings")
        st.json({
            'Ollama Model': settings.ollama_model,
            'Embedding Model': settings.ollama_embedding_model,
            'OpenSearch Host': f"{settings.opensearch_host}:{settings.opensearch_port}",
            'Neo4j URI': settings.neo4j_uri,
            'Input Directory': settings.input_dir,
            'Output Directory': settings.output_dir,
            'Chunk Size': settings.chunk_size,
            'Chunk Overlap': settings.chunk_overlap
        })

if __name__ == "__main__":
    main()
