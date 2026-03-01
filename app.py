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
from src.llm.model_registry import (
    PROVIDER_REGISTRY, get_provider, get_chat_models,
    get_embedding_models, get_embedding_dimension, estimate_cost,
)
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="OPSSIGHT OpenSearch",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
_defaults = {
    'processor': None,
    'opensearch_client': None,
    'neo4j_client': None,
    'graph_builder': None,
    'graph_visualizer': None,
    'agent': None,
    'embedding_model': None,
    'initialized': False,
    # LLM selector state
    'current_provider': settings.llm_provider,
    'current_model': None,
    'current_temperature': settings.llm_temperature,
    'current_max_tokens': settings.llm_max_tokens,
    'current_top_p': settings.llm_top_p,
    'provider_status': {},
    'embedding_provider': settings.llm_provider,
    'session_token_usage': {"input": 0, "output": 0, "cost_usd": 0.0},
    'ollama_models': [],
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def check_all_provider_status():
    """Check connectivity status of all providers."""
    for provider_id in PROVIDER_REGISTRY:
        st.session_state.provider_status[provider_id] = LLMFactory.check_provider_status(provider_id)
    # Also refresh Ollama models
    st.session_state.ollama_models = LLMFactory.list_ollama_models()


def switch_provider(provider, model_name, temperature, max_tokens):
    """Switch the active LLM provider and reinitialize the agent."""
    if not st.session_state.initialized:
        st.warning("System not initialized yet.")
        return

    try:
        # Check embedding dimension compatibility
        new_embed_provider = provider
        new_embed_models = get_embedding_models(new_embed_provider)
        if new_embed_models:
            new_dim = new_embed_models[0].embedding_dimensions or 768
        elif provider == "anthropic":
            # Anthropic falls back to OpenAI embeddings
            new_dim = 1536
        else:
            new_dim = 768

        if st.session_state.opensearch_client:
            current_dim = st.session_state.opensearch_client.get_index_dimension()
            if current_dim and current_dim != new_dim:
                st.warning(
                    f"**Embedding dimension change:** Current index uses {current_dim}d, "
                    f"new provider uses {new_dim}d. Existing indexed documents will NOT be "
                    f"compatible for search. You may need to re-index."
                )

        # Reinitialize the agent
        st.session_state.agent.reinitialize(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Update embedding model
        st.session_state.embedding_model = LLMFactory.get_embeddings(provider)

        # Reinitialize GraphBuilder with new LLM
        st.session_state.graph_builder = GraphBuilder(
            st.session_state.neo4j_client,
            provider=provider,
            model_name=model_name,
        )

        # Update session state
        st.session_state.current_provider = provider
        st.session_state.current_model = model_name
        st.session_state.current_temperature = temperature
        st.session_state.current_max_tokens = max_tokens
        st.session_state.embedding_provider = provider

        st.success(f"Switched to **{PROVIDER_REGISTRY[provider].display_name}** ({model_name})")

    except Exception as e:
        st.error(f"Failed to switch provider: {str(e)}")
        logger.error(f"Provider switch failed: {e}")


import traceback
def initialize_clients():
    """Initialize all clients with the selected LLM provider."""
    try:
        with st.spinner("Initializing clients..."):
            if not st.session_state.initialized:
                provider = st.session_state.current_provider
                model_name = st.session_state.current_model

                st.session_state.processor = DoclingProcessor()
                st.session_state.opensearch_client = OpenSearchClient()
                st.session_state.neo4j_client = Neo4jClient()
                st.session_state.embedding_model = LLMFactory.get_embeddings(provider)
                st.session_state.graph_builder = GraphBuilder(
                    st.session_state.neo4j_client,
                    provider=provider,
                    model_name=model_name,
                )
                st.session_state.graph_visualizer = GraphVisualizer(st.session_state.neo4j_client)
                st.session_state.agent = GraphragAgent(
                    st.session_state.opensearch_client,
                    st.session_state.neo4j_client,
                    provider=provider,
                    model_name=model_name,
                    temperature=st.session_state.current_temperature,
                    max_tokens=st.session_state.current_max_tokens,
                )
                st.session_state.initialized = True

                # Resolve the model name that was actually used
                if not st.session_state.current_model:
                    registry = get_provider(provider)
                    if registry:
                        st.session_state.current_model = registry.default_chat_model

                check_all_provider_status()
                st.success("All clients initialized successfully!")
                logger.info(f"All clients initialized with provider={provider}")
    except Exception as e:
        error_msg = repr(e)
        logger.error(f"Initialization error: {traceback.format_exc()}")
        if "Connection refused" in error_msg and settings.llm_provider == "ollama":
            st.error(
                "Cannot connect to local Ollama daemon. "
                "Select a cloud provider (OpenAI/Anthropic/Gemini) from the sidebar."
            )
        else:
            st.error(f"Error initializing clients: {error_msg}")


def process_single_file(uploaded_file):
    """Process a single uploaded file."""
    try:
        Path(settings.input_dir).mkdir(parents=True, exist_ok=True)
        temp_path = Path(settings.input_dir) / uploaded_file.name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Processing {uploaded_file.name}..."):
            doc_data = st.session_state.processor.process_document(str(temp_path))

            texts = [chunk['text'] for chunk in doc_data['chunks']]
            embeddings = st.session_state.embedding_model.embed_documents(texts)

            document_id = f"{Path(uploaded_file.name).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.opensearch_client.index_document(
                document_id=document_id,
                file_name=uploaded_file.name,
                file_path=str(temp_path),
                chunks=doc_data['chunks'],
                embeddings=embeddings,
                metadata=doc_data['metadata']
            )

            st.session_state.graph_builder.build_document_graph(
                document_id=document_id,
                file_name=uploaded_file.name,
                file_path=str(temp_path),
                chunks=doc_data['chunks'],
                metadata=doc_data['metadata']
            )

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
            doc_data = st.session_state.processor.process_document(str(file_path))

            texts = [chunk['text'] for chunk in doc_data['chunks']]
            embeddings = st.session_state.embedding_model.embed_documents(texts)

            document_id = f"{file_path.stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.opensearch_client.index_document(
                document_id=document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                chunks=doc_data['chunks'],
                embeddings=embeddings,
                metadata=doc_data['metadata']
            )

            st.session_state.graph_builder.build_document_graph(
                document_id=document_id,
                file_name=file_path.name,
                file_path=str(file_path),
                chunks=doc_data['chunks'],
                metadata=doc_data['metadata']
            )

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


# ──────────────────────────────────────────────
# Sidebar: LLM Selector
# ──────────────────────────────────────────────

def render_llm_selector():
    """Render the LLM provider/model selector in the sidebar."""
    st.subheader("🤖 LLM Provider")

    # Provider dropdown
    provider_options = list(PROVIDER_REGISTRY.keys())
    provider_labels = [PROVIDER_REGISTRY[p].display_name for p in provider_options]
    current_idx = provider_options.index(st.session_state.current_provider) if st.session_state.current_provider in provider_options else 0

    selected_provider = st.selectbox(
        "Provider",
        provider_options,
        index=current_idx,
        format_func=lambda p: PROVIDER_REGISTRY[p].display_name,
        key="llm_provider_select",
    )

    # Provider status indicator
    status = st.session_state.provider_status.get(selected_provider)
    if status:
        if status["available"]:
            latency = f" ({status['latency_ms']}ms)" if status.get("latency_ms") else ""
            st.markdown(f"🟢 **Connected**{latency}")
        else:
            st.markdown(f"🔴 **Unavailable:** {status.get('error', 'Unknown')}")
    else:
        st.markdown("⚪ Status unknown — initialize system to check")

    # Model dropdown
    registry = get_provider(selected_provider)
    chat_models = get_chat_models(selected_provider)

    if selected_provider == "ollama":
        # Show detected Ollama models + custom input
        ollama_models = st.session_state.ollama_models or []
        registered_ids = [m.model_id for m in chat_models]
        all_models = list(set(registered_ids + ollama_models))
        if not all_models:
            all_models = [registry.default_chat_model]
        all_models.append("Custom...")

        current_model = st.session_state.current_model or registry.default_chat_model
        model_idx = all_models.index(current_model) if current_model in all_models else 0

        selected_model = st.selectbox("Model", all_models, index=model_idx, key="llm_model_select")

        if selected_model == "Custom...":
            selected_model = st.text_input("Custom model name:", value=registry.default_chat_model, key="custom_ollama_model")
    else:
        model_ids = [m.model_id for m in chat_models]
        model_labels = {m.model_id: m.display_name for m in chat_models}
        current_model = st.session_state.current_model
        model_idx = model_ids.index(current_model) if current_model in model_ids else 0

        selected_model = st.selectbox(
            "Model",
            model_ids,
            index=model_idx,
            format_func=lambda m: model_labels.get(m, m),
            key="llm_model_select",
        )

    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        selected_temp = st.slider(
            "Temperature", 0.0, 2.0,
            value=st.session_state.current_temperature,
            step=0.05, key="llm_temp_slider",
        )
        selected_max_tokens = st.slider(
            "Max Tokens", 256, 16384,
            value=st.session_state.current_max_tokens,
            step=256, key="llm_max_tokens_slider",
        )

        # Cost estimate
        cost_in, cost_out = 0.0, 0.0
        for m in chat_models:
            if m.model_id == selected_model:
                cost_in = m.cost_per_1k_input or 0.0
                cost_out = m.cost_per_1k_output or 0.0
                break
        if cost_in > 0 or cost_out > 0:
            st.caption(f"💰 Est. cost: ${cost_in:.4f}/1K in, ${cost_out:.4f}/1K out")
        else:
            st.caption("💰 Free (local model)")

    # Check if anything changed
    changed = (
        selected_provider != st.session_state.current_provider
        or selected_model != st.session_state.current_model
        or selected_temp != st.session_state.current_temperature
        or selected_max_tokens != st.session_state.current_max_tokens
    )

    if changed and st.session_state.initialized:
        if st.button("🔄 Apply Changes", type="primary", key="apply_llm_btn"):
            switch_provider(selected_provider, selected_model, selected_temp, selected_max_tokens)
            st.rerun()
    elif changed and not st.session_state.initialized:
        # Just update session state for when initialization happens
        st.session_state.current_provider = selected_provider
        st.session_state.current_model = selected_model
        st.session_state.current_temperature = selected_temp
        st.session_state.current_max_tokens = selected_max_tokens

    # Refresh status button
    if st.button("🔄 Refresh Status", key="refresh_status_btn"):
        with st.spinner("Checking providers..."):
            check_all_provider_status()
        st.rerun()


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

def main():
    """Main application."""

    # Sidebar
    with st.sidebar:
        st.title("📚 OPSSIGHT OpenSearch")

        selected = option_menu(
            menu_title=None,
            options=["Home", "Upload", "Batch Process", "Search", "Graph Explorer", "Settings"],
            icons=["house", "cloud-upload", "files", "search", "diagram-3", "gear"],
            default_index=0,
        )

        st.divider()

        # LLM Provider Selector
        render_llm_selector()

        st.divider()

        # Display Mode Toggle
        st.subheader("📱 Display Mode")
        display_mode = st.radio(
            "View Mode",
            options=["Desktop", "Mobile"],
            horizontal=True
        )

        if display_mode == "Mobile":
            st.markdown("""
                <style>
                .block-container {
                    max-width: 480px !important;
                    margin: 0 auto !important;
                    padding-top: 2rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
        else:
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

    # ── Main Content ──

    if selected == "Home":
        st.title("🏠 Welcome to OPSSIGHT OpenSearch")
        st.markdown("""
        ### A Comprehensive Document Processing and RAG System

        This application combines:
        - **Docling**: Advanced document processing
        - **OpenSearch**: Vector search and retrieval
        - **Neo4j**: Knowledge graph construction
        - **Multi-LLM**: OpenAI, Anthropic, Gemini, or Ollama for embeddings and generation

        #### Features:
        - 📄 Process various document formats (PDF, DOCX, PPTX, etc.)
        - 🔍 Semantic search with vector embeddings
        - 🕸️ Knowledge graph construction and exploration
        - 🤖 Agentic GraphRAG with selectable LLM provider
        - 📊 Batch processing capabilities
        - 🔄 Dynamic LLM switching from the sidebar
        """)

        if not st.session_state.initialized:
            st.info("👈 Select your LLM provider and initialize the system from the sidebar to get started")

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
                    st.success("Document processed successfully!")
                    st.json({
                        'Document ID': result['document_id'],
                        'Chunks Created': result['chunks'],
                        'Output File': result['output_file']
                    })
                else:
                    st.error(f"Error: {result['error']}")

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
            st.text(f"  - {f.name}")

        if st.button("Process All Files"):
            if files:
                results = process_batch_files()

                if results:
                    st.subheader("Processing Results")
                    success_count = sum(1 for r in results if r['status'] == 'success')
                    st.metric("Successfully Processed", f"{success_count}/{len(results)}")

                    for result in results:
                        if result['status'] == 'success':
                            st.success(f"{result['file']} - {result['chunks']} chunks")
                        else:
                            st.error(f"{result['file']} - {result['error']}")
            else:
                st.warning("No files to process")

    elif selected == "Search":
        st.title("🤖 Agentic GraphRAG Search")

        if not st.session_state.initialized:
            st.warning("Please initialize the system first")
            return

        # Show active LLM info
        provider_name = PROVIDER_REGISTRY.get(st.session_state.current_provider, {})
        if hasattr(provider_name, 'display_name'):
            provider_name = provider_name.display_name
        else:
            provider_name = st.session_state.current_provider

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Provider:** `{provider_name}`")
        with col2:
            st.markdown(f"**Model:** `{st.session_state.current_model}`")
        with col3:
            st.markdown(f"**Temp:** `{st.session_state.current_temperature}`")

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

                # Approximate cost estimation
                approx_input_tokens = len(query) // 4 + 500  # query + system prompt
                approx_output_tokens = len(full_response) // 4
                cost = estimate_cost(
                    st.session_state.current_provider,
                    st.session_state.current_model or "",
                    approx_input_tokens,
                    approx_output_tokens,
                )
                if cost > 0:
                    st.caption(f"💰 Estimated cost: ${cost:.6f} (~{approx_input_tokens} in, ~{approx_output_tokens} out tokens)")

                # Accumulate session usage
                st.session_state.session_token_usage["input"] += approx_input_tokens
                st.session_state.session_token_usage["output"] += approx_output_tokens
                st.session_state.session_token_usage["cost_usd"] += cost

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
                                st.success(f"Graph generated for '{entity_name}'")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                    else:
                        st.warning("Please enter an entity name")

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
                                    st.success("Document graph generated")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.info("No documents found. Upload and process documents first.")
                except Exception as e:
                    st.error(f"Error fetching documents: {str(e)}")

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
                            st.success("Full graph generated")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

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
                                                st.text(f"  - {conn['rel_type']} -> {conn['labels'][0]}: {conn['count']}")
                            else:
                                st.info(f"No entities found matching '{entity_name}'")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
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
                st.error(f"Error fetching statistics: {str(e)}")

    elif selected == "Settings":
        st.title("⚙️ Settings")

        # Active LLM Configuration
        st.subheader("🤖 Active LLM Configuration")
        llm_config = {
            'Provider': PROVIDER_REGISTRY.get(st.session_state.current_provider, {}).display_name
                        if hasattr(PROVIDER_REGISTRY.get(st.session_state.current_provider, {}), 'display_name')
                        else st.session_state.current_provider,
            'Model': st.session_state.current_model or "Default",
            'Temperature': st.session_state.current_temperature,
            'Max Tokens': st.session_state.current_max_tokens,
        }
        st.json(llm_config)

        # Provider Status Table
        st.subheader("📡 Provider Status")
        if st.session_state.provider_status:
            for pid, status in st.session_state.provider_status.items():
                pname = PROVIDER_REGISTRY[pid].display_name
                if status["available"]:
                    latency = f" — {status['latency_ms']}ms" if status.get("latency_ms") else ""
                    st.markdown(f"🟢 **{pname}**: Connected{latency}")
                else:
                    st.markdown(f"🔴 **{pname}**: {status.get('error', 'Unavailable')}")
        else:
            st.info("Initialize the system to check provider status.")

        # Session Usage
        st.subheader("💰 Session Token Usage")
        usage = st.session_state.session_token_usage
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Tokens", f"~{usage['input']:,}")
        with col2:
            st.metric("Output Tokens", f"~{usage['output']:,}")
        with col3:
            st.metric("Est. Cost", f"${usage['cost_usd']:.6f}")

        # Infrastructure Settings
        st.subheader("🔧 Infrastructure")
        st.json({
            'Ollama Host': settings.ollama_host,
            'Ollama Model': settings.ollama_model,
            'Embedding Model': settings.ollama_embedding_model,
            'OpenSearch Host': f"{settings.opensearch_host}:{settings.opensearch_port}",
            'Neo4j URI': settings.neo4j_uri,
            'Input Directory': settings.input_dir,
            'Output Directory': settings.output_dir,
            'Chunk Size': settings.chunk_size,
            'Chunk Overlap': settings.chunk_overlap,
        })


if __name__ == "__main__":
    main()
