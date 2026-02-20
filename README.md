# OpenSearch-Docling-GraphRAG

An advanced next-generation Retrieval-Augmented Generation (RAG) platform integrating document parsing, vector search, local LLM inference, and knowledge graph visualization.

## Architecture

This project combines four powerful technologies to create a comprehensive, fully local RAG pipeline:

1.  **Docling**: For intelligent document parsing. It handles complex formats like PDFs, DOCXs, and PPTXs, extracting hierarchical structures and rich text chunks.
2.  **OpenSearch**: The primary vector database. It stores the document chunks and their high-dimensional embeddings (768-D), enabling ultra-fast, sub-millisecond semantic similarity search using the `hnsw` algorithm.
3.  **Neo4j**: The knowledge graph database. It maps the structural relationships between documents, chunks, and extracted semantic entities, allowing for multi-hop reasoning and visual relationship exploration.
4.  **Ollama**: The local LLM engine. It provides the embedding model (`granite-embedding:278m`) for vectorizing text and the generative model (`ibm/granite4:latest`) for synthesizing answers based on retrieved context.

## Prerequisites

-   Docker and Docker Compose
-   Python 3.10+
-   Ollama installed locally

## Setup Instructions

1.  **Clone the Repository** (or navigate to the project directory).

2.  **Ensure Ollama Models are Pulled**:
    ```bash
    ollama pull ibm/granite4:latest
    ollama pull granite-embedding:278m
    ```

3.  **Start the Infrastructure**:
    Use the provided script to spin up the OpenSearch and Neo4j Docker containers, and to initialize the Python environment.
    ```bash
    ./start.sh
    ```
    *Note: The first time OpenSearch starts, it initializes its security plugin. The default strong password used is `Gr@phR@g!2026_OpenSearch!`.*

4.  **Run the Streamlit Application**:
    Once the environment is built and active (`source .venv/bin/activate`), start the frontend interface:
    ```bash
    streamlit run app.py
    ```

## Usage Guide

1.  **Initialize System**: Open the Streamlit UI (`http://localhost:8501`) and click **Initialize System** in the sidebar. This connects to OpenSearch, Neo4j, and Ollama, and prepares the necessary indices.
2.  **Upload & Process**: Use the **Upload** tab to ingest single documents, or place files in the `input/` directory and use the **Batch Process** tab to ingest them all at once. The system will chunk the text, generate embeddings, push vectors to OpenSearch, and construct the knowledge graph in Neo4j.
3.  **Semantic Search**: Navigate to the **Search** tab. Ask a natural language question. The system will embed your query, retrieve the top K similar chunks from OpenSearch, and use the local LLM to generate a cited answer.
4.  **Graph Explorer**: Go to the **Graph Explorer** tab to visually interact with your data. You can explore the connections of a specific entity, view the structural layout of a document, or visualize the entire interconnected knowledge map.

## Project Structure

-   `app.py`: The main Streamlit frontend application.
-   `src/processors.py`: Contains the `DoclingProcessor` for document parsing.
-   `src/rag/opensearch_client.py`: Handles vector indexing and semantic retrieval.
-   `src/rag/ollama_client.py`: Manages local LLM interactions for embeddings and generation.
-   `src/graphrag/neo4j_client.py`: Core logic for interacting with the Neo4j graph database.
-   `src/graphrag/builder.py`: Orchestrates the creation of nodes and relationships from text.
-   `src/graphrag/visualizer.py`: Generates the interactive `pyvis` HTML visualizations.
-   `docker-compose.yml`: Infrastructure definition for OpenSearch and Neo4j.
-   `config/settings.py`: Centralized environment and configuration management.
