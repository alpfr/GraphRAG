# OpenSearch-Docling-GraphRAG

An advanced next-generation Retrieval-Augmented Generation (RAG) platform integrating document parsing, vector search, local LLM inference, and knowledge graph visualization.

## Architecture

This project combines four powerful technologies to create a comprehensive, fully local RAG pipeline:

1.  **Docling**: For intelligent document parsing. It handles complex formats like PDFs, HTML, MD, and all modern/legacy Microsoft Office formats (DOCX/DOC, PPTX/PPT, XLSX/XLS), extracting hierarchical structures and rich text chunks.
2.  **OpenSearch**: The primary vector database. It stores the document chunks and their high-dimensional embeddings (768-D), enabling ultra-fast, sub-millisecond semantic similarity search using the `hnsw` algorithm.
3.  **Neo4j**: The knowledge graph database. It maps the structural relationships between documents, chunks, and extracted semantic entities, allowing for multi-hop reasoning and visual relationship exploration using a native CDN-backed `PyVis` Network edge renderer.
4.  **Ollama**: The local and cloud-native LLM engine. Rebuilt with a natively integrated LangGraph autonomous agent via the `langchain-ollama` SDK. It provides the embedding model (`granite-embedding:278m`) for vectorizing text and the generative model (`ibm/granite4:latest`) for routing tools and synthesizing answers based on retrieved context.

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

## AWS EKS Deployment

This project includes fully configured Kubernetes manifests in the `k8s/` directory to deploy the platform to AWS Elastic Kubernetes Service (EKS).

To optimize AWS hosting costs, the EKS configuration incorporates an in-cluster **CPU-bound** Ollama Deployment using an automated `postStart` lifecycle hook to securely pull Granite models natively inside the VPC. The Streamlit frontend is powered by an automated multi-arch compiled AMD64 Docker container pushed to Amazon ECR to avoid `exec format error` on standard EC2 worker nodes.

### EKS Setup Instructions

1.  **Build and Push the Docker Image**:
    ```bash
    # Replace the registry URIs with your AWS account ID and region
    docker build -t <aws-account-id>.dkr.ecr.<region>.amazonaws.com/graphrag:latest .
    docker push <aws-account-id>.dkr.ecr.<region>.amazonaws.com/graphrag:latest
    ```
2.  **Configure Environment Variables**:
    Edit the `k8s/secrets.yaml` and input your cloud API keys. **Do not commit these to source control**.
3.  **Apply Manifests**:
    Ensure your EKS cluster has the EBS CSI Driver installed to support PersistentVolumeClaims, then apply:
    ```bash
    kubectl apply -f k8s/configmap.yaml -f k8s/secrets.yaml
    kubectl apply -f k8s/opensearch-statefulset.yaml -f k8s/neo4j-statefulset.yaml
    kubectl apply -f k8s/graphrag-deployment.yaml -f k8s/ingress.yaml
    ```
4.  **Access the Application**:
    Wait for the AWS Application Load Balancer to provision, then navigate to the Ingress address (`kubectl get ingress`).

## GCP GKE Deployment

We also provide native Google Kubernetes Engine (GKE) manifests located in the `k8s-gke/` directory. These manifests are tailored for GCP infrastructure, utilizing `standard-rwo` persistent disks and GCE Ingress Controllers.

### GKE Setup Instructions

1.  **Build and Push to Artifact Registry**:
    ```bash
    # Replace the registry URIs with your GCP project details
    docker build -t <REGION>-docker.pkg.dev/<YOUR_GCP_PROJECT_ID>/<REPOSITORY_NAME>/graphrag:latest .
    docker push <REGION>-docker.pkg.dev/<YOUR_GCP_PROJECT_ID>/<REPOSITORY_NAME>/graphrag:latest
    ```
2.  **Configure Environment Variables**:
    Edit `k8s-gke/secrets.yaml` and input your cloud API keys. **Do not commit these to source control**.
3.  **Apply GKE Manifests**:
    Ensure your GKE cluster has the Compute Engine Persistent Disk CSI Driver enabled, then apply:
    ```bash
    kubectl apply -f k8s-gke/configmap.yaml -f k8s-gke/secrets.yaml
    kubectl apply -f k8s-gke/opensearch-statefulset.yaml -f k8s-gke/neo4j-statefulset.yaml
    kubectl apply -f k8s-gke/graphrag-deployment.yaml -f k8s-gke/ingress.yaml
    ```
4.  **Access the Application**:
    Wait for the Google Cloud Load Balancer (GCLB) to provision, then navigate to the Ingress IP address (`kubectl get ingress`).

## Azure AKS Deployment

We also provide native Azure Kubernetes Service (AKS) manifests located in the `k8s-aks/` directory. These manifests are tailored for Azure infrastructure, utilizing `managed-premium` Azure Disks and the Azure Application Gateway Ingress Controller (AGIC).

### AKS Setup Instructions

1.  **Build and Push to Azure Container Registry (ACR)**:
    ```bash
    # Replace <YOUR_ACR_NAME> with your Azure Container Registry name
    docker build -t <YOUR_ACR_NAME>.azurecr.io/graphrag:latest .
    az acr login --name <YOUR_ACR_NAME>
    docker push <YOUR_ACR_NAME>.azurecr.io/graphrag:latest
    ```
2.  **Configure Environment Variables**:
    Edit `k8s-aks/secrets.yaml` and input your cloud API keys. **Do not commit these to source control**.
3.  **Apply AKS Manifests**:
    Ensure your AKS cluster has the Application Gateway Ingress Controller (AGIC) enabled, then apply:
    ```bash
    kubectl apply -f k8s-aks/configmap.yaml -f k8s-aks/secrets.yaml
    kubectl apply -f k8s-aks/opensearch-statefulset.yaml -f k8s-aks/neo4j-statefulset.yaml
    kubectl apply -f k8s-aks/graphrag-deployment.yaml -f k8s-aks/ingress.yaml
    ```
4.  **Access the Application**:
    Wait for the Azure Application Gateway to provision the public IP, then manually navigate to the App Gateway address (`kubectl get ingress`).

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
