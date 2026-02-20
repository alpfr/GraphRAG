import requests
import json
import logging
from typing import List, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for generating local embeddings and executing local LLM prompts."""

    def __init__(self):
        self.host = settings.ollama_host
        self.model = settings.ollama_model
        self.embedding_model = settings.ollama_embedding_model
        logger.info("Ollama client initialized")

    def generate_embedding(self, text: str) -> List[float]:
        """Generates embeddings using the specified embedding model."""
        url = f"{self.host}/api/embeddings"
        payload = {
            "model": self.embedding_model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Processes and returns a batch of text embeddings."""
        return [self.generate_embedding(text) for text in texts]

    def generate_rag_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Excecutes a generation operation with explicit contextual augmentation."""
        
        # Build prompt context
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"[Document Chunk {i+1} from {doc['file_name']}]\n{doc['text']}")
            sources.append({
                "file_name": doc['file_name'],
                "chunk_id": doc['chunk_id'],
                "score": doc['score']
            })
            
        context = "\n\n".join(context_parts)
        
        system_prompt = (
            "You are a helpful and extremely precise AI Assistant. Use the provided context "
            "to answer the user's query. If you do not know the answer based explicitly on "
            "the context, state that you don't know without making anything up."
        )
        
        user_prompt = f"Context Information:\n{context}\n\nQuery: {query}\nAnswer:"
        
        payload = {
            "model": self.model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "temperature": 0.1
        }
        
        url = f"{self.host}/api/generate"
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            return {
                "answer": data["response"],
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error in RAG generation: {str(e)}")
            raise
