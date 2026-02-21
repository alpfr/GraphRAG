import logging
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating Chat Models and Embeddings across different providers."""
    
    @staticmethod
    def get_chat_model(provider: Optional[str] = None) -> BaseChatModel:
        provider = provider or settings.llm_provider
        logger.info(f"Initializing Chat Model for provider: {provider}")
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            return ChatOpenAI(
                model="gpt-4o", 
                temperature=0.1, 
                api_key=settings.openai_api_key
            )
            
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            return ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022", 
                temperature=0.1, 
                api_key=settings.anthropic_api_key
            )
            
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", 
                temperature=0.1, 
                google_api_key=settings.google_api_key
            )
            
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model, 
                temperature=0.1
            )
            
        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama.")
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model, 
                temperature=0.1
            )

    @staticmethod
    def get_embeddings(provider: Optional[str] = None) -> Embeddings:
        provider = provider or settings.llm_provider
        logger.info(f"Initializing Embedding Model for provider: {provider}")
        
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            return OpenAIEmbeddings(
                model="text-embedding-3-small", 
                api_key=settings.openai_api_key
            )
            
        elif provider == "anthropic":
            # Anthropic does not have a native embedding offering in LangChain,
            # falling back to OpenAI instead of localhost Ollama.
            logger.warning("Anthropic selected but no native embeddings exist. Falling back to OpenAI embeddings.")
            from langchain_openai import OpenAIEmbeddings
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set. Anthropic requires OpenAI API Key for embeddings fallback.")
            return OpenAIEmbeddings(
                model="text-embedding-3-small", 
                api_key=settings.openai_api_key
            )
            
        elif provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004", 
                google_api_key=settings.google_api_key
            )
            
        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=settings.ollama_host,
                model=settings.ollama_embedding_model
            )
            
        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama.")
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=settings.ollama_host,
                model=settings.ollama_embedding_model
            )
