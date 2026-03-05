import logging
import time
import requests
from typing import Optional, List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from config.settings import settings
from src.llm.model_registry import PROVIDER_REGISTRY, get_provider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating Chat Models and Embeddings across different providers."""

    @staticmethod
    def get_chat_model(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """Create a chat model for the given provider and model configuration."""
        provider = provider or settings.llm_provider
        registry = get_provider(provider)

        # Resolve defaults from settings then registry
        if temperature is None:
            temperature = settings.llm_temperature
        if max_tokens is None:
            max_tokens = settings.llm_max_tokens

        logger.info(f"Initializing Chat Model: provider={provider}, model={model_name}, temp={temperature}")

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            model_name = model_name or settings.openai_model
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=settings.openai_api_key,
            )

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            model_name = model_name or settings.anthropic_model
            return ChatAnthropic(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=settings.anthropic_api_key,
            )

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            model_name = model_name or settings.gemini_model
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=settings.google_api_key,
            )

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            model_name = model_name or settings.ollama_model
            return ChatOllama(
                base_url=settings.ollama_host,
                model=model_name,
                temperature=temperature,
            )

        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama.")
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_host,
                model=settings.ollama_model,
                temperature=temperature,
            )

    @staticmethod
    def get_embeddings(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Embeddings:
        """Create an embedding model for the given provider."""
        provider = provider or settings.llm_provider
        logger.info(f"Initializing Embedding Model: provider={provider}, model={model_name}")

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            model_name = model_name or settings.openai_embedding_model
            return OpenAIEmbeddings(
                model=model_name,
                api_key=settings.openai_api_key,
            )

        elif provider == "anthropic":
            # Anthropic has no native embeddings — fall back to OpenAI
            logger.warning("Anthropic selected but no native embeddings. Falling back to OpenAI embeddings.")
            from langchain_openai import OpenAIEmbeddings
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set. Anthropic requires OpenAI for embeddings.")
            model_name = model_name or settings.openai_embedding_model
            return OpenAIEmbeddings(
                model=model_name,
                api_key=settings.openai_api_key,
            )

        elif provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            model_name = model_name or settings.gemini_embedding_model
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.google_api_key,
            )

        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            model_name = model_name or settings.ollama_embedding_model
            return OllamaEmbeddings(
                base_url=settings.ollama_host,
                model=model_name,
            )

        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama.")
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(
                base_url=settings.ollama_host,
                model=settings.ollama_embedding_model,
            )

    @staticmethod
    def check_provider_status(provider: str) -> Dict[str, Any]:
        """Check connectivity and availability of a provider. Returns {available, error, latency_ms}."""
        start = time.time()
        try:
            if provider == "openai":
                if not settings.openai_api_key:
                    return {"available": False, "error": "API key not configured", "latency_ms": None}
                resp = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    timeout=5,
                )
                latency = int((time.time() - start) * 1000)
                if resp.status_code == 200:
                    return {"available": True, "error": None, "latency_ms": latency}
                return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}

            elif provider == "anthropic":
                if not settings.anthropic_api_key:
                    return {"available": False, "error": "API key not configured", "latency_ms": None}
                resp = requests.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": settings.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=5,
                )
                latency = int((time.time() - start) * 1000)
                if resp.status_code in (200, 201):
                    return {"available": True, "error": None, "latency_ms": latency}
                return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}

            elif provider == "gemini":
                if not settings.google_api_key:
                    return {"available": False, "error": "API key not configured", "latency_ms": None}
                resp = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={settings.google_api_key}",
                    timeout=5,
                )
                latency = int((time.time() - start) * 1000)
                if resp.status_code == 200:
                    return {"available": True, "error": None, "latency_ms": latency}
                return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}

            elif provider == "ollama":
                resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
                latency = int((time.time() - start) * 1000)
                if resp.status_code == 200:
                    return {"available": True, "error": None, "latency_ms": latency}
                return {"available": False, "error": f"HTTP {resp.status_code}", "latency_ms": latency}

            else:
                return {"available": False, "error": f"Unknown provider: {provider}", "latency_ms": None}

        except requests.exceptions.ConnectionError:
            return {"available": False, "error": "Connection refused", "latency_ms": None}
        except requests.exceptions.Timeout:
            return {"available": False, "error": "Timeout", "latency_ms": None}
        except Exception as e:
            return {"available": False, "error": str(e), "latency_ms": None}

    @staticmethod
    def list_ollama_models() -> List[str]:
        """Query Ollama for available local models."""
        try:
            resp = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
        return []

    @staticmethod
    def get_chat_model_with_fallback(
        provider: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        fallback_chain: Optional[List[str]] = None,
    ) -> tuple:
        """Try to create a chat model, falling back through providers on failure.

        Returns (model, actual_provider, actual_model_name).
        """
        fallback_chain = fallback_chain or ["openai", "anthropic", "gemini", "ollama"]

        # Try primary provider first
        try:
            model = LLMFactory.get_chat_model(provider, model_name, temperature, max_tokens)
            return model, provider, model_name
        except Exception as e:
            logger.warning(f"Primary provider '{provider}' failed: {e}")

        # Try fallbacks
        for fallback in fallback_chain:
            if fallback == provider:
                continue
            try:
                logger.info(f"Trying fallback provider: {fallback}")
                model = LLMFactory.get_chat_model(fallback, temperature=temperature, max_tokens=max_tokens)
                registry = get_provider(fallback)
                return model, fallback, registry.default_chat_model if registry else fallback
            except Exception as e:
                logger.warning(f"Fallback provider '{fallback}' also failed: {e}")

        raise RuntimeError("All LLM providers failed. Check API keys and connectivity.")
