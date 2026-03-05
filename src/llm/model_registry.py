"""
Central registry of all LLM providers, models, and their configurations.
Single source of truth for model metadata, costs, and capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    display_name: str
    provider: str
    model_type: str  # "chat" or "embedding"
    default_temperature: float = 0.1
    max_temperature: float = 2.0
    default_max_tokens: int = 4096
    max_tokens_limit: int = 128000
    supports_streaming: bool = True
    supports_tool_calling: bool = True
    embedding_dimensions: Optional[int] = None
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider_id: str
    display_name: str
    requires_api_key: bool = True
    api_key_env_var: str = ""
    chat_models: List[ModelConfig] = field(default_factory=list)
    embedding_models: List[ModelConfig] = field(default_factory=list)
    default_chat_model: str = ""
    default_embedding_model: str = ""
    prompt_style: str = "standard"  # "standard", "xml", "function_calling"


# ──────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        provider_id="openai",
        display_name="OpenAI",
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        default_chat_model="gpt-4o",
        default_embedding_model="text-embedding-3-small",
        prompt_style="function_calling",
        chat_models=[
            ModelConfig(
                model_id="gpt-4o",
                display_name="GPT-4o",
                provider="openai",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=128000,
                cost_per_1k_input=0.0025,
                cost_per_1k_output=0.01,
            ),
            ModelConfig(
                model_id="gpt-4o-mini",
                display_name="GPT-4o Mini",
                provider="openai",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=128000,
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
            ),
        ],
        embedding_models=[
            ModelConfig(
                model_id="text-embedding-3-small",
                display_name="Embedding 3 Small",
                provider="openai",
                model_type="embedding",
                embedding_dimensions=1536,
                cost_per_1k_input=0.00002,
            ),
            ModelConfig(
                model_id="text-embedding-3-large",
                display_name="Embedding 3 Large",
                provider="openai",
                model_type="embedding",
                embedding_dimensions=3072,
                cost_per_1k_input=0.00013,
            ),
        ],
    ),

    "anthropic": ProviderConfig(
        provider_id="anthropic",
        display_name="Anthropic",
        requires_api_key=True,
        api_key_env_var="ANTHROPIC_API_KEY",
        default_chat_model="claude-sonnet-4-20250514",
        default_embedding_model="",  # No native embeddings — falls back to OpenAI
        prompt_style="xml",
        chat_models=[
            ModelConfig(
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                provider="anthropic",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=200000,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
            ),
            ModelConfig(
                model_id="claude-3-5-haiku-20241022",
                display_name="Claude 3.5 Haiku",
                provider="anthropic",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=200000,
                cost_per_1k_input=0.0008,
                cost_per_1k_output=0.004,
            ),
        ],
        embedding_models=[],  # Anthropic has no embedding models
    ),

    "gemini": ProviderConfig(
        provider_id="gemini",
        display_name="Google Gemini",
        requires_api_key=True,
        api_key_env_var="GOOGLE_API_KEY",
        default_chat_model="gemini-2.0-flash",
        default_embedding_model="models/text-embedding-004",
        prompt_style="standard",
        chat_models=[
            ModelConfig(
                model_id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                provider="gemini",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=1000000,
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0004,
            ),
            ModelConfig(
                model_id="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                provider="gemini",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=2000000,
                cost_per_1k_input=0.00125,
                cost_per_1k_output=0.005,
            ),
        ],
        embedding_models=[
            ModelConfig(
                model_id="models/text-embedding-004",
                display_name="Text Embedding 004",
                provider="gemini",
                model_type="embedding",
                embedding_dimensions=768,
                cost_per_1k_input=0.00001,
            ),
        ],
    ),

    "ollama": ProviderConfig(
        provider_id="ollama",
        display_name="Ollama (Local)",
        requires_api_key=False,
        api_key_env_var="",
        default_chat_model="ibm/granite4:latest",
        default_embedding_model="granite-embedding:278m",
        prompt_style="standard",
        chat_models=[
            ModelConfig(
                model_id="ibm/granite4:latest",
                display_name="IBM Granite 4",
                provider="ollama",
                model_type="chat",
                default_temperature=0.1,
                max_tokens_limit=32768,
                supports_tool_calling=True,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
            ),
        ],
        embedding_models=[
            ModelConfig(
                model_id="granite-embedding:278m",
                display_name="Granite Embedding 278M",
                provider="ollama",
                model_type="embedding",
                embedding_dimensions=768,
                cost_per_1k_input=0.0,
            ),
        ],
    ),
}


# ──────────────────────────────────────────────
# Embedding Dimension Map
# ──────────────────────────────────────────────

EMBEDDING_DIMENSION_MAP: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "models/text-embedding-004": 768,
    "granite-embedding:278m": 768,
}


def get_provider(provider_id: str) -> Optional[ProviderConfig]:
    """Get a provider config by ID."""
    return PROVIDER_REGISTRY.get(provider_id)


def get_chat_models(provider_id: str) -> List[ModelConfig]:
    """Get available chat models for a provider."""
    provider = PROVIDER_REGISTRY.get(provider_id)
    return provider.chat_models if provider else []


def get_embedding_models(provider_id: str) -> List[ModelConfig]:
    """Get available embedding models for a provider."""
    provider = PROVIDER_REGISTRY.get(provider_id)
    return provider.embedding_models if provider else []


def get_embedding_dimension(model_id: str) -> int:
    """Get embedding dimension for a model, defaults to 768."""
    return EMBEDDING_DIMENSION_MAP.get(model_id, 768)


def get_model_cost(provider_id: str, model_id: str) -> tuple:
    """Get (cost_per_1k_input, cost_per_1k_output) for a model."""
    provider = PROVIDER_REGISTRY.get(provider_id)
    if provider:
        for model in provider.chat_models + provider.embedding_models:
            if model.model_id == model_id:
                return (model.cost_per_1k_input or 0.0, model.cost_per_1k_output or 0.0)
    return (0.0, 0.0)


def estimate_cost(provider_id: str, model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given number of tokens."""
    cost_in, cost_out = get_model_cost(provider_id, model_id)
    return (input_tokens / 1000 * cost_in) + (output_tokens / 1000 * cost_out)
