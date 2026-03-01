import logging
from typing import Optional, Iterator
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.llm.factory import LLMFactory
from src.llm.prompts import get_system_prompt
from src.llm.model_registry import get_provider
from src.agents.tools import vector_search, graph_traversal, set_clients, update_embedding_model

logger = logging.getLogger(__name__)


class GraphragAgent:
    """Autonomous ReAct agent for routing OpenSearch and Neo4j queries."""

    def __init__(
        self,
        opensearch_client,
        neo4j_client,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ):
        self.opensearch_client = opensearch_client
        self.neo4j_client = neo4j_client

        # Store current config
        self.current_provider = provider
        self.current_model = model_name
        self.current_temperature = temperature
        self.current_max_tokens = max_tokens

        # Initialize the global tools with the singleton clients
        self.embedding_model = LLMFactory.get_embeddings(provider=provider)
        set_clients(self.opensearch_client, self.neo4j_client, self.embedding_model)

        # Pull the primary chat model
        self.llm = LLMFactory.get_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Define the tools available to the agent
        self.tools = [vector_search, graph_traversal]

        # Create the LangGraph compiled ReAct agent with provider-optimized prompt
        system_prompt = get_system_prompt(provider or "ollama")

        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt,
        )

        # Resolve display names
        registry = get_provider(provider or "ollama")
        if not self.current_model and registry:
            self.current_model = registry.default_chat_model

        logger.info(
            f"GraphragAgent compiled: provider={self.current_provider}, "
            f"model={self.current_model}, temp={temperature}"
        )

    def reinitialize(
        self,
        provider: str,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        """Rebuild the LLM and agent executor without touching DB clients."""
        logger.info(f"Reinitializing agent: provider={provider}, model={model_name}")

        # Update embedding model
        self.embedding_model = LLMFactory.get_embeddings(provider=provider)
        update_embedding_model(self.embedding_model)

        # Rebuild chat model
        self.llm = LLMFactory.get_chat_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Rebuild agent with provider-optimized prompt
        system_prompt = get_system_prompt(provider)
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt,
        )

        # Update stored config
        self.current_provider = provider
        registry = get_provider(provider)
        self.current_model = model_name or (registry.default_chat_model if registry else provider)
        self.current_temperature = temperature
        self.current_max_tokens = max_tokens

        logger.info(f"Agent reinitialized: provider={provider}, model={self.current_model}")

    def run(self, query: str) -> str:
        """Executes the agent synchronously and returns the final answer."""
        inputs = {"messages": [("user", query)]}
        try:
            result = self.agent_executor.invoke(inputs)
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Attempt fallback
            try:
                model, fallback_provider, fallback_model = LLMFactory.get_chat_model_with_fallback(
                    self.current_provider,
                    temperature=self.current_temperature,
                    max_tokens=self.current_max_tokens,
                )
                fallback_prompt = get_system_prompt(fallback_provider)
                fallback_executor = create_react_agent(model, self.tools, prompt=fallback_prompt)
                result = fallback_executor.invoke(inputs)
                return (
                    f"*[Fallback: switched from {self.current_provider} to {fallback_provider}]*\n\n"
                    + result["messages"][-1].content
                )
            except Exception as fallback_err:
                return f"Error: All providers failed. Primary: {e}. Fallback: {fallback_err}"

    def stream_steps(self, query: str) -> Iterator[str]:
        """Streams the thought process and intermediate tool calls of the agent."""
        inputs = {"messages": [("user", query)]}

        try:
            for step in self.agent_executor.stream(inputs, stream_mode="values"):
                message = step["messages"][-1]
                if isinstance(message, HumanMessage):
                    continue

                # If it's an AIMessage with a tool call
                if getattr(message, "tool_calls", None):
                    for tc in message.tool_calls:
                        yield f"**🧠 Agent Thought:** I need to use the `{tc['name']}` tool with arguments: `{tc['args']}`"

                # If it's a ToolMessage returning results
                elif getattr(message, "name", None):
                    yield f"**🛠️ Tool Output ({message.name}):**\n```\n{message.content[:500]}...\n```"

                # If it's the final answer
                elif message.content:
                    yield f"\n**✅ Final Answer:**\n{message.content}"

        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"**⚠️ Error with {self.current_provider}:** {str(e)}"
            yield "Attempting fallback..."
            try:
                result = self.run(query)  # run() has built-in fallback
                yield f"\n**✅ Final Answer (via fallback):**\n{result}"
            except Exception as fallback_err:
                yield f"**❌ All providers failed:** {str(fallback_err)}"
