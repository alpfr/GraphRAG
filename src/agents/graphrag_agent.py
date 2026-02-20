import logging
from typing import AsyncGenerator, Iterator
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from src.llm.factory import LLMFactory
from src.agents.tools import vector_search, graph_traversal, set_clients

logger = logging.getLogger(__name__)

class GraphragAgent:
    """Autonomous ReAct agent for routing OpenSearch and Neo4j queries."""

    def __init__(self, opensearch_client, neo4j_client):
        self.opensearch_client = opensearch_client
        self.neo4j_client = neo4j_client
        
        # Initialize the global tools with the singleton clients
        self.embedding_model = LLMFactory.get_embeddings()
        set_clients(self.opensearch_client, self.neo4j_client, self.embedding_model)
        
        # Pull the primary chat model
        self.llm = LLMFactory.get_chat_model()
        
        # Define the tools available to the agent
        self.tools = [vector_search, graph_traversal]
        
        # Create the LangGraph compiled ReAct agent
        system_prompt = (
            "You are an advanced Next-Gen GraphRAG Artificial Intelligence Assistant. \n"
            "You have access to an OpenSearch Vector Database and a Neo4j Knowledge Graph. \n"
            "Use the `vector_search` tool to look up general knowledge, semantic context, and read document chunks. \n"
            "Use the `graph_traversal` tool to understand how specific entities (people, organizations, concepts) are related to one another in the graph map.\n"
            "Answer the user's question completely. If you do not know the answer, state that you do not know. "
            "Always cite your sources if possible based on the tool outputs. "
            "Think step-by-step. If a user asks about complex connections, use the graph traversal tool. If they ask about general content, use the vector search tool."
        )
        
        self.agent_executor = create_react_agent(
            self.llm,
            self.tools,
            prompt=system_prompt
        )
        logger.info("GraphragAgent compiled successfully.")

    def run(self, query: str) -> str:
        """Executes the agent synchronously and returns the final answer."""
        inputs = {"messages": [("user", query)]}
        result = self.agent_executor.invoke(inputs)
        return result["messages"][-1].content
        
    def stream_steps(self, query: str) -> Iterator[str]:
        """Streams the thought process and intermediate tool calls of the agent."""
        inputs = {"messages": [("user", query)]}
        
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
