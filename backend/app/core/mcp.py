import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from typing import Dict, List, Any, Optional


logger = logging.getLogger(__name__)


class MCPToolManager:
    """Manages the lifecycle of the MultiServerMCPClient and provide tools"""

    def __init__(self, server_configs: Dict[str, Any]):
        """
        Initializes the manager with server configurations

        Args:
            server_configs (Dict[str, Any]): A dictionairy where keys are server names and
                values are their config dictionairies
        """
        self._raw_server_configs = server_configs
        self._active_server_configs: Dict[str, Any] = {}
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: List[BaseTool] = []
        self._lock = asyncio.Lock()
        self._is_running = False

    async def start_client(self):
        """Starts the MultiServerMCPClient with active configs"""
        async with self._lock:
            if self._is_running:
                logger.info("MCP client is already running")
                return

            logger.info("Starting MCP Client")

            self._active_server_configs = {
                name: config
                for name, config in self._raw_server_configs.items()
                if not config.get("disabled", False)
            }

            if not self._active_server_configs:
                logger.warning(
                    "MCPToolManager: No active MCP servers configured. Client will not start"
                )
                self._client = None
                self._tools = []
                self._is_running = False
                return

            try:
                logger.info(
                    f"Initializing MCP Client with servers: {list(self._active_server_configs.keys())}"
                )
                self._client = MultiServerMCPClient(self._active_server_configs)

                await self._client.__aenter__()
                self._is_running = True
                logger.info("MCP client is starting...")
                self._tools = self._client.get_tools()
                logger.info(
                    f"MCP client started. Tools available: {[t.name for t in self._tools]}"
                )

                if not self._tools and self._active_server_configs:
                    logger.warning(
                        "MCP client started but found no active tools. Check MCP server logs and configurations"
                    )

            except Exception as e:
                logger.error(f"Error starting MCP client: {e}", exc_info=True)

                if self._client:
                    try:
                        await self._client.__aexit__(None, None, None)
                    except Exception as cleanup_e:
                        logger.error(
                            f"Error during cleanup after failed start: {cleanup_e}",
                            exc_info=True,
                        )
                self._client = None
                self._tools = []
                self._is_running = False

    async def stop_client(self):
        """Stops the MultiServerMCPClient and cleans up resources"""
        async with self._lock:
            if not self._is_running or not self._client:
                logger.info("MCP client is not running or not initialized")
                return

            logger.info("Stopping MCP client...")
            try:
                await self._client.__aexit__(None, None, None)
                logger.info("MCP clinent stopped succesfully")
            except Exception as e:
                logger.error(f"Error stopping MCP client {e}", exc_info=True)
            finally:
                self._client = None
                self._tools = []
                self._is_running = False

    def get_all_tools(self) -> List[BaseTool]:
        """Returns a list of all tools obtained from the active MCP servers."""
        if not self._is_running:
            logger.warning("MCP Client not running. Cannot get tools.")
            return []
        return self._tools

    def get_tools_for_agent(
        self, required_tool_names: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """
        Returns a filtered list of tools based on names

        Args:
            required_tool_names (Optional[List[str]], optional): A list of tool names required by the agent
                                                                        if None, returns all available tools.

        Returns:
            List[BaseTool]: A list of BaseTool objects matching the requirements
        """
        all_tools = self.get_all_tools()
        if not required_tool_names:
            return all_tools

        agent_tools = [tool for tool in all_tools if tool.name in required_tool_names]

        if len(agent_tools) != len(required_tool_names):
            found_names = {t.name for t in agent_tools}
            missing_names = set(required_tool_names) - found_names
            logger.warning(
                f"Could not find all required tools. Missing: {list(missing_names)}"
            )

        return agent_tools

    @property
    def is_running(self) -> bool:
        """Returns True if the client is currently running"""
        return self._is_running
