import json
import os
import logging
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


def load_mcp_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads MCP server configuration from JSON file, resolves env variables,
    and transforms the structure to match MultiServerMCPClient expectations.
    """
    transformed_servers = {}

    if config_path is None:
        module_dir = Path(__file__).parent
        config_file = module_dir / "mcp_servers.json"
    else:
        config_file = Path(config_path)

    if not config_file.is_file():
        logger.error(f"MCP configuration file was not found at {config_file}")
        return transformed_servers

    try:
        with open(config_file, "r") as f:
            raw_config_data = json.load(f)

        logger.info(f"Loading MCP configuration from {config_file}")

        for server_name, raw_conf in raw_config_data.items():
            # Start with a clean dictionary for the transformed config
            transformed_conf = {}

            # --- Copy essential flags first ---
            transformed_conf["disabled"] = raw_conf.get("disabled", False)
            transformed_conf["auto_approve"] = raw_conf.get("autoApprove", raw_conf.get("auto_approve", [])) # Allow both casings

            # --- Environment variable resolution (if needed) ---
            if "env" in raw_conf and isinstance(raw_conf["env"], dict):
                resolved_env = {}
                for key, value in raw_conf["env"].items():
                    if isinstance(value, str) and value.startswith("env:"):
                        env_var_name = value[4:]
                        env_var_value = os.getenv(env_var_name)
                        if env_var_value is None:
                            logger.warning(
                                f"Environment variable {env_var_name} not found for MCP server {server_name}, key '{key}'. Using empty string"
                            )
                            resolved_env[key] = ""
                        else:
                            resolved_env[key] = env_var_value
                    else:
                        resolved_env[key] = value
                transformed_conf["env"] = resolved_env


            # --- Transformation based on connection type ---
            connection_type = raw_conf.get("connection_type", raw_conf.get("connectionType", "")).lower() # Allow both casings

            if connection_type == "http":
                base_url = raw_conf.get("base_url", raw_conf.get("baseUrl", "")) # Allow both casings
                if not base_url:
                     logger.warning(f"HTTP server '{server_name}' missing 'base_url'. Skipping.")
                     continue # Skip this server if essential info is missing

                # Use 'sse' transport and 'url' as per documentation for HTTP-based servers
                transformed_conf["transport"] = "sse" # Use 'sse' as shown in docs
                transformed_conf["url"] = f"{base_url}/sse" # Construct the expected SSE endpoint URL
                logger.info(f"Transformed '{server_name}' config for HTTP: transport='sse', url='{transformed_conf['url']}'")

            elif connection_type == "stdio":
                 command = raw_conf.get("command")
                 args = raw_conf.get("args")
                 if not command:
                     logger.warning(f"Stdio server '{server_name}' missing 'command'. Skipping.")
                     continue # Skip this server

                 transformed_conf["transport"] = "stdio"
                 transformed_conf["command"] = command
                 transformed_conf["args"] = args if args is not None else [] # Ensure args is a list
                 logger.info(f"Transformed '{server_name}' config for Stdio: transport='stdio', command='{command}'")

            else:
                logger.warning(f"Unknown connection_type '{connection_type}' for server '{server_name}'. Skipping.")
                continue # Skip unsupported types

            transformed_servers[server_name] = transformed_conf

        logger.info(
            f"Successfully loaded and transformed {len(transformed_servers)} MCP server configurations"
        )
        logger.debug(f"Transformed MCP Config for Client: {transformed_servers}")
        return transformed_servers

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing MCP JSON configuration file '{config_path}' : {e}")
        return {}
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading/transforming MCP config '{config_path}' : {e}"
        )
        return {}
