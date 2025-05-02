import json
import os
import logging
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


def load_mcp_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads MCP server configuration from JSON file and resolve env variables

    Args:
        config_path (str, optional): path to the JSON configuration file.
                                    If None, will look for the file relative to this module.

    Returns:
        A dictionairy containing the configuration for each MCP server.
        Returns an empty dictionary if the file is not found or if there is an error in parsing the JSON.
    """
    resolved_servers = {}

    # If no path provided, use the mcp_servers.json file in the same directory as this module
    if config_path is None:
        module_dir = Path(__file__).parent
        config_file = module_dir / "mcp_servers.json"
    else:
        config_file = Path(config_path)

    if not config_file.is_file():
        logger.error(f"MCP configuration file was not found at {config_file}")
        return resolved_servers

    try:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        logger.info(f"Loading MCP configuration from {config_file}")

        for server_name, server_conf in config_data.items():
            resolved_conf = server_conf.copy()
            # app/config/mcp.py  ─ inside the for‑loop, right after  resolved_conf = server_conf.copy()
            resolved_conf["connection_type"] = resolved_conf.pop("connectionType", resolved_conf.get("connection_type", "")).lower()
            resolved_conf["base_url"]        = resolved_conf.pop("baseUrl",        resolved_conf.get("base_url", ""))
            resolved_conf["auto_approve"]    = resolved_conf.pop("autoApprove",    resolved_conf.get("auto_approve", []))


            if "env" in resolved_conf and isinstance(resolved_conf["env"], dict):
                resolved_env = {}
                for key, value in resolved_conf["env"].items():
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
                            logger.debug(
                                f"resolved env var '{env_var_name}' for MCP server {server_name}"
                            )
                    else:
                        resolved_env[key] = value
                resolved_conf["env"] = resolved_env

            resolved_servers[server_name] = resolved_conf

        logger.info(
            f"succesfully loaded and resolved {len(resolved_servers)} MCP server configurations"
        )
        return resolved_servers

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing MCP JSON configuration file '{config_path}' : {e}")
        return {}
    except Exception as e:
        logger.error(
            f"An unexpected error occured while loading MCP config '{config_path}' : {e}"
        )
        return {}
