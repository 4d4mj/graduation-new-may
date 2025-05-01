from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .config.settings import settings
from .db.base import get_engine
from .db.session import get_db_session
import logging

from app.core.mcp import MCPToolManager
from app.config.mcp import load_mcp_config
from app.routes.chat.router import init_graphs

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# lifespan context manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup…")

    # --- DATABASE ---
    engine = await get_engine(str(settings.database_url))
    app.state.engine = engine
    logger.info("DB engine ready")

    # --- MCP TOOL MANAGER ---
    mcp_server_configs = load_mcp_config()
    tool_manager = MCPToolManager(mcp_server_configs)
    app.state.tool_manager = tool_manager
    try:
        await tool_manager.start_client()
        if tool_manager.is_running:
            logger.info("MCP Tool Manager started successfully")
        else:
            logger.error("MCP Tool Manager failed to start")
    except Exception as e:
        logger.exception("Critical error starting MCP Tool Manager")

    # Initialize graphs for both patient and doctor roles
    app.state.graphs = init_graphs()
    logger.info(f"Initialized graphs for roles: {list(app.state.graphs.keys())}")

    # give control back to FastAPI…
    yield

    # --- SHUTDOWN SEQUENCE (reverse order) ---
    logger.info("Application shutdown…")

    if hasattr(app.state, "tool_manager") and app.state.tool_manager.is_running:
        try:
            await app.state.tool_manager.stop_client()
            logger.info("MCP Tool Manager stopped")
        except Exception:
            logger.exception("Error stopping MCP Tool Manager")
    else:
        logger.info("No running MCP Tool Manager to stop")

    if hasattr(app.state, "engine"):
        try:
            await app.state.engine.dispose()
            logger.info("Database engine disposed")
        except Exception:
            logger.exception("Error disposing database engine")

    logger.info("Shutdown complete")

app = FastAPI(title="MultiAgent Medical Assistant", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
async def get_db():
    async for session in get_db_session(str(settings.database_url)):
        yield session

# health check
@app.get("/health")
async def health_check(request: Request):
    """basic health check endpoint"""
    tool_manager_status = "stopped"
    graphs_status = {}

    if hasattr(request.app.state, "tool_manager"):
        tool_manager_status = (
            "running" if request.app.state.tool_manager.is_running else "stopped"
        )

    if hasattr(request.app.state, "graphs"):
        for role, graph in request.app.state.graphs.items():
            graphs_status[role] = "loaded" if graph is not None else "not loaded"

    return {
        "status": "ok",
        "mcp_client": tool_manager_status,
        "graphs": graphs_status,
    }

from app.routes.auth.router import router as auth_router
from app.routes.chat.router import router as chat_router
app.include_router(auth_router)
app.include_router(chat_router)



