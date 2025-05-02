from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config.settings import settings
from app.db.base import get_engine
from app.db.session import get_db_session

# Optional: generic MCP (e.g. Tavily) -------------------------------------------------
from app.config.mcp import load_mcp_config
from app.core.mcp import MCPToolManager

# LangGraph orchestration -------------------------------------------------------------
from app.routes.chat.router import init_graphs
from app.graphs.sub import agent_node  # import the module, not just the variable

# -------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------
# FastAPI helpers
# -------------------------------------------------------------------------------------
async def get_db():
    """Yield an async SQLAlchemy session (dependency)."""
    async for session in get_db_session(str(settings.database_url)):
        yield session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application start-up & shutdown hooks."""
    # ------------------------------------------------------------------ start‑up -----
    logger.info("Application startup …")

    # 1️⃣  Database --------------------------------------------------
    engine = await get_engine(str(settings.database_url))
    app.state.engine = engine
    logger.info("DB engine ready")

    # 2️⃣  MCP tool discovery (optional) -----------------------------
    mcp_tools = []  # will stay empty if no servers / failure
    try:
        mcp_servers = load_mcp_config()
        if mcp_servers:
            tool_manager = MCPToolManager(mcp_servers)
            app.state.tool_manager = tool_manager
            await tool_manager.start_client()

            if tool_manager.is_running:
                mcp_tools = tool_manager.get_all_tools()
                names = [t.name for t in mcp_tools]
                logger.info(f"MCP client running – discovered tools: {names if names else 'none'}")
            else:
                logger.warning("MCP client not running (no active servers or connection failed)")
        else:
            logger.info("No MCP servers configured – skipping client startup")
    except Exception:
        logger.exception("Unexpected error while initialising MCP client – continuing without MCP tools")
        # ensure downstream code still works
        app.state.tool_manager = None

    # 3️⃣  Build medical agent  --------------------------------------
    agent_node.medical_agent = agent_node.build_medical_agent(mcp_tools)

    # 4️⃣  Compile LangGraph graphs ----------------------------------
    app.state.graphs = init_graphs()
    logger.info(f"Graphs initialised for roles: {list(app.state.graphs.keys())}")

    # ------------------------------------------------ give control back
    yield

    # ------------------------------------------------ shutdown --------
    logger.info("Application shutdown …")

    # Stop MCP client first (if any)
    if (tm := getattr(app.state, "tool_manager", None)) and tm.is_running:
        try:
            await tm.stop_client()
            logger.info("MCP client stopped")
        except Exception:
            logger.exception("Error stopping MCP client")

    # Dispose DB engine
    if (eng := getattr(app.state, "engine", None)):
        try:
            await eng.dispose()
            logger.info("DB engine disposed")
        except Exception:
            logger.exception("Error disposing DB engine")

    logger.info("Shutdown complete")


# -------------------------------------------------------------------------------------
# FastAPI application instance
# -------------------------------------------------------------------------------------
app = FastAPI(title="Multi‑Agent Medical Assistant", lifespan=lifespan)

# CORS -------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(o) for o in settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- health‑check -----
@app.get("/health")
async def health_check(request: Request):
    tool_mgr_state = "stopped"
    if tm := getattr(request.app.state, "tool_manager", None):
        tool_mgr_state = "running" if tm.is_running else "stopped"

    graphs_status = {}
    if graphs := getattr(request.app.state, "graphs", None):
        graphs_status = {role: "loaded" if g else "not loaded" for role, g in graphs.items()}

    return {
        "status": "ok",
        "mcp_client": tool_mgr_state,
        "graphs": graphs_status,
    }


# ------------------------------------------------------------------- routes ---------
from app.routes.auth.router import router as auth_router  # noqa: E402  (after app creation)
from app.routes.chat.router import router as chat_router  # noqa: E402

app.include_router(auth_router)
app.include_router(chat_router)
