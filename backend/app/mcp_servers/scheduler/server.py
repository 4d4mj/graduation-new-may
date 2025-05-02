# scheduler_mcp/server.py
from __future__ import annotations
import os, uuid, datetime as dt
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import logging

# Add basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set!")
    # Consider exiting or setting a default for local dev if appropriate
    # exit(1)
    DATABASE_URL = "postgresql+asyncpg://user:pass@host/db" # Example Placeholder

ENGINE       = create_async_engine(DATABASE_URL, echo=False)
Session      = sessionmaker(ENGINE, class_=AsyncSession, expire_on_commit=False)

app = FastAPI(
    title="scheduler-mcp",
    description="MCP tool-server that manages appointments",
    version="0.1.0"
)

# ─────────────────────────────────────────────────────────────
# Pydantic IO models  (→ become MCP tool signatures)
# ─────────────────────────────────────────────────────────────
class Slot(BaseModel):
    doctor_id : int
    starts_at : dt.datetime
    ends_at   : dt.datetime
    location  : str

class BookRequest(BaseModel):
    patient_id : int
    doctor_id  : int
    starts_at  : dt.datetime
    ends_at    : dt.datetime
    location   : str = Field(..., max_length=128)
    notes      : str | None = None

class CancelRequest(BaseModel):
    appointment_id : int
    patient_id     : int     # optional extra auth check

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
async def execute(sql:str, **params):
    async with Session() as session:
        await session.execute(text(sql), params)
        await session.commit()

async def fetchall(sql:str, **params):
    async with Session() as session:
        res = await session.execute(text(sql), params)
        rows = res.mappings().all()
        return [dict(r) for r in rows]

# ─────────────────────────────────────────────────────────────
# MCP tool endpoints
# Each path == tool.name
# ─────────────────────────────────────────────────────────────
@app.post("/list_free_slots", response_model=list[Slot])
async def list_free_slots(doctor_id:int, day:str):
    """
    Return 30‑min free slots for a doctor on given ISO‑date.
    """
    day_start = f"{day} 00:00+00"
    day_end   = f"{day} 23:59+00"
    # naive example: you can pre‑generate slots table instead
    rows = await fetchall("""
        SELECT starts_at, ends_at
        FROM appointments
        WHERE doctor_id = :doc
          AND tstzrange(starts_at, ends_at, '[)') && tstzrange(:ds, :de, '[)')
        ORDER BY starts_at
    """, doc=doctor_id, ds=day_start, de=day_end)

    # … derive gaps between rows to yield free slots …
    # omitted for brevity – just return mocked slots for now
    return [Slot(doctor_id=doctor_id,
                 starts_at=dt.datetime.fromisoformat(day_start)+dt.timedelta(hours=9),
                 ends_at  =dt.datetime.fromisoformat(day_start)+dt.timedelta(hours=9, minutes=30),
                 location="Main clinic")]

@app.post("/book_appointment")
async def book_appointment(req: BookRequest):
    # very naive overlap check – rely on UNIQUE constraint too
    try:
        await execute("""
            INSERT INTO appointments
              (patient_id, doctor_id, starts_at, ends_at, location, notes)
            VALUES (:pat, :doc, :st, :en, :loc, :notes)
        """, pat=req.patient_id, doc=req.doctor_id, st=req.starts_at,
             en=req.ends_at, loc=req.location, notes=req.notes)
    except Exception as e:
        raise HTTPException(400, f"Booking failed: {e}")
    return {"ok": True, "starts_at": req.starts_at, "location": req.location}

@app.post("/cancel_appointment")
async def cancel_appointment(req: CancelRequest):
    await execute("""
        DELETE FROM appointments
        WHERE id=:id AND patient_id=:pid
    """, id=req.appointment_id, pid=req.patient_id)
    return {"ok": True}

# --- Add a basic SSE endpoint for client connection ---
@app.get("/sse")
async def sse_endpoint(request: Request) -> Dict[str, Any]:
    """
    Placeholder endpoint for MCP client using SSE transport.
    Actual tool discovery might happen via other means by the client library,
    but this endpoint might be needed for the initial connection.
    """
    logger.info(f"Received connection request on /sse from {request.client.host if request.client else 'unknown'}")
    # You might return basic server info or just status ok
    # The library might handle the actual SSE protocol internally
    return {"status": "ok", "message": "Scheduler MCP SSE endpoint ready."}
