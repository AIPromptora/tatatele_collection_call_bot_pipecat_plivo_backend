import asyncpg
import os
from datetime import datetime, date
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_pool: asyncpg.Pool | None = None


async def init_db():
    global _pool
    _pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"), min_size=2, max_size=20)
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id    TEXT PRIMARY KEY,
                file_name   TEXT NOT NULL,
                total       INTEGER NOT NULL DEFAULT 0,
                completed   INTEGER NOT NULL DEFAULT 0,
                active      INTEGER NOT NULL DEFAULT 0,
                failed      INTEGER NOT NULL DEFAULT 0,
                pending     INTEGER NOT NULL DEFAULT 0,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS calls (
                call_uuid       TEXT PRIMARY KEY,
                batch_id        TEXT REFERENCES batches(batch_id),
                user_id         TEXT,
                access_token    TEXT,
                customer_name   TEXT NOT NULL,
                phone_number    TEXT NOT NULL,
                service_name    TEXT NOT NULL,
                amount          TEXT NOT NULL DEFAULT '0',
                billing_period  TEXT NOT NULL DEFAULT '',
                language        TEXT NOT NULL DEFAULT 'English',
                status          TEXT NOT NULL DEFAULT 'pending',
                created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS user_id TEXT")
        await conn.execute("ALTER TABLE calls ADD COLUMN IF NOT EXISTS access_token TEXT")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id          SERIAL PRIMARY KEY,
                call_uuid   TEXT REFERENCES calls(call_uuid),
                role        TEXT NOT NULL,
                text        TEXT NOT NULL,
                turn_index  INTEGER NOT NULL
            )
        """)
    logger.info("Database initialised — all tables ready")


async def close_db():
    if _pool:
        await _pool.close()


def _row(r):
    if not r:
        return None
    out = {}
    for k, v in dict(r).items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, date):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ── Batches ────────────────────────────────────────────────────────────────────

async def insert_batch(batch_id: str, file_name: str, total: int):
    async with _pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO batches (batch_id, file_name, total, pending)
               VALUES ($1, $2, $3, $3)""",
            batch_id, file_name, total
        )


async def get_batches():
    async with _pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM batches ORDER BY created_at DESC")
        return [_row(r) for r in rows]


async def get_batch(batch_id: str):
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM batches WHERE batch_id=$1", batch_id)
        return _row(row)


async def update_batch_counts(batch_id: str):
    async with _pool.acquire() as conn:
        await conn.execute("""
            UPDATE batches SET
                completed = (SELECT COUNT(*) FROM calls WHERE batch_id=$1 AND status='completed'),
                active    = (SELECT COUNT(*) FROM calls WHERE batch_id=$1 AND status='active'),
                failed    = (SELECT COUNT(*) FROM calls WHERE batch_id=$1 AND status='failed'),
                pending   = (SELECT COUNT(*) FROM calls WHERE batch_id=$1 AND status='pending')
            WHERE batch_id=$1
        """, batch_id)


# ── Calls ──────────────────────────────────────────────────────────────────────

async def insert_call(
    call_uuid: str,
    batch_id: str | None,
    user_id: str | None,
    access_token: str | None,
    customer_name: str,
    phone_number: str,
    service_name: str,
    amount: str,
    billing_period: str,
    language: str,
):
    async with _pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO calls
                (call_uuid, batch_id, user_id, access_token, customer_name, phone_number,
                 service_name, amount, billing_period, language)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """, call_uuid, batch_id, user_id, access_token, customer_name, phone_number,
             service_name, amount, billing_period, language)


async def update_call_status(call_uuid: str, status: str):
    async with _pool.acquire() as conn:
        await conn.execute(
            "UPDATE calls SET status=$1 WHERE call_uuid=$2", status, call_uuid
        )


async def get_calls(batch_id: str | None = None):
    async with _pool.acquire() as conn:
        if batch_id:
            rows = await conn.fetch(
                "SELECT * FROM calls WHERE batch_id=$1 ORDER BY created_at ASC", batch_id
            )
        else:
            rows = await conn.fetch("SELECT * FROM calls ORDER BY created_at DESC")
        return [_row(r) for r in rows]


async def get_call(call_uuid: str):
    async with _pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM calls WHERE call_uuid=$1", call_uuid)
        return _row(row)


# ── Transcripts ────────────────────────────────────────────────────────────────

async def insert_transcript(call_uuid: str, turns: list[dict]):
    async with _pool.acquire() as conn:
        await conn.executemany("""
            INSERT INTO transcripts (call_uuid, role, text, turn_index)
            VALUES ($1, $2, $3, $4)
        """, [(call_uuid, t["role"], t["text"], i) for i, t in enumerate(turns)])


async def get_transcript(call_uuid: str):
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM transcripts WHERE call_uuid=$1 ORDER BY turn_index ASC",
            call_uuid
        )
        return [_row(r) for r in rows]
