import base64
import asyncio
import json
import os
import sys
import time
import urllib.parse

import aiohttp
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from loguru import logger as _logger
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Logging filter ─────────────────────────────────────────────────────────────
_logger.remove()

def _log_filter(record):
    level = record["level"].no
    name  = record["name"]
    msg   = record["message"]
    if level >= 30:
        return True
    if name.startswith(("helpers", "routers", "__main__")):
        return level >= 20
    if "TTFB" in msg:
        return True
    if "STT: [" in msg:
        return True
    if "Generating TTS" in msg:
        return True
    return False

_logger.add(
    sys.stderr,
    filter=_log_filter,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv(override=True)

from helpers.supabase_push import push_to_supabase

# ── WebRTC state ──────────────────────────────────────────────────────────────

_pcs_map: dict[str, SmallWebRTCConnection] = {}

_ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]

# ── Call rate limiting ─────────────────────────────────────────────────────────

CALL_INITIATION_GAP_MS = 1000
_call_initiation_lock = asyncio.Lock()
_next_call_slot_at = 0.0


async def _wait_for_call_slot() -> None:
    """
    Ensure outbound calls are initiated with a minimum spacing.
    """
    global _next_call_slot_at

    async with _call_initiation_lock:
        now = time.monotonic()
        scheduled_at = max(now, _next_call_slot_at)
        _next_call_slot_at = scheduled_at + (CALL_INITIATION_GAP_MS / 1000.0)

    wait_seconds = scheduled_at - time.monotonic()
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)


# ── Supabase helpers ───────────────────────────────────────────────────────────

# ── Plivo call helper ──────────────────────────────────────────────────────────

async def make_plivo_call(
    session: aiohttp.ClientSession,
    to_number: str,
    from_number: str | None,
    answer_url: str,
):
    auth_id    = os.getenv("PLIVO_AUTH_ID")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN")

    if not auth_id:
        raise ValueError("Missing PLIVO_AUTH_ID")
    if not auth_token:
        raise ValueError("Missing PLIVO_AUTH_TOKEN")

    data = {
        "to":            to_number,
        "from":          from_number,
        "answer_url":    answer_url,
        "answer_method": "POST",
    }
    url  = f"https://api.plivo.com/v1/Account/{auth_id}/Call/"
    auth = aiohttp.BasicAuth(auth_id, auth_token)

    async with session.post(url, json=data, auth=auth) as resp:
        if resp.status != 201:
            err = await resp.text()
            raise Exception(f"Plivo API error ({resp.status}): {err}")
        return await resp.json()


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from helpers.db import init_db, close_db
    from helpers.ultravox_setup import ensure_ultravox_agent
    await init_db()
    await ensure_ultravox_agent()
    app.state.session = aiohttp.ClientSession()
    yield
    await app.state.session.close()
    await close_db()
    coros = [pc.disconnect() for pc in _pcs_map.values()]
    await asyncio.gather(*coros)
    _pcs_map.clear()


app = FastAPI(title="Tata Tele Call Bot", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/start")
async def start_call(request: Request) -> JSONResponse:
    data = await request.json()

    phone_number = str(data.get("phone_number", "")).strip()
    if not phone_number:
        raise HTTPException(status_code=400, detail="Missing phone_number")

    user_id        = str(data.get("user_id",       "")).strip() or None
    access_token   = str(data.get("access_token",  "")).strip() or None
    customer_name  = str(data.get("customer_name",  "")).strip()
    service_name   = str(data.get("service_name",   "")).strip()
    amount         = str(data.get("amount",         "0")).strip()
    billing_period = str(data.get("billing_period", "")).strip()
    language       = str(data.get("language",       "English")).strip()

    if not customer_name:
        raise HTTPException(status_code=400, detail="Missing customer_name")
    if not service_name:
        raise HTTPException(status_code=400, detail="Missing service_name")

    body_data = {
        "customer_name":  customer_name,
        "service_name":   service_name,
        "amount":         amount,
        "billing_period": billing_period,
        "language":       language,
    }

    server_url = os.getenv("SERVER_URL", "").rstrip("/")
    if not server_url:
        host       = request.headers.get("host", "")
        protocol   = "https" if not host.startswith(("localhost", "127.0.0.1")) else "http"
        server_url = f"{protocol}://{host}"

    body_json    = json.dumps(body_data)
    body_encoded = urllib.parse.quote(body_json)
    answer_url   = f"{server_url}/answer?body_data={body_encoded}"

    try:
        await _wait_for_call_slot()
        result    = await make_plivo_call(
            session=request.app.state.session,
            to_number=phone_number,
            from_number=os.getenv("PLIVO_PHONE_NUMBER"),
            answer_url=answer_url,
        )
        call_uuid = result.get("request_uuid") or result.get("call_uuid") or "unknown"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate call: {e}")

    from helpers.db import insert_call
    await insert_call(
        call_uuid      = call_uuid,
        batch_id       = None,
        user_id        = user_id,
        access_token   = access_token,
        customer_name  = customer_name,
        phone_number   = phone_number,
        service_name   = service_name,
        amount         = amount,
        billing_period = billing_period,
        language       = language,
    )

    _logger.info(f"Call initiated — uuid={call_uuid} to={phone_number} customer={customer_name} user_id={user_id}")
    return JSONResponse({"call_uuid": call_uuid, "status": "initiated"})


@app.api_route("/answer", methods=["GET", "POST"])
async def answer(
    request: Request,
    CallUUID:  str = Query(None),
    body_data: str = Query(None),
) -> HTMLResponse:
    if request.method == "POST":
        form = await request.form()
        if not CallUUID:
            CallUUID = form.get("CallUUID")

    parsed_body: dict = {}
    if body_data:
        try:
            parsed_body = json.loads(body_data)
        except json.JSONDecodeError:
            pass

    server_url = os.getenv("SERVER_URL", "").rstrip("/")
    if not server_url:
        host       = request.headers.get("host", "")
        protocol   = "https" if not host.startswith(("localhost", "127.0.0.1")) else "http"
        server_url = f"{protocol}://{host}"

    base_ws    = server_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    record_url = f"{server_url}/recording-ready"

    query_params = []
    if parsed_body:
        body_b64 = base64.b64encode(json.dumps(parsed_body).encode()).decode()
        query_params.append(f"body={body_b64}")
    if CallUUID:
        query_params.append(f"call_uuid={CallUUID}")

    ws_url = f"{base_ws}?{'&amp;'.join(query_params)}" if query_params else base_ws

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Record action="{record_url}" redirect="false" recordSession="true" maxLength="3600" fileFormat="mp3" callbackUrl="{record_url}" callbackMethod="POST" />
    <Stream bidirectional="true" audioTrack="inbound" contentType="audio/x-mulaw;rate=8000" keepCallAlive="true">
        {ws_url}
    </Stream>
</Response>"""

    _logger.info(f"[{CallUUID}] Answer XML → ws={ws_url}")
    return HTMLResponse(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    body:      str = Query(None),
    call_uuid: str = Query(None),
):
    await websocket.accept()
    _logger.info(f"WebSocket accepted — call_uuid={call_uuid}")

    body_data: dict = {}
    if body:
        try:
            body_data = json.loads(base64.b64decode(body).decode("utf-8"))
        except Exception as e:
            _logger.warning(f"Failed to decode body param: {e}")

    transcript: list = []

    try:
        from helpers.bot import bot
        from pipecat.runner.types import WebSocketRunnerArguments

        runner_args = WebSocketRunnerArguments(websocket=websocket, body=body_data)
        await bot(runner_args, transcript_out=transcript)
    except Exception as e:
        _logger.error(f"[{call_uuid}] Bot error: {e}")
        await websocket.close()
    finally:
        if call_uuid and transcript:
            from helpers.db import insert_transcript
            try:
                await insert_transcript(call_uuid, transcript)
                _logger.info(f"[{call_uuid}] Transcript stored — {len(transcript)} turns")
            except Exception as e:
                _logger.error(f"[{call_uuid}] Failed to store transcript: {e}")


# ── Recording webhook ──────────────────────────────────────────────────────────

@app.post("/recording-ready")
async def recording_ready(request: Request) -> JSONResponse:
    form         = await request.form()
    record_url   = form.get("RecordUrl",        "")
    call_uuid    = form.get("CallUUID",          "")
    duration     = form.get("RecordingDuration", "-1")

    _logger.info(f"[recording-ready] CallUUID={call_uuid} Duration={duration} RecordUrl={record_url}")

    if duration == "-1" or not record_url:
        return JSONResponse({"status": "start_event_ignored"})

    # Download MP3 into memory
    mp3_bytes: bytes | None = None
    try:
        async with request.app.state.session.get(record_url) as resp:
            if resp.status == 200:
                mp3_bytes = await resp.read()
                _logger.info(f"[{call_uuid}] Recording downloaded — {len(mp3_bytes)} bytes")
            else:
                _logger.error(f"[{call_uuid}] Failed to download recording: HTTP {resp.status}")
                return JSONResponse({"status": "download_failed"})
    except Exception as e:
        _logger.error(f"[{call_uuid}] Recording download error: {e}")
        return JSONResponse({"status": "download_error"})

    # Lookup call details (user_id, phone_number)
    from helpers.db import get_call
    call = await get_call(call_uuid)
    if not call:
        _logger.error(f"[{call_uuid}] Call not found in DB")
        return JSONResponse({"status": "call_not_found"})

    user_id      = call.get("user_id")
    access_token = call.get("access_token")
    phone_number = call.get("phone_number", "")

    if not access_token:
        _logger.error(f"[{call_uuid}] No access_token found — cannot push to Supabase")
        return JSONResponse({"status": "missing_access_token"})

    # Push to Supabase pipeline (non-blocking — errors are logged, not raised)
    try:
        await push_to_supabase(
            session      = request.app.state.session,
            call_uuid    = call_uuid,
            phone_number = phone_number,
            user_id      = user_id,
            access_token = access_token,
            mp3_bytes    = mp3_bytes,
        )
    except Exception as e:
        _logger.error(f"[{call_uuid}] Supabase pipeline error: {e}")

    return JSONResponse({"status": "ok"})


# ── WebRTC ─────────────────────────────────────────────────────────────────────

@app.post("/api/offer")
async def webrtc_offer(raw_request: Request, background_tasks: BackgroundTasks) -> dict:
    request = await raw_request.json()
    pc_id   = request.get("pc_id")
    body    = request.get("body") or {}

    if pc_id and pc_id in _pcs_map:
        conn = _pcs_map[pc_id]
        _logger.info(f"[webrtc] Renegotiating existing session pc_id={pc_id}")
        await conn.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        conn = SmallWebRTCConnection(_ice_servers)
        await conn.initialize(sdp=request["sdp"], type=request["type"])

        @conn.event_handler("closed")
        async def on_closed(c: SmallWebRTCConnection):
            _logger.info(f"[webrtc] Connection closed — removing pc_id={c.pc_id}")
            _pcs_map.pop(c.pc_id, None)

        from helpers.bot import webrtc_bot
        background_tasks.add_task(webrtc_bot, conn, body, raw_request.app.state.session)
        _logger.info(f"[webrtc] New WebRTC session started")

    answer = conn.get_answer()
    _pcs_map[answer["pc_id"]] = conn
    return answer


# ── Ultravox (browser WebRTC via ultravox-client SDK) ─────────────────────────

@app.post("/api/call/start")
async def ultravox_call_start(request: Request) -> JSONResponse:
    """Start an Ultravox browser call; returns joinUrl for the frontend SDK."""
    if not os.getenv("ULTRAVOX_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Ultravox is not configured. Set ULTRAVOX_API_KEY in backend .env",
        )

    agent_id = os.getenv("AGENT_ID", "").strip().strip("'\"")
    if not agent_id:
        raise HTTPException(
            status_code=503,
            detail="AGENT_ID is not configured. Restart the server to create an Ultravox agent.",
        )

    data = await request.json()
    metadata = data.get("metadata") if isinstance(data, dict) else None

    from helpers.ultravox_api import create_agent_call

    try:
        call = await create_agent_call(agent_id=agent_id, metadata=metadata)
    except Exception as e:
        _logger.error(f"[ultravox] Failed to create call: {e}")
        raise HTTPException(status_code=502, detail=f"Ultravox API error: {e}")

    return JSONResponse({"callId": call["callId"], "joinUrl": call["joinUrl"]})


@app.post("/api/outbound/call")
async def ultravox_outbound_call(request: Request) -> JSONResponse:
    """Initiate an outbound Plivo call via Ultravox."""
    if not os.getenv("ULTRAVOX_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Ultravox is not configured. Set ULTRAVOX_API_KEY in backend .env",
        )

    agent_id = os.getenv("AGENT_ID", "").strip().strip("'\"")
    if not agent_id:
        raise HTTPException(status_code=503, detail="AGENT_ID is not configured.")

    from_number = (
        os.getenv("PLIVO_FROM_NUMBER", "").strip()
        or os.getenv("PLIVO_PHONE_NUMBER", "").strip()
    )
    if not from_number:
        raise HTTPException(
            status_code=503,
            detail="PLIVO_FROM_NUMBER or PLIVO_PHONE_NUMBER is not configured.",
        )

    data = await request.json()
    phone_number = str(data.get("phone_number", "")).strip().replace(" ", "").replace("-", "")
    if not phone_number:
        raise HTTPException(status_code=400, detail="Missing phone_number")

    metadata = None
    if isinstance(data, dict):
        metadata = {
            k: str(v)
            for k, v in data.items()
            if k not in ("phone_number",) and v is not None
        } or None

    from helpers.ultravox_api import create_outbound_call

    try:
        await _wait_for_call_slot()
        call = await create_outbound_call(
            agent_id=agent_id,
            to_number=phone_number,
            from_number=from_number,
            metadata=metadata,
        )
    except Exception as e:
        _logger.error(f"[ultravox] Outbound call failed: {e}")
        raise HTTPException(status_code=502, detail=f"Ultravox API error: {e}")

    _logger.info(f"[ultravox] Outbound call initiated callId={call['callId']} to={phone_number}")
    return JSONResponse({
        "callId": call["callId"],
        "status": "initiated",
        "to_number": phone_number,
        "message": f"Calling {phone_number}. The AI will connect shortly.",
    })


# ── LLM Test ──────────────────────────────────────────────────────────────────

@app.post("/test-llm")
async def test_llm(request: Request) -> JSONResponse:
    data   = await request.json()
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    import time
    start = time.time()
    try:
        async with request.app.state.session.post(
            os.getenv("LOCAL_LLM_URL", "http://164.52.198.104:8049/v1") + "/chat/completions",
            json={
                "model":    os.getenv("LOCAL_LLM_MODEL", "google/gemma-4-26B-A4B-it"),
                "messages": [{"role": "user", "content": prompt}],
            },
            headers={"Authorization": "Bearer local"},
        ) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise HTTPException(status_code=502, detail=f"LLM error: {err}")
            result = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM unreachable: {e}")

    elapsed = round(time.time() - start, 2)
    text    = result["choices"][0]["message"]["content"]
    _logger.info(f"[test-llm] prompt_len={len(prompt)} elapsed={elapsed}s")
    return JSONResponse({"response": text, "elapsed_seconds": elapsed})


# ── Logs ───────────────────────────────────────────────────────────────────────

@app.get("/logs")
async def get_logs() -> JSONResponse:
    from helpers.db import get_calls
    return JSONResponse(await get_calls())


@app.get("/logs/{call_uuid}")
async def get_log_detail(call_uuid: str) -> JSONResponse:
    from helpers.db import get_call, get_transcript
    call = await get_call(call_uuid)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    transcript = await get_transcript(call_uuid)
    return JSONResponse({
        "call":       call,
        "transcript": transcript,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)