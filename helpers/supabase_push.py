import os
import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def _sb_json_headers(access_token: str) -> dict:
    return {
        "apikey":        SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _sb_storage_headers(access_token: str) -> dict:
    return {
        "apikey":        SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type":  "audio/mpeg",
    }


async def push_to_supabase(
    session: aiohttp.ClientSession,
    call_uuid: str,
    phone_number: str,
    user_id: str | None,
    access_token: str,
    mp3_bytes: bytes,
    filename: str | None = None,
) -> None:
    if not SUPABASE_URL or not access_token:
        logger.error(f"[{call_uuid}] Missing SUPABASE_URL or access_token — skipping push")
        return

    display_name = filename or f"Call to {phone_number}"

    # 1. Insert call_analyses row
    payload: dict = {
        "filename":      display_name,
        "mobile_number": phone_number,
        "status":        "uploading",
    }
    if user_id:
        payload["user_id"] = user_id

    analysis_id = None
    async with session.post(
        f"{SUPABASE_URL}/rest/v1/call_analyses",
        json=payload,
        headers=_sb_json_headers(access_token),
    ) as resp:
        if resp.status in (200, 201):
            rows = await resp.json()
            analysis_id = rows[0]["id"] if rows else None
            logger.info(f"[{call_uuid}] call_analyses row created — id={analysis_id}")
        else:
            err = await resp.text()
            logger.error(f"[{call_uuid}] Failed to insert call_analyses: {resp.status} {err}")
            return

    if not analysis_id:
        logger.error(f"[{call_uuid}] No analysis_id returned from Supabase insert")
        return

    # 2. Upload MP3 to Supabase storage
    file_path = f"{analysis_id}/recording.mp3"
    async with session.post(
        f"{SUPABASE_URL}/storage/v1/object/call-recordings/{file_path}",
        data=mp3_bytes,
        headers=_sb_storage_headers(access_token),
    ) as resp:
        if resp.status not in (200, 201):
            err = await resp.text()
            logger.error(f"[{call_uuid}] Storage upload failed: {resp.status} {err}")
            async with session.patch(
                f"{SUPABASE_URL}/rest/v1/call_analyses?id=eq.{analysis_id}",
                json={"status": "failed"},
                headers=_sb_json_headers(access_token),
            ) as _:
                pass
            return
        logger.info(f"[{call_uuid}] MP3 uploaded → {file_path}")

    # 3. Update row: file_path + status transcribing
    async with session.patch(
        f"{SUPABASE_URL}/rest/v1/call_analyses?id=eq.{analysis_id}",
        json={"file_path": file_path, "status": "transcribing"},
        headers=_sb_json_headers(access_token),
    ) as resp:
        if resp.status not in (200, 204):
            err = await resp.text()
            logger.error(f"[{call_uuid}] Failed to update call_analyses status: {resp.status} {err}")

    # 4. Invoke process-call edge function
    async with session.post(
        f"{SUPABASE_URL}/functions/v1/process-call",
        json={"analysisId": analysis_id, "filePath": file_path},
        headers=_sb_json_headers(access_token),
    ) as resp:
        if resp.status not in (200, 201):
            err = await resp.text()
            logger.error(f"[{call_uuid}] Edge function failed: {resp.status} {err}")
        else:
            logger.info(f"[{call_uuid}] process-call edge function invoked ✓")