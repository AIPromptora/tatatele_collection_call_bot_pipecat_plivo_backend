import os

import httpx

from helpers.ultravox_prompts import build_greeting

ULTRAVOX_BASE_URL = "https://api.ultravox.ai/api"


def _headers() -> dict:
    api_key = os.getenv("ULTRAVOX_API_KEY")
    if not api_key or api_key == "your_ultravox_api_key_here":
        raise ValueError("ULTRAVOX_API_KEY is not set in .env")
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }


def _call_template(
    system_prompt: str,
    voice: str,
    model: str,
    max_duration: str,
) -> dict:
    return {
        "systemPrompt": system_prompt,
        "model": model,
        "voice": voice,
        "maxDuration": max_duration,
        "firstSpeakerSettings": {"agent": {}},
        "selectedTools": [],
        "temperature": 0.3,
        "inactivityMessages": [
            {
                "duration": "5s",
                "message": "Are you still there? Please let me know if you need assistance with Tata Tele Business Services",
                "endBehavior": "END_BEHAVIOR_UNSPECIFIED",
            }
        ],
    }


async def create_agent(
    name: str,
    system_prompt: str,
    voice: str,
    model: str,
    max_duration: str,
) -> dict:
    payload = {
        "name": name,
        "callTemplate": _call_template(system_prompt, voice, model, max_duration),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{ULTRAVOX_BASE_URL}/agents",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def create_agent_call(
    agent_id: str,
    metadata: dict | None = None,
) -> dict:
    language = os.getenv("CALL_LANGUAGE", "English")
    payload: dict = {
        "medium": {"webRtc": {}},
        "firstSpeakerSettings": {
            "agent": {"text": build_greeting(language)},
        },
    }
    if metadata:
        payload["metadata"] = metadata

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{ULTRAVOX_BASE_URL}/agents/{agent_id}/calls",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def patch_agent(
    agent_id: str,
    system_prompt: str,
    voice: str,
    model: str,
    max_duration: str,
) -> dict:
    payload = {"callTemplate": _call_template(system_prompt, voice, model, max_duration)}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.patch(
            f"{ULTRAVOX_BASE_URL}/agents/{agent_id}",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def create_outbound_call(
    agent_id: str,
    to_number: str,
    from_number: str,
    metadata: dict | None = None,
) -> dict:
    """Outbound phone call via Ultravox + Plivo."""
    language = os.getenv("CALL_LANGUAGE", "English")
    payload: dict = {
        "medium": {
            "plivo": {
                "outgoing": {
                    "to": to_number,
                    "from": from_number,
                }
            }
        },
        "firstSpeakerSettings": {
            "agent": {
                "text": build_greeting(language),
                "delay": "1s",
            },
        },
    }
    if metadata:
        payload["metadata"] = metadata

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{ULTRAVOX_BASE_URL}/agents/{agent_id}/calls",
            headers=_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()
