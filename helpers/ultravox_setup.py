import os
import logging

from dotenv import set_key

from helpers.ultravox_api import create_agent, patch_agent
from helpers.ultravox_prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


async def ensure_ultravox_agent() -> None:
    if not os.getenv("ULTRAVOX_API_KEY", "").strip():
        logger.warning("ULTRAVOX_API_KEY not set — /api/call/start will be unavailable")
        return

    agent_id = os.getenv("AGENT_ID", "").strip().strip("'\"")
    voice = os.getenv("ULTRAVOX_VOICE", os.getenv("VOICE", "Mark"))
    model = os.getenv("ULTRAVOX_MODEL", os.getenv("MODEL", "fixie-ai/ultravox-70B"))
    max_duration = os.getenv("ULTRAVOX_MAX_DURATION", os.getenv("MAX_DURATION", "3600s"))

    if agent_id:
        try:
            await patch_agent(agent_id, SYSTEM_PROMPT, voice, model, max_duration)
            logger.info(f"Ultravox agent {agent_id} config synced")
        except Exception as e:
            logger.warning(f"Could not sync Ultravox agent: {e}")
        return

    logger.info("No AGENT_ID — creating Ultravox agent...")
    agent = await create_agent(
        name=os.getenv("ULTRAVOX_AGENT_NAME", "TataTeleVoiceBot"),
        system_prompt=SYSTEM_PROMPT,
        voice=voice,
        model=model,
        max_duration=max_duration,
    )
    new_id = agent["agentId"]
    set_key(ENV_PATH, "AGENT_ID", new_id)
    os.environ["AGENT_ID"] = new_id
    logger.info(f"Ultravox agent created: {new_id}")
