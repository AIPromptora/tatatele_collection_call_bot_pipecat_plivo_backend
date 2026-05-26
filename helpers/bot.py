import asyncio
import os
import uuid
from io import BytesIO
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TTSSpeakFrame,
    TranscriptionFrame,
    AudioRawFrame,
    InputAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.sarvam.tts import SarvamTTSService
load_dotenv(override=True)


# ── Audio capture ──────────────────────────────────────────────────────────────

class _AudioCapture(FrameProcessor):
    """
    Buffers PCM audio frames without affecting the pipeline.
    Reads sample_rate and num_channels directly from each frame so the
    saved recording always matches the actual audio rate — fixes slow-motion
    playback caused by a sample-rate mismatch between capture and pydub.
    """

    def __init__(self, capture_input: bool = False):
        super().__init__()
        self._capture_input = capture_input  # True = user mic, False = bot TTS
        self.chunks: list[bytes] = []
        self.sample_rate: int = 8000
        self.num_channels: int = 1

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._capture_input and isinstance(frame, InputAudioRawFrame):
            self.chunks.append(frame.audio)
            self.sample_rate  = frame.sample_rate
            self.num_channels = frame.num_channels
        elif (
            not self._capture_input
            and isinstance(frame, AudioRawFrame)
            and not isinstance(frame, InputAudioRawFrame)
        ):
            self.chunks.append(frame.audio)
            self.sample_rate  = frame.sample_rate
            self.num_channels = frame.num_channels
        await self.push_frame(frame, direction)

    def get_bytes(self) -> bytes:
        return b"".join(self.chunks)


def _mix_to_mp3_bytes(
    user_cap: "_AudioCapture",
    bot_cap: "_AudioCapture",
) -> bytes | None:
    """
    Mix user-mic and bot-TTS PCM into a single MP3 (in memory).
    Both tracks are resampled to the higher of the two sample rates and
    converted to mono before overlaying — this ensures correct playback speed
    regardless of what rate the WebRTC transport delivers.
    """
    user_bytes = user_cap.get_bytes()
    bot_bytes  = bot_cap.get_bytes()

    if not user_bytes and not bot_bytes:
        logger.warning("[recording] No audio captured — nothing to mix")
        return None

    try:
        from pydub import AudioSegment

        def _to_seg(raw: bytes, cap: "_AudioCapture") -> AudioSegment:
            return AudioSegment(
                raw,
                sample_width=2,          # 16-bit PCM (pipecat standard)
                frame_rate=cap.sample_rate,
                channels=cap.num_channels,
            )

        user_seg = (
            _to_seg(user_bytes, user_cap)
            if user_bytes
            else AudioSegment.silent(0, frame_rate=user_cap.sample_rate)
        )
        bot_seg = (
            _to_seg(bot_bytes, bot_cap)
            if bot_bytes
            else AudioSegment.silent(0, frame_rate=bot_cap.sample_rate)
        )

        # Normalise: mono + same sample rate so overlay works correctly
        target_rate = max(user_cap.sample_rate, bot_cap.sample_rate)
        user_seg = user_seg.set_channels(1).set_frame_rate(target_rate)
        bot_seg  = bot_seg.set_channels(1).set_frame_rate(target_rate)

        # Pad the shorter track
        diff = len(user_seg) - len(bot_seg)
        if diff > 0:
            bot_seg  = bot_seg  + AudioSegment.silent(diff,  frame_rate=target_rate)
        elif diff < 0:
            user_seg = user_seg + AudioSegment.silent(-diff, frame_rate=target_rate)

        mixed = user_seg.overlay(bot_seg)
        buf   = BytesIO()
        mixed.export(buf, format="mp3", bitrate="64k")
        mp3_bytes = buf.getvalue()
        logger.info(
            f"[recording] Mixed MP3 ready — "
            f"{len(mixed)/1000:.1f}s @ {target_rate}Hz, {len(mp3_bytes)} bytes"
        )
        return mp3_bytes

    except Exception as e:
        logger.error(f"[recording] MP3 encoding failed: {e}")
        return None


# ── Transcription logger ───────────────────────────────────────────────────────

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug(f"STT: [{frame.text}] | lang: {frame.language}")
        await self.push_frame(frame, direction)


# ── Core pipeline ──────────────────────────────────────────────────────────────

async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    body: Optional[dict] = None,
    transcript_out: Optional[list] = None,
    user_capture: Optional[_AudioCapture] = None,
    bot_capture: Optional[_AudioCapture] = None,
):
    llm = OpenAILLMService(
        api_key="local",
        base_url=os.getenv("LOCAL_LLM_URL", "http://164.52.198.104:8049/v1"),
        model="google/gemma-4-26B-A4B-it",
    )

    stt = SarvamSTTService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        settings=SarvamSTTService.Settings(
            model="saarika:v2.5",
            vad_signals=True,
        ),
    )

    b = body or {}
    customer_name  = b.get("customer_name",  "Sir/Madam")
    amount         = b.get("amount",         "25,000")
    billing_period = b.get("billing_period", "March 2026")
    service_name   = b.get("service_name",   "Monthly Telecom Services (Voice, Internet & Business Connectivity)")
    language       = b.get("language",       "English")

    agent_name = "Arjun"

    _lang_code_map = {
        "English":  "en-IN",
        "Hindi":    "hi-IN",
        "Marathi":  "mr-IN",
        "Gujarati": "gu-IN",
        "Bengali":  "bn-IN",
    }

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        voice_id=b.get("voice_id", "shubh"),
        model="bulbul:v3",
        params=SarvamTTSService.InputParams(
            pace=0.9,
            temperature=0.8
        )
    )

    logger.info(f"TTS: Sarvam bulbul:v3 shubh ({language} → {_lang_code_map.get(language, 'en-IN')})")

    # ── Greeting ──────────────────────────────────────────────────────────────

    greetings = {
        "English": (
            f"Hi, this is {agent_name} from Tata Tele services regarding a pending payment for {service_name}. "
            f"Would you like to continue in English or Hindi?"
        ),
        "Hindi": (
            f"नमस्ते, मैं {agent_name} बोल रहा हूँ Tata Tele services से, आपके {service_name} के pending payment के बारे में। "
            f"क्या आप हिंदी में बात करना चाहेंगे या English में?"
        ),
    }

    greeting_text = greetings.get(language, greetings["English"])

    # ── System prompt ─────────────────────────────────────────────────────────

    system_content = f"""
        You are {agent_name}, a professional but warm collection agent calling on behalf of a telecom company.

        CALL PURPOSE:
        You are following up on a pending payment of INR {amount} from {customer_name} for {service_name}
        for the billing period of {billing_period}. The invoice has already been sent to the customer's
        registered email. Your goal is to get a payment commitment or understand the reason for delay.

        CALL DETAILS:
        - Customer Name: {customer_name}
        - Amount Due: INR {amount}
        - Service: {service_name}
        - Billing Period: {billing_period}

        YOUR GOAL:
        1. Confirm the payment status with the customer.
        2. If pending → get an expected payment date.
        3. If any issue (dispute, not received invoice, approval pending) → acknowledge and offer next steps.
        4. Always close the call politely.

        HOW TO HANDLE COMMON SITUATIONS:

        Payment is pending / they know about it:
        → Thank them for confirming. Ask for an expected payment timeline. Note it and close warmly.

        Invoice not received:
        → Apologize, ask them to confirm their email ID, assure them it will be resent immediately.

        Payment is under approval / with finance team:
        → Acknowledge. Ask for an approximate approval or payment release date. Offer to provide any supporting documents.

        Customer is irritated about repeated calls:
        → Sincerely apologize. Explain you just want to avoid any inconvenience. Ask for payment timeline once and close.

        Customer refuses to pay now / says stop calling:
        → Stay calm and respectful. Don't argue. Explain you just need to update internal records. Ask for an approximate timeline.

        Customer raises a dispute (wrong amount, service issue, missing document):
        → Apologize for the inconvenience. Ask what the concern is (amount, service, terms, document). Assure them the support team will follow up and resolve quickly.

        Customer is too busy to talk:
        → Apologize for the interruption. Ask for a convenient callback time.

        Customer asks something you cannot answer (contract terms, technical details, internal details):
        → Acknowledge. Tell them the support team will contact them directly to clarify.

        Customer is very angry or uses strong language:
        → Apologize sincerely. Do not escalate. Tell them you will immediately update your internal team and have the right support person contact them. Do not ask or repeat for payment timeline in this case.

        Customer says they already paid:
        → Apologize — it may not have reflected in records yet, and ask them the details of the payment. Thank them for paying and close the call.

        IMPORTANT CONVERSATION RULES:
        - Listen carefully. The customer may not respond exactly as expected — understand the intent and respond appropriately.
        - Keep responses SHORT — max 2–3 sentences. This is a voice call.
        - Never repeat what the customer just said back to them.
        - Never be pushy or aggressive.
        - Always sound like a real human — warm, professional, never robotic or scripted.
        - Plain text only. No emojis, no symbols, no markdown, no bullet points.
        - Never say "INR" — say "rupees" instead. E.g. "twenty five thousand rupees".
        - Be confident, warm and compassionate.

        Instructions:
        You are a professional customer service assistant.
        - Do not engage in personal conversations.
        - Politely decline personal questions.
        - Always redirect to the business goal (support or payment).
        - Maintain a respectful and calm tone.
        - Always have meaningful, context-aware responses. We should respond in the most meaningful way based on the user response.
        - Use meaningful validations whenever required. For example, if user says they are busy, ask when would be a good time to call back. If they say they have a dispute, ask what the dispute is about. If they say they will pay soon, ask when exactly they will pay.


        LANGUAGE RULES:
        - You can only understand and speak in English and Hindi.
        - ALWAYS respond in the same language the customer is speaking.
        - If they switch language mid-call, you switch too. So strictly understand which language user is speaking and reply in that language. If you are not clear about anything, then take the input from the user and continue the conversation.

        English:
        - Plain, warm, conversational. Not scripted.

        Hindi:
        - Warm Hinglish — mix English words naturally.
        - Use Devanagari script only (no Roman transliteration — it degrades TTS).
        - Every Hindi sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.

        - If customer mixes English and Hindi freely, respond in the same casual mixed style.

        CLOSING THE CALL:
        Once you have the payment timeline or resolved the concern, close warmly:
        English: "Thank you for your time. Please feel free to reach out if you need anything. Have a great day!"
        Hindi:   "आपके time के लिए thank you। कोई भी सवाल हो तो हमें call करें। Have a great day!"
        """

    logger.info(f"SYSTEM PROMPT ({'─'*60})\n{system_content}\n{'─'*70}")

    messages = [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": "begin"},
        {"role": "assistant", "content": greeting_text},
    ]

    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_stop_timeout=0.7,
        ),
    )

    # ── Pipeline — captures inserted only when provided ────────────────────────

    stages = [transport.input()]
    if user_capture:
        stages.append(user_capture)          # captures user mic PCM here
    stages += [stt, TranscriptionLogger(), user_aggregator, llm, tts]
    if bot_capture:
        stages.append(bot_capture)           # captures bot TTS PCM here
    stages += [transport.output(), assistant_aggregator]

    pipeline = Pipeline(stages)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Call started")
        await task.queue_frames([TTSSpeakFrame(text=greeting_text)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    try:
        await runner.run(task)
    finally:
        if transcript_out is not None and not transcript_out:
            for msg in context._messages[2:]:
                role    = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                    transcript_out.append({"role": role, "text": content})
            logger.info(f"Transcript snapshot: {len(transcript_out)} turns")


# ── Telephony entry point ─────────────────────────────────────────────────────

async def bot(runner_args: RunnerArguments, transcript_out: Optional[list] = None):
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Transport: {transport_type}")

    body = runner_args.body or {}

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
        params=PlivoFrameSerializer.InputParams(
            plivo_sample_rate=8000,
            auto_hang_up=True,
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, body=body, transcript_out=transcript_out)


# ── WebRTC entry point ────────────────────────────────────────────────────────

async def webrtc_bot(
    webrtc_connection: SmallWebRTCConnection,
    body: dict | None = None,
    aiohttp_session=None,
):
    user_capture = _AudioCapture(capture_input=True)
    bot_capture  = _AudioCapture(capture_input=False)

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    await run_bot(
        transport,
        handle_sigint=False,
        body=body,
        user_capture=user_capture,
        bot_capture=bot_capture,
    )

    # ── Post-session: mix audio and push to Supabase analysis pipeline ─────────

    b            = body or {}
    access_token = b.get("access_token", "").strip()
    user_id      = b.get("user_id", "").strip() or None
    customer_name = b.get("customer_name", "Browser User")
    service_name  = b.get("service_name", "Voice Test")

    if not access_token:
        logger.warning("[webrtc] No access_token in body — skipping Supabase push")
        return

    mp3_bytes = _mix_to_mp3_bytes(user_capture, bot_capture)
    if not mp3_bytes:
        logger.warning("[webrtc] No audio to push — skipping Supabase push")
        return

    session_id = getattr(webrtc_connection, "pc_id", None) or str(uuid.uuid4())
    filename   = f"Voice Test — {customer_name} ({service_name})"

    from helpers.supabase_push import push_to_supabase
    import aiohttp as _aiohttp

    close_session = False
    sess = aiohttp_session
    if sess is None:
        sess = _aiohttp.ClientSession()
        close_session = True

    try:
        await push_to_supabase(
            session      = sess,
            call_uuid    = session_id,
            phone_number = "WebRTC Browser Session",
            user_id      = user_id,
            access_token = access_token,
            mp3_bytes    = mp3_bytes,
            filename     = filename,
        )
    except Exception as e:
        logger.error(f"[webrtc] Supabase push failed: {e}")
    finally:
        if close_session:
            await sess.close()