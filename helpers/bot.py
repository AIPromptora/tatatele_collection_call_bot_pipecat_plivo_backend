import asyncio
import os
import wave
import struct
from datetime import datetime
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


# ── Audio capture processor ────────────────────────────────────────────────────

class _AudioCapture(FrameProcessor):
    """Silently buffers audio frames passing through — zero impact on pipeline."""

    def __init__(self, capture_input: bool = False):
        super().__init__()
        self._capture_input = capture_input  # True = user mic, False = bot TTS
        self.chunks: list[bytes] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self._capture_input and isinstance(frame, InputAudioRawFrame):
            self.chunks.append(frame.audio)
        elif not self._capture_input and isinstance(frame, AudioRawFrame):
            self.chunks.append(frame.audio)
        await self.push_frame(frame, direction)

    def get_bytes(self) -> bytes:
        return b"".join(self.chunks)


def _save_recording(
    user_bytes: bytes,
    bot_bytes: bytes,
    sample_rate: int = 8000,
    session_id: str = "",
) -> str | None:
    """Mix user + bot PCM and save as MP3 (falls back to WAV if pydub/ffmpeg unavailable)."""
    if not user_bytes and not bot_bytes:
        logger.warning("[recording] No audio captured — skipping save")
        return None

    recordings_dir = os.path.join(os.path.dirname(__file__), "..", "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = session_id[-8:] if session_id else "webrtc"

    try:
        from pydub import AudioSegment

        def _to_segment(raw: bytes) -> AudioSegment:
            return AudioSegment(raw, sample_width=2, frame_rate=sample_rate, channels=1)

        user_seg = _to_segment(user_bytes) if user_bytes else AudioSegment.silent(0, frame_rate=sample_rate)
        bot_seg  = _to_segment(bot_bytes)  if bot_bytes  else AudioSegment.silent(0, frame_rate=sample_rate)

        # Pad shorter track so overlay works correctly
        diff = len(user_seg) - len(bot_seg)
        if diff > 0:
            bot_seg  = bot_seg  + AudioSegment.silent(diff,  frame_rate=sample_rate)
        elif diff < 0:
            user_seg = user_seg + AudioSegment.silent(-diff, frame_rate=sample_rate)

        mixed    = user_seg.overlay(bot_seg)
        filepath = os.path.join(recordings_dir, f"webrtc_{ts}_{sid}.mp3")
        mixed.export(filepath, format="mp3")
        logger.info(f"[recording] Saved MP3 → {filepath} ({len(mixed)/1000:.1f}s)")
        return filepath

    except Exception as e:
        logger.warning(f"[recording] MP3 save failed ({e}) — falling back to WAV")
        try:
            # WAV fallback — no ffmpeg needed
            all_bytes = user_bytes + bot_bytes  # sequential, not mixed
            filepath  = os.path.join(recordings_dir, f"webrtc_{ts}_{sid}.wav")
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(all_bytes)
            logger.info(f"[recording] Saved WAV → {filepath}")
            return filepath
        except Exception as e2:
            logger.error(f"[recording] WAV save also failed: {e2}")
            return None


# ── Transcription logger ───────────────────────────────────────────────────────

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug(f"STT: [{frame.text}] | lang: {frame.language}")
        await self.push_frame(frame, direction)


# ── Core bot pipeline ──────────────────────────────────────────────────────────

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
        "Telugu":   "te-IN",
    }

    tts = SarvamTTSService(
        api_key=os.getenv("SARVAM_API_KEY", ""),
        voice_id="shubh",
        model="bulbul:v3",
        params=SarvamTTSService.InputParams(
            pace=1.0,
            temperature=0.8
        )
    )

    logger.info(f"TTS: Sarvam bulbul:v3 shubh ({language} → {_lang_code_map.get(language, 'en-IN')})")

    # ── Greeting ──────────────────────────────────────────────────────────────

    greetings = {
        "English": (
            f"Hi, this is {agent_name} from Tata Tele services regarding a pending payment for {service_name}. "
            f"Would you like to continue in English, Hindi, Marathi, Gujarati, Bengali, or Telugu?"
        ),
        "Hindi": (
            f"नमस्ते, मैं {agent_name} बोल रहा हूँ Tata Tele services से, आपके {service_name} के pending payment के बारे में। "
            f"क्या आप Hindi, English, Marathi, Gujarati, Bengali, या Telugu में बात करना चाहेंगे?"
        ),
        "Marathi": (
            f"नमस्कार, मी {agent_name} बोलतोय Tata Tele services कडून, आपल्या {service_name} च्या pending payment बद्दल। "
            f"आपण Marathi, English, Hindi, Gujarati, Bengali, किंवा Telugu मध्ये बोलू का?"
        ),
        "Gujarati": (
            f"નમસ્તે, હું {agent_name} બોલું છું Tata Tele services તરફથી, તમારા {service_name} ના pending payment વિશે। "
            f"તમે Gujarati, English, Hindi, Marathi, Bengali, અથવા Telugu માં વાત કરશો?"
        ),
        "Bengali": (
            f"নমস্কার, আমি {agent_name} বলছি Tata Tele services থেকে, আপনার {service_name}-এর pending payment সম্পর্কে। "
            f"আপনি Bengali, English, Hindi, Marathi, Gujarati, নাকি Telugu তে কথা বলবেন?"
        ),
        "Telugu": (
            f"నమస్కారం, నేను {agent_name} మాట్లాడుతున్నాను Tata Tele services నుండి, మీ {service_name} కి సంబంధించిన pending payment గురించి। "
            f"మీరు Telugu, English, Hindi, Marathi, Gujarati, లేదా Bengali లో మాట్లాడతారా?"
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
        → Apologize — it may not have reflected in records yet. Thank them for paying and close the call.

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

        LANGUAGE RULES:
        - You can only understand and speak in English, Hindi, Marathi, Gujarati, Bengali, and Telugu.
        - ALWAYS respond in the same language the customer is speaking.
        - If they switch language mid-call, you switch too. So strictly understand which language user is speaking and reply in that language. If you are not clear about anything, then take the input from the user and continue the conversation.

        English:
        - Plain, warm, conversational. Not scripted.

        Hindi:
        - Warm Hinglish — mix English words naturally.
        - Use Devanagari script only (no Roman transliteration — it degrades TTS).
        - Every Hindi sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.

        Marathi:
        - Warm, conversational Marathi — mix common English words naturally (e.g. payment, invoice, amount).
        - Use Devanagari script only (no Roman transliteration).
        - Every Marathi sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.
        - Use polite forms (आपण / तुम्ही) consistently throughout the call.

        Gujarati:
        - Warm, conversational Gujarati — mix common English words naturally (e.g. payment, invoice, amount).
        - Use Gujarati script only (no Roman transliteration).
        - Every Gujarati sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.
        - Use polite forms (આપ / તમે) consistently throughout the call.

        Bengali:
        - Warm, conversational Bengali — mix common English words naturally (e.g. payment, invoice, amount).
        - Use Bengali script only (no Roman transliteration).
        - Every Bengali sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.
        - Use polite forms (আপনি) consistently throughout the call.

        Telugu:
        - Warm, natural Tenglish — mix English words freely and naturally (e.g. payment, invoice, amount, bill, due date).
        - Use Telugu script only for Telugu words (no Roman transliteration — it degrades TTS).
        - Every Telugu sentence must end with । (danda), NEVER a period (.).
        - Keep sentences under 20 words.
        - Use polite forms (మీరు) consistently throughout the call.
        - Sound natural and conversational — like how Telugu people actually speak with English mixed in.

        - If customer mixes any of these languages freely, respond in the same casual mixed style.

        CLOSING THE CALL:
        Once you have the payment timeline or resolved the concern, close warmly:
        English:  "Thank you for your time. Please feel free to reach out if you need anything. Have a great day!"
        Hindi:    "आपके time के लिए thank you। कोई भी सवाल हो तो हमें call करें। Have a great day!"
        Marathi:  "आपल्या वेळासाठी धन्यवाद। काही प्रश्न असल्यास आम्हाला call करा। Have a great day!"
        Gujarati: "તમારા સમય બદલ આભાર। કોઈ પ્રશ્ન હોય તો અમને call કરો। Have a great day!"
        Bengali:  "আপনার সময়ের জন্য ধন্যবাদ। কোনো প্রশ্ন থাকলে আমাদের call করুন। Have a great day!"
        Telugu:   "మీ time కి చాలా thanks। ఏదైనా అవసరమైతే మాకు call చేయండి। Have a great day!"
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

    # Build pipeline — insert capture processors when recording is enabled
    pipeline_stages = [transport.input()]
    if user_capture is not None:
        pipeline_stages.append(user_capture)
    pipeline_stages += [
        stt,
        TranscriptionLogger(),
        user_aggregator,
        llm,
        tts,
    ]
    if bot_capture is not None:
        pipeline_stages.append(bot_capture)
    pipeline_stages += [
        transport.output(),
        assistant_aggregator,
    ]

    pipeline = Pipeline(pipeline_stages)

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

async def webrtc_bot(webrtc_connection: SmallWebRTCConnection, body: dict | None = None):
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

    # Session ended — save recording
    _save_recording(
        user_bytes=user_capture.get_bytes(),
        bot_bytes=bot_capture.get_bytes(),
        sample_rate=8000,
        session_id=getattr(webrtc_connection, "pc_id", ""),
    )