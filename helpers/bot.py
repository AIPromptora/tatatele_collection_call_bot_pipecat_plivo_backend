import asyncio
import os
from typing import Optional
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TTSSpeakFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.sarvam.tts import SarvamTTSService
load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            logger.debug(f"STT: [{frame.text}] | lang: {frame.language}")
        await self.push_frame(frame, direction)


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    body: Optional[dict] = None,
    transcript_out: Optional[list] = None,
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
        voice_id="shubh",
        model="bulbul:v3",
        params=SarvamTTSService.InputParams(
            pace=1.0,
            temperature=0.8
        )
    )
  
    logger.info(f"TTS: Sarvam bulbul:v3 aditya ({language} → {_lang_code_map.get(language, 'en-IN')})") 

    # ── Greeting ──────────────────────────────────────────────────────────────

    greetings = {
        "English": (
            f"Hi, this is {agent_name} from Tata Tele services regarding a pending payment for {service_name}. "
            f"Would you like to continue in English or Hindi, Marathi, Gujarati, Bengali?"
        ),
        "Hindi": (
            f"नमस्ते, मैं {agent_name} बोल रहा हूँ Tata Tele services से, आपके {service_name} के pending payment के बारे में। "
            f"क्या आप हिंदी में बात करना चाहेंगे या English, Marathi, Gujarati, Bengali?"
        ),
        "Marathi": (
            f"नमस्कार, मी {agent_name} बोलतोय Tata Tele services कडून, आपल्या {service_name} च्या pending payment बद्दल। "
            f"आपण मराठीत बोलू का, की English, Hindi, Gujarati, Bengali?"
        ),
        "Gujarati": (
            f"નમસ્તે, હું {agent_name} બોલું છું Tata Tele services તરફથી, તમારા {service_name} ના pending payment વિશે। "
            f"તમે ગુજરાતી માં વાત કરશો કે English, Hindi, Marathi, Bengali?"
        ),
        "Bengali": (
            f"নমস্কার, আমি {agent_name} বলছি Tata Tele services থেকে, আপনার {service_name}-এর pending payment সম্পর্কে। "
            f"আপনি কি বাংলায় কথা বলবেন, না English, Hindi, Marathi, Gujarati?"
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
        - You can only understand and speak in English, Hindi, Marathi, Gujarati, and Bengali.
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

        - If customer mixes any of these languages freely, respond in the same casual mixed style.

        CLOSING THE CALL:
        Once you have the payment timeline or resolved the concern, close warmly:
        English:  "Thank you for your time. Please feel free to reach out if you need anything. Have a great day!"
        Hindi:    "आपके time के लिए thank you। कोई भी सवाल हो तो हमें call करें। Have a great day!"
        Marathi:  "आपल्या वेळासाठी धन्यवाद। काही प्रश्न असल्यास आम्हाला call करा। Have a great day!"
        Gujarati: "તમારા સમય બદલ આભાર। કોઈ પ્રશ્ન હોય તો અમને call કરો। Have a great day!"
        Bengali:  "আপনার সময়ের জন্য ধন্যবাদ। কোনো প্রশ্ন থাকলে আমাদের call করুন। Have a great day!"
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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionLogger(),
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

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


# ── Entry point ───────────────────────────────────────────────────────────────


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