import json
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import PlainTextResponse
import httpx

from src import config, logging
from src.ca_client import get_ca_client
from src.gemini_client import get_gemini_client
from src.utils import (
    build_twiml_message,
    validate_phone_number,
    validate_twilio_signature,
)

# Initialize logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients and log configuration on startup."""
    logger.info("=" * 60)
    logger.info("Twilio-GCA-Gemini Integration Service Starting...")
    logger.info("=" * 60)
    logger.info(f"GCA Project: {config.GCA_PROJECT_ID}")
    logger.info(f"GCA Agent: {config.GCA_AGENT_ID}")
    logger.info(f"GCA Location: {config.GCA_LOCATION}")
    logger.info(f"Gemini Model: {config.GEMINI_MODEL}")
    logger.info(f"Max Image Size: {config.MAX_IMAGE_SIZE_MB}MB")
    logger.info("=" * 60)

    # Initialize clients (singletons will be created on first use)
    try:
        get_ca_client()
        logger.info("✓ Conversational Agent client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize CA client: {e}")

    try:
        get_gemini_client()
        logger.info("✓ Gemini client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Gemini client: {e}")

    logger.info("=" * 60)
    logger.info("Service ready to accept requests")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown (if needed)


# Initialize FastAPI app
app = FastAPI(
    title="Twilio-GCA-Gemini Integration",
    description="Integration service for Twilio, Google Conversational Agent, and Gemini AI",
    version="0.1.0",
    lifespan=lifespan,
)


# ===========================
# Helper Functions
# ===========================


def extract_metadata(form_or_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from various possible field names in request body.

    Args:
        form_or_json: Request body as dictionary

    Returns:
        Parsed metadata dictionary, or empty dict if not found
    """
    for key in ("metadata", "Metadata", "dialogflow_metadata"):
        if key in form_or_json:
            try:
                value = form_or_json[key]
                if isinstance(value, str):
                    meta = json.loads(value)
                    if isinstance(meta, dict):
                        return meta
                elif isinstance(value, dict):
                    return value
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse metadata from key '{key}': {e}")
                continue
    return {}


async def download_media(url: str, max_size_bytes: int) -> bytes:
    """
    Download media from Twilio with size limit.

    Args:
        url: Media URL from Twilio
        max_size_bytes: Maximum allowed size in bytes

    Returns:
        Downloaded media bytes

    Raises:
        ValueError: If media exceeds size limit
        httpx.HTTPStatusError: If download fails
    """
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream(
            "GET",
            url,
            auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
        ) as r:
            r.raise_for_status()

            # Read with size limit
            media_bytes = b""
            async for chunk in r.aiter_bytes():
                media_bytes += chunk
                if len(media_bytes) > max_size_bytes:
                    raise ValueError(f"Media too large (>{config.MAX_IMAGE_SIZE_MB}MB)")

            return media_bytes


async def process_media_attachments(form: Dict[str, Any], num_media: int) -> list[str]:
    """
    Process MMS media attachments using Gemini AI.

    Args:
        form: Request form data
        num_media: Number of media attachments

    Returns:
        List of media summaries/descriptions
    """
    summaries = []
    gemini_client = get_gemini_client()

    for i in range(num_media):
        url = form.get(f"MediaUrl{i}")
        content_type = form.get(f"MediaContentType{i}")

        if not url or not content_type:
            continue

        logger.info(f"Processing media {i + 1}/{num_media}: {url} ({content_type})")

        # Only process images for now
        if not content_type.startswith("image/"):
            logger.info(f"Skipping unsupported media type: {content_type}")
            continue

        try:
            # Download image with size limit
            image_bytes = await download_media(url, config.MAX_IMAGE_SIZE_BYTES)

            # Process with Gemini
            caption = form.get(f"MediaCaption{i}")
            summary = await gemini_client.process_image(
                image_bytes, content_type, caption
            )
            logger.info(f"Image {i + 1} processed: {summary[:100]}...")
            summaries.append(summary)

        except ValueError as e:
            logger.warning(f"Image {i + 1} size limit exceeded: {e}")
            summaries.append(
                f"Image too large to process (max {config.MAX_IMAGE_SIZE_MB}MB)"
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image {i + 1}: {e}")
            summaries.append("Error downloading image.")
        except httpx.TimeoutException:
            logger.error(f"Timeout downloading image {i + 1}")
            summaries.append("Timeout downloading image.")
        except Exception as e:
            logger.error(f"Error processing image {i + 1}: {e}")
            summaries.append("Error processing image.")

    return summaries


# ===========================
# API Endpoints
# ===========================


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Twilio-GCA-Gemini Integration",
        "version": "0.1.0",
    }


@app.post("/sms-webhook")
async def receive_sms(
    request: Request,
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: int = Form(0),
):
    """
    Handle incoming SMS messages from Twilio.

    Processes text and MMS images, sends to Conversational Agent,
    and detects live agent handoff requests.

    Args:
        request: FastAPI request object
        From: Sender phone number (E.164 format)
        Body: Message text
        NumMedia: Number of media attachments

    Returns:
        TwiML response
    """
    logger.info(f"📩 Incoming SMS from {From}")
    logger.info(f"Body: {Body}, Media: {NumMedia}")

    # Get form data for signature validation and media processing
    form = dict((await request.form()).items())

    # Validate Twilio signature for security
    request_url = str(request.url)
    headers = dict(request.headers)

    if not validate_twilio_signature(
        req_url=request_url,
        headers=headers,
        body=form,
        auth_token=config.TWILIO_AUTH_TOKEN,
    ):
        logger.warning(f"⚠️ Invalid Twilio signature from {From} for URL: {request_url}")
        return PlainTextResponse("Forbidden", status_code=403)

    logger.info("✓ Twilio signature validated")

    # Validate phone number format
    if not validate_phone_number(From):
        logger.warning(f"Invalid phone number format: {From}")
        return PlainTextResponse("Invalid phone number", status_code=400)

    # Process media attachments
    media_summaries = []
    if NumMedia > 0:
        media_summaries = await process_media_attachments(form, NumMedia)

    # Combine text and media summaries
    if media_summaries:
        full_message = Body + "\n\nImage summary:\n" + "\n".join(media_summaries)
    else:
        full_message = Body

    # Send to Conversational Agent
    # Use phone number as user_id (remove '+' for session ID compatibility)
    user_id = From.replace("+", "")

    # Validate session ID length (6-15 digits for E.164)
    if len(user_id) < 6 or len(user_id) > 15:
        logger.warning(f"Invalid session ID length: {user_id}")
        return PlainTextResponse("Invalid phone number", status_code=400)

    try:
        ca_client = get_ca_client()
        ca_response = await ca_client.detect_intent(text=full_message, user_id=user_id)

        logger.info(
            f"CA Response - Intent: {ca_response['intent']}, "
            f"Confidence: {ca_response['confidence']:.2f}, "
            f"Handoff: {ca_response['is_handoff']}"
        )

    except Exception as e:
        logger.error(f"Error calling Conversational Agent: {e}")
        return PlainTextResponse("Error processing message", status_code=500)

    # Determine response based on handoff status
    if ca_response["is_handoff"]:
        logger.info(f"🚨 Live Agent Handoff detected for SMS from {From}")
        reply = "A human assistant will reach out to you shortly."
    else:
        logger.info(f"🤖 Normal agent reply for {From}")
        # Use the agent's response text if available
        reply = (
            ca_response.get("response_text")
            or "Thank you! We'll analyze this and let you know."
        )

    # Generate TwiML response
    twiml_response = build_twiml_message(reply)
    logger.info(f"📤 Sending SMS reply to {From}: {reply[:100]}...")
    logger.debug(f"TwiML Response: {twiml_response}")

    return Response(content=twiml_response, media_type="text/xml")
