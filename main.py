import json
import hmac
import hashlib
import base64
import re
from typing import Dict, Any

from fastapi import FastAPI, Form, Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from google.oauth2 import service_account
import httpx
import google.auth.transport.requests

import config

app = FastAPI()


def validate_twilio_signature(
    req_url: str, headers: Dict[str, str], body: Dict[str, Any]
) -> bool:
    """
    Validates Twilio webhook signature to ensure request authenticity.
    Returns False if validation fails or if required credentials are missing.
    """
    try:
        # Require auth token to be set - never bypass validation
        if not config.TWILIO_AUTH_TOKEN:
            print("⚠️ TWILIO_AUTH_TOKEN not configured - rejecting request")
            return False

        signature = headers.get("x-twilio-signature")
        if not signature:
            print("⚠️ Missing x-twilio-signature header")
            return False

        # Build signature data string
        data = req_url
        for key in sorted(body.keys()):
            data += key + str(body[key])

        # Compute expected signature
        digest = hmac.new(
            config.TWILIO_AUTH_TOKEN.encode("utf-8"),
            msg=data.encode("utf-8"),
            digestmod=hashlib.sha1,
        ).digest()
        expected = base64.b64encode(digest).decode("utf-8")

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected)
    except Exception as e:
        print(f"⚠️ Signature validation error: {e}")
        return False


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_twiml_dial(
    to: str, caller_id: str, say: str = "Connecting you now. Please hold."
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{escape_xml(say)}</Say>
  <Dial callerId="{escape_xml(caller_id)}"><Number>{escape_xml(to)}</Number></Dial>
</Response>""".strip()


def extract_metadata(form_or_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts metadata from various possible field names.
    Returns empty dict if no valid metadata found.
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
                print(f"⚠️ Failed to parse metadata from key '{key}': {e}")
                continue
    return {}


@app.post("/after-agent")
async def after_agent(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type:
        form = dict((await request.form()).items())
        body = form
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}

    if config.PUBLIC_URL:
        req_url = config.PUBLIC_URL
    else:
        req_url = str(request.url)

    if not validate_twilio_signature(req_url, request.headers, body):
        raise HTTPException(status_code=403, detail="Invalid signature")

    status = (
        body.get("VirtualAgentStatus") or body.get("virtualAgentStatus") or ""
    ).lower()
    if status != "live-agent-handoff":
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="text/xml",
        )

    meta = extract_metadata(body)
    to = meta.get("transfer_to") or body.get("transfer_to") or config.CAREGIVER_E164
    say = meta.get("announce") or "Connecting you now. Please hold."

    twiml = build_twiml_dial(to=to, caller_id=config.CALLER_ID_E164, say=say)
    return Response(content=twiml, media_type="text/xml")


# Global credential cache
_cached_credentials = None


def load_service_account_credentials():
    """
    Loads and caches service account credentials.
    Credentials are cached globally to enable proper token refresh.
    """
    global _cached_credentials

    if _cached_credentials is not None:
        return _cached_credentials

    if config.GCP_SA_JSON:
        info = json.loads(config.GCP_SA_JSON)
        _cached_credentials = service_account.Credentials.from_service_account_info(
            info, scopes=config.GOOGLE_SCOPES
        )
        return _cached_credentials
    else:
        raise RuntimeError("No service account provided. Set GCP_SA_JSON.")


def get_google_access_token() -> str:
    """
    Returns a fresh access token from the service account.
    Caches credentials and refreshes when expired.
    """
    creds = load_service_account_credentials()
    if not creds.valid:
        request = google.auth.transport.requests.Request()
        creds.refresh(request)
    return creds.token


async def summarize_image_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """
    Calls Gemini multimodal model to describe/summarize the image content.

    Args:
        image_bytes: Raw image data
        mime_type: MIME type of the image (e.g., 'image/jpeg', 'image/png')

    Returns:
        Description of the image or error message
    """
    try:
        # Encode to base64
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Describe briefly what you see in this image in plain language, focusing on any text, warning, or message content."
                        },
                        {"inline_data": {"mime_type": mime_type, "data": img_b64}},
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
        }

        # Gemini API uses API key in URL parameter, not Authorization header
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"

        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                url,
                headers=headers,
                json=payload,
            )

            # Check for HTTP errors
            if res.status_code != 200:
                error_detail = res.text[:200]
                print(f"⚠️ Gemini API error {res.status_code}: {error_detail}")
                return f"Error analyzing image (API error {res.status_code})."

            data = res.json()

        # Extract text from response
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        return text.strip() or "Unable to summarize image."
    except httpx.TimeoutException:
        print("⚠️ Gemini API timeout")
        return "Error analyzing image (timeout)."
    except Exception as e:
        print(f"⚠️ Gemini summarization error: {e}")
        return "Error analyzing image."


# --- Helper: send text to Conversational Agent ---
async def send_to_gca(session_id: str, text: str):
    """
    Sends text to GCA detectIntent API.

    Args:
        session_id: Unique session identifier
        text: Message text to send to the agent

    Returns:
        JSON response from GCA API

    Raises:
        httpx.HTTPStatusError: If API returns error status
        httpx.TimeoutException: If request times out
    """
    session_url = f"{config.GCA_SESSION_BASE_URL}/{session_id}:detectIntent"

    payload = {"textInput": {"text": text, "languageCode": "en"}}

    headers = {
        "Authorization": f"Bearer {get_google_access_token()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(session_url, json=payload, headers=headers)

        # Check for HTTP errors
        if resp.status_code != 200:
            error_detail = resp.text[:200]
            print(f"⚠️ GCA API error {resp.status_code}: {error_detail}")
            resp.raise_for_status()

        return resp.json()


def validate_phone_number(phone: str) -> bool:
    """
    Basic validation for E.164 phone number format.
    Returns True if valid, False otherwise.
    """
    if not phone:
        return False
    # E.164 format: +[country code][number] (6-15 digits total)
    # Must start with +, followed by 1-15 digits
    pattern = r'^\+[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone))


@app.post("/sms-webhook")
async def receive_sms(
    request: Request,
    From: str = Form(...),
    Body: str = Form(""),
    NumMedia: int = Form(0),
):
    print(f"\n📩 Incoming message from {From}")
    print("Text Body:", Body)

    # Validate phone number
    if not validate_phone_number(From):
        print(f"⚠️ Invalid phone number format: {From}")
        return PlainTextResponse("Invalid phone number", status_code=400)

    form = await request.form()

    # Step 1: handle image attachments
    summarized_texts = []
    if NumMedia > 0:
        for i in range(NumMedia):
            url = form.get(f"MediaUrl{i}")
            ctype = form.get(f"MediaContentType{i}")
            if not url or not ctype:
                continue

            print(f"📎 Found media: {url} ({ctype})")

            if ctype.startswith("image/"):
                try:
                    # Download image using Twilio auth with size limit
                    async with httpx.AsyncClient(timeout=30) as client:
                        # Stream download to check size
                        async with client.stream(
                            "GET",
                            url,
                            auth=(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN),
                        ) as r:
                            r.raise_for_status()

                            # Read with size limit
                            image_bytes = b""
                            async for chunk in r.aiter_bytes():
                                image_bytes += chunk
                                if len(image_bytes) > config.MAX_IMAGE_SIZE_BYTES:
                                    print(f"⚠️ Image too large (>{config.MAX_IMAGE_SIZE_MB}MB)")
                                    summarized_texts.append(
                                        f"Image too large to process (max {config.MAX_IMAGE_SIZE_MB}MB)"
                                    )
                                    break
                            else:
                                # Successfully downloaded within size limit
                                summary = await summarize_image_with_gemini(image_bytes, ctype)
                                print("🖼️ Gemini summary:", summary)
                                summarized_texts.append(summary)

                except httpx.HTTPStatusError as e:
                    print(f"⚠️ HTTP error downloading image: {e}")
                    summarized_texts.append("Error downloading image.")
                except httpx.TimeoutException:
                    print("⚠️ Timeout downloading image")
                    summarized_texts.append("Timeout downloading image.")
                except Exception as e:
                    print(f"⚠️ Error processing image: {e}")
                    summarized_texts.append("Error processing image.")
            else:
                print(f"🗂️ Skipping unsupported media type: {ctype}")

    # Step 2: combine text + summaries
    if summarized_texts:
        full_message = Body + "\n\nImage summary:\n" + "\n".join(summarized_texts)
    else:
        full_message = Body

    # Step 3: send to Conversational Agent
    # Remove '+' and validate session ID length
    session_id = From.replace("+", "")
    if len(session_id) < 6 or len(session_id) > 15:
        print(f"⚠️ Invalid session ID length: {session_id}")
        return PlainTextResponse("Invalid phone number", status_code=400)

    try:
        gca_response = await send_to_gca(session_id, full_message)
    except httpx.HTTPStatusError as e:
        print(f"❌ GCA API error: {e}")
        return PlainTextResponse("Error processing message", status_code=500)
    except httpx.TimeoutException:
        print("❌ GCA timeout")
        return PlainTextResponse("Request timeout", status_code=504)
    except Exception as e:
        print(f"❌ Error calling GCA: {e}")
        return PlainTextResponse("Error processing message", status_code=500)

    # Step 4: detect Live Agent Handoff
    handoff_detected = False

    if (
        "responseType" in gca_response
        and gca_response["responseType"] == "LIVE_AGENT_HANDOFF"
    ):
        handoff_detected = True
    else:
        for msg in gca_response.get("responseMessages", []):
            if msg.get("responseType") == "LIVE_AGENT_HANDOFF":
                handoff_detected = True
                break

    if handoff_detected:
        print(f"🚨 Live Agent Handoff detected for SMS from {From}")
    else:
        print(f"🤖 No handoff. Normal agent reply for {From}")

    # Step 5: respond to Twilio
    reply = "Thank you! We'll analyze this and let you know."
    if handoff_detected:
        reply = "A human assistant will reach out to you shortly."

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>{reply}</Message>
</Response>
"""
    return Response(content=twiml, media_type="text/xml")
