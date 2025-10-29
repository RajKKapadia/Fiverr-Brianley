"""
Utility functions for TwiML generation, validation, and common operations.
"""

import re
import hmac
import hashlib
import base64
from typing import Dict, Any


def escape_xml(s: str) -> str:
    """
    Escape special XML characters for TwiML responses.

    Args:
        s: String to escape

    Returns:
        XML-escaped string
    """
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def build_twiml_message(message: str) -> str:
    """
    Build a TwiML response with a message.

    Args:
        message: Message text to include in response

    Returns:
        TwiML XML string
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Message>{escape_xml(message)}</Message>
</Response>""".strip()


def build_twiml_dial(
    to: str, caller_id: str, say: str = "Connecting you now. Please hold."
) -> str:
    """
    Build a TwiML dial response for call forwarding.

    Args:
        to: Phone number to dial (E.164 format)
        caller_id: Caller ID to display (E.164 format)
        say: Message to say before dialing

    Returns:
        TwiML XML string
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say>{escape_xml(say)}</Say>
  <Dial callerId="{escape_xml(caller_id)}"><Number>{escape_xml(to)}</Number></Dial>
</Response>""".strip()


def build_twiml_empty() -> str:
    """
    Build an empty TwiML response.

    Returns:
        Empty TwiML XML string
    """
    return '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'


def validate_phone_number(phone: str) -> bool:
    """
    Validate phone number in E.164 format.

    E.164 format: +[country code][number] (6-15 digits total)
    Must start with +, followed by 1-15 digits

    Args:
        phone: Phone number to validate

    Returns:
        True if valid E.164 format, False otherwise
    """
    if not phone:
        return False
    pattern = r"^\+[1-9]\d{1,14}$"
    return bool(re.match(pattern, phone))


def validate_twilio_signature(
    req_url: str, headers: Dict[str, str], body: Dict[str, Any], auth_token: str
) -> bool:
    """
    Validate Twilio webhook signature to ensure request authenticity.

    Uses HMAC-SHA1 with constant-time comparison to prevent timing attacks.

    Args:
        req_url: Full request URL
        headers: Request headers dict
        body: Request body as dict
        auth_token: Twilio auth token

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Require auth token to be set - never bypass validation
        if not auth_token:
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
            auth_token.encode("utf-8"),
            msg=data.encode("utf-8"),
            digestmod=hashlib.sha1,
        ).digest()
        expected = base64.b64encode(digest).decode("utf-8")

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(signature, expected)
    except Exception as e:
        print(f"⚠️ Signature validation error: {e}")
        return False
