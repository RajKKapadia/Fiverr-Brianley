import os
import json

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError(
        "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set in environment variables"
    )

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in environment variables")

# Google Conversational Agent Configuration
GCA_PROJECT_ID = os.getenv("GCA_PROJECT_ID")
GCA_LOCATION = os.getenv("GCA_LOCATION", "us-central1")
GCA_AGENT_ID = os.getenv("GCA_AGENT_ID")

if not GCA_PROJECT_ID or not GCA_AGENT_ID:
    raise RuntimeError(
        "GCA_PROJECT_ID and GCA_AGENT_ID must be set in environment variables"
    )

GCA_SESSION_BASE_URL = f"https://{GCA_LOCATION}-dialogflow.googleapis.com/v3beta1/projects/{GCA_PROJECT_ID}/locations/{GCA_LOCATION}/agents/{GCA_AGENT_ID}/sessions"

# Google Cloud Service Account
GCP_SA_JSON = os.getenv("GCP_SA_JSON")

if not GCP_SA_JSON:
    raise RuntimeError("GCP_SA_JSON must be set in environment variables")

# Validate service account JSON is valid
try:
    json.loads(GCP_SA_JSON)
except json.JSONDecodeError as e:
    raise RuntimeError(f"GCP_SA_JSON is not valid JSON: {e}")

GOOGLE_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Public URL for webhook validation
PUBLIC_URL = os.getenv("PUBLIC_URL")

# Image processing limits
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Aliases for backward compatibility (CA = Conversational Agent)
CA_PROJECT_ID = GCA_PROJECT_ID
CA_LOCATION = GCA_LOCATION
CA_AGENT_ID = GCA_AGENT_ID


def get_gcp_credentials_dict() -> dict:
    """
    Parse and return GCP service account credentials as a dictionary.

    Returns:
        Dictionary containing service account credentials

    Raises:
        ValueError: If GCP_SA_JSON is not set or invalid
    """
    if not GCP_SA_JSON:
        raise ValueError("GCP_SA_JSON is not set in environment variables")

    try:
        credentials_dict = json.loads(GCP_SA_JSON)
        return credentials_dict
    except json.JSONDecodeError as e:
        raise ValueError(f"GCP_SA_JSON is not valid JSON: {e}")
