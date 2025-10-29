"""
Google Conversational Agents (Dialogflow CX) client for intent detection.
"""

import asyncio
from typing import Any

from google.cloud.dialogflowcx_v3 import SessionsClient
from google.cloud.dialogflowcx_v3.types import (
    DetectIntentRequest,
    QueryInput,
    TextInput,
)
from google.oauth2 import service_account

from src import logging
from src.config import (
    CA_AGENT_ID,
    CA_LOCATION,
    CA_PROJECT_ID,
    get_gcp_credentials_dict,
)

logger = logging.getLogger(__name__)


class ConversationalAgentClient:
    """Client for interacting with Google Conversational Agents (Dialogflow CX)"""

    def __init__(self, session_prefix: str = "meta-whatsapp"):
        """
        Initialize the Conversational Agent client.

        Args:
            session_prefix: Prefix to use for session IDs (default: "meta-whatsapp")
        """
        # Validate configuration
        if not all([CA_PROJECT_ID, CA_AGENT_ID, CA_LOCATION]):
            raise ValueError(
                "Missing required CA configuration: CA_PROJECT_ID, CA_AGENT_ID, CA_LOCATION"
            )

        credentials_dict = get_gcp_credentials_dict()
        if not credentials_dict:
            raise ValueError(
                "GCP_SERVICE_ACCOUNT_JSON must be set to either a JSON string or path to credentials file"
            )

        # Create credentials from service account dict
        self.credentials = service_account.Credentials.from_service_account_info(
            credentials_dict
        )

        # Initialize the Sessions client
        self.project_id = CA_PROJECT_ID
        self.agent_id = CA_AGENT_ID
        self.location = CA_LOCATION
        self.session_prefix = session_prefix

        # Build the agent path
        self.agent_path = f"projects/{self.project_id}/locations/{self.location}/agents/{self.agent_id}"

        logger.info(
            f"CA client initialized - Project: {self.project_id}, "
            f"Agent: {self.agent_id}, Location: {self.location}, "
            f"Session Prefix: {self.session_prefix}"
        )

    def _build_session_id(self, user_id: str) -> str:
        """
        Build a session ID using the fixed format: {prefix}-{user_id}.

        Args:
            user_id: Unique identifier for the user (e.g., phone number)

        Returns:
            Session ID in format: "meta-whatsapp-{user_id}"
        """
        # Remove any special characters and ensure valid session ID format
        clean_user_id = user_id.replace("+", "").replace("-", "").replace(" ", "")
        session_id = f"{self.session_prefix}-{clean_user_id}"
        return session_id

    def _build_session_path(self, session_id: str) -> str:
        """
        Build the full session path.

        Args:
            session_id: The session ID

        Returns:
            Full session path
        """
        return f"{self.agent_path}/sessions/{session_id}"

    async def detect_intent(
        self,
        text: str,
        user_id: str,
        language_code: str = "en",
    ) -> dict[str, Any]:
        """
        Detect intent from user text using Dialogflow CX.

        Session ID is automatically generated as: {session_prefix}-{user_id}
        Example: "meta-whatsapp-1234567890"

        Args:
            text: The user's text message
            user_id: Unique identifier for the user (e.g., phone number from WhatsApp)
            language_code: Language code (default: "en")

        Returns:
            Dictionary containing:
                - response_text: The agent's response
                - intent: Detected intent name
                - confidence: Intent detection confidence
                - parameters: Extracted parameters
                - session_id: Session ID used
                - user_id: The user ID provided
                - is_handoff: Boolean indicating if live agent handoff was detected
        """
        try:
            # Build session ID using fixed format
            session_id = self._build_session_id(user_id)
            session_path = self._build_session_path(session_id)

            logger.info(f"Detecting intent for user {user_id}, session: {session_id}, text: {text}")

            # Create the Sessions client with credentials
            client = SessionsClient(credentials=self.credentials)

            # Prepare the query input
            text_input = TextInput(text=text)
            query_input = QueryInput(text=text_input, language_code=language_code)

            # Create the request
            request = DetectIntentRequest(
                session=session_path,
                query_input=query_input,
            )

            # Make the request in thread pool (SDK is synchronous)
            response = await asyncio.to_thread(client.detect_intent, request=request)

            # Extract response information
            query_result = response.query_result

            # Get the response messages
            response_messages = []
            for response_message in query_result.response_messages:
                if response_message.text:
                    for text_part in response_message.text.text:
                        response_messages.append(text_part)

            response_text = "\n".join(response_messages) if response_messages else ""

            # Get intent information
            intent_name = ""
            confidence = 0.0
            if query_result.intent:
                intent_name = query_result.intent.display_name
                confidence = query_result.intent_detection_confidence

            # Get parameters
            parameters = {}
            if query_result.parameters:
                parameters = dict(query_result.parameters)

            # Detect live agent handoff
            is_handoff = self._detect_handoff(query_result)

            result = {
                "response_text": response_text,
                "intent": intent_name,
                "confidence": confidence,
                "parameters": parameters,
                "session_id": session_id,
                "user_id": user_id,
                "match_type": query_result.match.match_type.name if query_result.match else "UNKNOWN",
                "is_handoff": is_handoff,
            }

            logger.info(
                f"Intent detected - Intent: {intent_name}, "
                f"Confidence: {confidence:.2f}, "
                f"Match Type: {result['match_type']}, "
                f"Handoff: {is_handoff}, "
                f"Session: {session_id}"
            )

            return result

        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            raise

    def _detect_handoff(self, query_result) -> bool:
        """
        Detect if the response indicates a live agent handoff.

        Args:
            query_result: QueryResult from Dialogflow CX

        Returns:
            True if handoff detected, False otherwise
        """
        # Check response messages for live agent handoff
        for response_message in query_result.response_messages:
            # Check for live_agent_handoff message type
            if response_message.live_agent_handoff:
                logger.info("Live agent handoff detected in response_message.live_agent_handoff")
                return True

            # Check for payload with handoff indicator
            if response_message.payload:
                payload_dict = dict(response_message.payload)
                if payload_dict.get("liveAgentHandoff") or payload_dict.get("live_agent_handoff"):
                    logger.info("Live agent handoff detected in payload")
                    return True

        return False


# Singleton instance
_ca_client: ConversationalAgentClient | None = None


def get_ca_client() -> ConversationalAgentClient:
    """Get or create the Conversational Agent client singleton"""
    global _ca_client
    if _ca_client is None:
        _ca_client = ConversationalAgentClient()
    return _ca_client
