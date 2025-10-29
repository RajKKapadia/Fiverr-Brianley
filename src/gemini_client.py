"""
Google Gemini API client for processing media files.
"""

import asyncio
import io

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY
from src import logging

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google Gemini API"""

    def __init__(self):
        """Initialize the Gemini client"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in environment variables")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = "gemini-2.0-flash-exp"
        logger.info(f"Gemini client initialized with model: {self.model_id}")

    async def process_image(
        self, image_data: bytes, mime_type: str, caption: str | None = None
    ) -> str:
        """
        Process an image and generate a text summary.

        Args:
            image_data: Raw image bytes
            mime_type: MIME type of the image (e.g., 'image/jpeg')
            caption: Optional caption provided with the image

        Returns:
            Text summary of the image
        """
        try:
            logger.info(f"Processing image with Gemini (mime_type: {mime_type})")

            # Prepare the prompt
            prompt = "Analyze this image and provide a detailed description. "
            if caption:
                prompt += f"The user provided this caption: '{caption}'. "
            prompt += "Describe what you see, including objects, people, text, actions, and any relevant context."

            # Upload the image file
            upload_file = await self._upload_file(image_data, mime_type)

            # Generate content (run in thread pool as SDK is synchronous)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=upload_file.uri, mime_type=mime_type),
                            types.Part(text=prompt),
                        ],
                    )
                ],
            )

            summary = response.text
            logger.info(f"Image processed successfully. Summary length: {len(summary)}")
            return summary

        except Exception as e:
            logger.error(f"Error processing image with Gemini: {e}")
            raise

    async def process_document(
        self, document_data: bytes, mime_type: str, filename: str | None = None
    ) -> str:
        """
        Process a document and generate a text summary.

        Args:
            document_data: Raw document bytes
            mime_type: MIME type of the document (e.g., 'application/pdf')
            filename: Optional filename

        Returns:
            Text summary of the document
        """
        try:
            logger.info(f"Processing document with Gemini (mime_type: {mime_type}, filename: {filename})")

            # Prepare the prompt
            prompt = "Analyze this document and provide a comprehensive summary. "
            if filename:
                prompt += f"The filename is '{filename}'. "
            prompt += "Extract and summarize the key information, main points, and any important details."

            # Upload the document file
            upload_file = await self._upload_file(document_data, mime_type)

            # Generate content (run in thread pool as SDK is synchronous)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=upload_file.uri, mime_type=mime_type),
                            types.Part(text=prompt),
                        ],
                    )
                ],
            )

            summary = response.text
            logger.info(f"Document processed successfully. Summary length: {len(summary)}")
            return summary

        except Exception as e:
            logger.error(f"Error processing document with Gemini: {e}")
            raise

    async def process_audio(self, audio_data: bytes, mime_type: str) -> str:
        """
        Process an audio file and generate a transcription/summary.

        Args:
            audio_data: Raw audio bytes
            mime_type: MIME type of the audio (e.g., 'audio/ogg', 'audio/mpeg')

        Returns:
            Transcription and summary of the audio
        """
        try:
            logger.info(f"Processing audio with Gemini (mime_type: {mime_type})")

            # Prepare the prompt
            prompt = (
                "Transcribe this audio file and provide a summary. "
                "Include both the full transcription and a brief summary of the main points discussed."
            )

            # Upload the audio file
            upload_file = await self._upload_file(audio_data, mime_type)

            # Generate content (run in thread pool as SDK is synchronous)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_id,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(file_uri=upload_file.uri, mime_type=mime_type),
                            types.Part(text=prompt),
                        ],
                    )
                ],
            )

            summary = response.text
            logger.info(f"Audio processed successfully. Summary length: {len(summary)}")
            return summary

        except Exception as e:
            logger.error(f"Error processing audio with Gemini: {e}")
            raise

    async def _upload_file(self, file_data: bytes, mime_type: str) -> types.File:
        """
        Upload a file to Gemini API.

        Args:
            file_data: Raw file bytes
            mime_type: MIME type of the file

        Returns:
            Uploaded file object
        """
        try:
            # Create a file-like object from bytes
            file_obj = io.BytesIO(file_data)

            # Upload the file (run in thread pool as SDK is synchronous)
            upload_file = await asyncio.to_thread(
                self.client.files.upload,
                file=file_obj,
                config={"mime_type": mime_type}
            )

            logger.info(f"File uploaded successfully: {upload_file.name}, URI: {upload_file.uri}")
            return upload_file

        except Exception as e:
            logger.error(f"Error uploading file to Gemini: {e}")
            raise


# Singleton instance
_gemini_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get or create the Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client
