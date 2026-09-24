"""Domain-level exceptions and helpers.

These are caught by a global FastAPI exception handler in ``main.py`` and
converted into the JSON error contract:

    {"status": "failed", "error": {"code": ..., "message": ...}}

Keeping these in one place means route handlers and services never need to
leak stack traces or provider credentials to the client.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors."""

    #: Stable machine-readable code returned to clients.
    code: str = "ERROR"
    #: Default HTTP status for this error type.
    http_status: int = 500
    #: Human-readable message returned to clients.
    message: str = "Unexpected error"

    def __init__(self, message: str | None = None, *, code: str | None = None) -> None:
        if message is not None:
            self.message = message
        if code is not None:
            self.code = code
        super().__init__(self.message)

    def to_payload(self) -> dict:
        return {"status": "failed", "error": {"code": self.code, "message": self.message}}


class NotFoundError(AppError):
    code = "NOT_FOUND"
    http_status = 404
    message = "The requested resource was not found"


class InvalidStateError(AppError):
    """Raised when a comic is not in the state required for the operation."""

    code = "INVALID_STATE"
    http_status = 409
    message = "The comic is not in the required state"


class ValidationError(AppError):
    code = "VALIDATION_ERROR"
    http_status = 422
    message = "The request failed validation"


class AICallError(AppError):
    """Raised when an external AI provider (Gemini / image API) fails.

    Never embeds provider internals or credentials.
    """

    code = "AI_CALL_FAILED"
    http_status = 502
    message = "An upstream AI service call failed"


class ImageGenerationError(AICallError):
    code = "IMAGE_GENERATION_FAILED"
    message = "Image generation failed"


class StoryGenerationError(AICallError):
    code = "STORY_GENERATION_FAILED"
    message = "Story generation failed"


class ExportError(AppError):
    code = "EXPORT_FAILED"
    http_status = 500
    message = "Comic export failed"