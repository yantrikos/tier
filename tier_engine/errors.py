"""Tier Engine error hierarchy."""
from typing import Optional

class TierError(Exception):
    code: str = "TIER_ERROR"
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.details = details or {}

class ToolNotFoundError(TierError):
    code = "TIER_TOOL_NOT_FOUND"

class TierDetectionError(TierError):
    code = "TIER_DETECTION_ERROR"

class EmbeddingError(TierError):
    code = "TIER_EMBEDDING_ERROR"
