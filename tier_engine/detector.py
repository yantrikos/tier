"""Model tier detection — determine capability level from model name/metadata."""

import re
import logging
from typing import Optional

from tier_engine.models import ModelTier, MODEL_SIZE_PATTERNS

logger = logging.getLogger("tier.detector")


def detect_tier(model_name: str, context: Optional[dict] = None) -> ModelTier:
    """
    Detect model tier from its name.

    Checks:
    1. Exact match in known model patterns
    2. Size extraction from name (e.g., "qwen3.5:9b" → 9B → MEDIUM)
    3. Provider hints (e.g., "ollama/" prefix likely local/smaller)
    4. Context override if provided
    """
    if not model_name:
        return ModelTier.MEDIUM  # safe default

    # Context override
    if context and "tier" in context:
        tier_str = context["tier"].upper()
        for t in ModelTier:
            if t.value == tier_str:
                return t

    name_lower = model_name.lower().strip()

    # Strip provider prefix (e.g., "ollama/", "openai/")
    if "/" in name_lower:
        name_lower = name_lower.split("/", 1)[1]

    # Try to extract size from name FIRST (most precise)
    # Matches: "27b", ":9b-8k", "-8b-instruct", "32b-instruct"
    size_match = re.search(r'(?:^|[:\-_/])(\d+(?:\.\d+)?)\s*[bB]', name_lower)
    if size_match:
        size_b = float(size_match.group(1))
        if size_b <= 3:
            return ModelTier.SMALL
        elif size_b <= 14:
            return ModelTier.MEDIUM
        elif size_b <= 35:
            return ModelTier.LARGE
        else:
            return ModelTier.XLARGE

    # Named model match (for models without size in name)
    for pattern, tier in MODEL_SIZE_PATTERNS.items():
        if pattern in name_lower:
            logger.debug("Tier detected: %s → %s (pattern: %s)", model_name, tier.value, pattern)
            return tier

    # Provider-based heuristics
    if any(p in model_name.lower() for p in ["ollama", "local", "gguf", "q4_", "q8_"]):
        return ModelTier.MEDIUM  # local models are usually medium-sized

    # Default: MEDIUM (safe — doesn't overwhelm small models, doesn't limit large ones too much)
    logger.debug("Tier defaulting to MEDIUM for: %s", model_name)
    return ModelTier.MEDIUM


def get_tier_config(tier: ModelTier) -> dict:
    """Get configuration for a tier."""
    configs = {
        ModelTier.SMALL: {
            "max_tools": 4,
            "format": "mcq",
            "description_length": 50,
            "show_parameters": False,
            "token_budget": 200,
        },
        ModelTier.MEDIUM: {
            "max_tools": 8,
            "format": "condensed",
            "description_length": 100,
            "show_parameters": True,
            "token_budget": 500,
        },
        ModelTier.LARGE: {
            "max_tools": 20,
            "format": "ranked",
            "description_length": 200,
            "show_parameters": True,
            "token_budget": 1500,
        },
        ModelTier.XLARGE: {
            "max_tools": 0,  # unlimited
            "format": "full",
            "description_length": 0,  # unlimited
            "show_parameters": True,
            "token_budget": 0,  # unlimited
        },
    }
    return configs.get(tier, configs[ModelTier.MEDIUM])
