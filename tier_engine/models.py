"""Tier Engine data models."""
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class ModelTier(Enum):
    SMALL = "S"      # 0.5B - 3B: MCQ selection
    MEDIUM = "M"     # 4B - 14B: condensed top-K
    LARGE = "L"      # 15B - 35B: ranked full descriptions
    XLARGE = "XL"    # 35B+: full tool set passthrough


# Model size hints from common model naming patterns
MODEL_SIZE_PATTERNS = {
    # Small (S): 0.5B - 3B
    "0.5b": ModelTier.SMALL, "0.8b": ModelTier.SMALL,
    "1b": ModelTier.SMALL, "1.5b": ModelTier.SMALL,
    "2b": ModelTier.SMALL, "3b": ModelTier.SMALL,
    ":0.5b": ModelTier.SMALL, ":1b": ModelTier.SMALL,
    ":2b": ModelTier.SMALL, ":3b": ModelTier.SMALL,
    "nano": ModelTier.SMALL, "tiny": ModelTier.SMALL,
    "270m": ModelTier.SMALL, "500m": ModelTier.SMALL,
    "ministral-3": ModelTier.SMALL, "granite4:3b": ModelTier.SMALL,

    # Medium (M): 4B - 14B
    "4b": ModelTier.MEDIUM, "7b": ModelTier.MEDIUM,
    "8b": ModelTier.MEDIUM, "9b": ModelTier.MEDIUM,
    "12b": ModelTier.MEDIUM, "14b": ModelTier.MEDIUM,
    ":4b": ModelTier.MEDIUM, ":7b": ModelTier.MEDIUM,
    ":8b": ModelTier.MEDIUM, ":9b": ModelTier.MEDIUM,

    # Large (L): 15B - 35B
    "20b": ModelTier.LARGE, "22b": ModelTier.LARGE,
    "27b": ModelTier.LARGE, "30b": ModelTier.LARGE,
    "32b": ModelTier.LARGE, "35b": ModelTier.LARGE,
    ":27b": ModelTier.LARGE, ":32b": ModelTier.LARGE,

    # XLarge (XL): 35B+
    "70b": ModelTier.XLARGE, "72b": ModelTier.XLARGE,
    "110b": ModelTier.XLARGE, "405b": ModelTier.XLARGE,

    # Named models (known tiers)
    "gpt-4": ModelTier.XLARGE, "gpt-5": ModelTier.XLARGE,
    "claude": ModelTier.XLARGE, "opus": ModelTier.XLARGE,
    "sonnet": ModelTier.LARGE, "haiku": ModelTier.MEDIUM,
    "deepseek-chat": ModelTier.XLARGE, "deepseek-coder": ModelTier.XLARGE,
    "command-r-plus": ModelTier.XLARGE, "command-r": ModelTier.LARGE,
    "gemini-pro": ModelTier.XLARGE, "gemini-flash": ModelTier.LARGE,
    "mistral-large": ModelTier.XLARGE, "mistral-medium": ModelTier.LARGE,
}


@dataclass
class ToolDef:
    """A registered tool with its description and embedding."""
    name: str = ""
    description: str = ""
    short_description: str = ""  # condensed for medium tier
    parameters: dict = field(default_factory=dict)
    category: str = "general"
    embedding: Optional[list] = None
    usage_count: int = 0
    success_rate: float = 1.0


@dataclass
class RouteResult:
    """Result of routing an intent to a tool."""
    tier: str = ""
    format: str = ""        # mcq, condensed, ranked, direct
    tools: list = field(default_factory=list)  # matched tools
    scores: list = field(default_factory=list)  # similarity scores
    mcq_options: Optional[dict] = None   # A/B/C/D for small models
    selected_tool: Optional[str] = None
    executed: bool = False
    execution_result: Any = None


@dataclass
class MCQOption:
    label: str = ""    # A, B, C, D
    tool_name: str = ""
    description: str = ""
    score: float = 0.0
