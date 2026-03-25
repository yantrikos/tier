"""Tier — Tier-based tool routing for AI agents of any size."""
__version__ = "0.1.0"
from tier_engine.engine import TierEngine
from tier_engine.models import ModelTier, ToolDef, RouteResult
from tier_engine.detector import detect_tier, get_tier_config
from tier_engine.errors import TierError, ToolNotFoundError
