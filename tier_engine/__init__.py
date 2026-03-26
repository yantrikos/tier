"""Tier — Tier-based tool routing for AI agents of any size."""
__version__ = "0.2.0"
from tier_engine.engine import TierEngine
from tier_engine.models import ModelTier, ToolDef, RouteResult
from tier_engine.detector import detect_tier, get_tier_config
from tier_engine.errors import TierError, ToolNotFoundError

# SDK bridge (available when yantrikos-sdk is installed)
try:
    from tier_engine.sdk_bridge import SDKBridge
except ImportError:
    SDKBridge = None
