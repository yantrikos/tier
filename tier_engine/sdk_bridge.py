"""
SDK Bridge — connects yantrikos-sdk tools to the Tier plugin.

Tools registered via the SDK's @register decorator are automatically
available to the Tier engine. The SDK's TierRouter provides the
routing strategies backed by the whitepaper benchmarks.

Usage:
    from tier_engine.sdk_bridge import SDKBridge

    bridge = SDKBridge(model_name="qwen2.5:1.5b")

    # Import any module that uses @register
    import my_tools

    # Route using SDK strategies
    native_tools = bridge.route("Read the file config.yaml")
    native_tools, hint = bridge.route_with_hint("Read config.yaml")

    # Get info
    print(bridge.info())
"""

import logging
from typing import Optional, Callable

logger = logging.getLogger("tier.sdk_bridge")

try:
    from yantrikos import (
        TierRouter, BaseTool, Tier, ToolResult,
        detect_tier, detect_model_family,
        all_tools, by_category, categories, count,
        to_native_tool, to_native_tools, schemas, full_schemas,
    )
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    logger.warning("yantrikos-sdk not installed. SDK bridge disabled. pip install yantrikos-sdk")


class SDKBridge:
    """
    Bridge between yantrikos-sdk and the Tier plugin.

    Wraps TierRouter and provides the interface the plugin needs.
    """

    def __init__(
        self,
        model_name: str = "",
        tier: Optional[str] = None,
        ranker: Optional[Callable] = None,
        detailed_k: int = 8,
    ):
        if not SDK_AVAILABLE:
            raise RuntimeError("yantrikos-sdk not installed. Run: pip install yantrikos-sdk")

        tier_enum = None
        if tier:
            tier_enum = Tier(tier)

        self._router = TierRouter(
            model_name=model_name,
            tier=tier_enum,
            ranker=ranker,
            detailed_k=detailed_k,
        )
        self._model_name = model_name

        logger.info(
            "SDKBridge initialized: model=%s tier=%s strategy=%s sdk_tools=%d",
            model_name, self._router.tier.value,
            self._router._get_strategy_name(), count(),
        )

    def route(self, user_prompt: str) -> list[dict]:
        """
        Route tools for a user prompt. Returns native tool definitions.

        Uses the SDK's TierRouter which selects strategy based on tier:
        - Tier S/M: hybrid (K detailed + rest name-only)
        - Tier L: all tools, semantically reordered
        - Tier XL: all tools, original order
        """
        tools = all_tools()
        if not tools:
            logger.warning("No SDK tools registered")
            return []
        return self._router.route(user_prompt, tools)

    def route_with_hint(self, user_prompt: str) -> tuple:
        """Route tools AND return a system prompt hint. Returns (native_tools, hint)."""
        tools = all_tools()
        if not tools:
            return [], ""
        return self._router.route_with_hint(user_prompt, tools)

    def get_native_tools(self, tier_str: str = "") -> list[dict]:
        """Get all tools in native format for a specific tier."""
        tier = Tier(tier_str) if tier_str else self._router.tier
        tools = all_tools()
        return [to_native_tool(t, tier) for t in tools]

    def get_schemas(self, tier_str: str = "") -> list[dict]:
        """Get all tool schemas for a tier."""
        tier = Tier(tier_str) if tier_str else self._router.tier
        return schemas(tier)

    def execute_tool(self, tool_name: str, input_data: dict) -> dict:
        """Execute a registered SDK tool."""
        from yantrikos import get
        tool = get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}

        result = tool.safe_execute(input_data, self._router.tier)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }

    def info(self) -> dict:
        """Router and SDK info."""
        return {
            **self._router.info(),
            "sdk_version": _get_sdk_version(),
            "sdk_tools": count(),
            "sdk_categories": categories(),
        }

    @property
    def tier(self) -> str:
        return self._router.tier.value

    @property
    def strategy(self) -> str:
        return self._router._get_strategy_name()


def _get_sdk_version() -> str:
    try:
        from yantrikos import __version__
        return __version__
    except:
        return "unknown"
