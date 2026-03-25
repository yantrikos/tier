"""Tool router — the core intelligence of Tier.

Routes user intents to the right tools based on:
1. Embedding similarity between intent and tool descriptions
2. Model tier for response format
3. Usage statistics for ranking boost
"""

import logging
from typing import Optional, Callable

from tier_engine.models import (
    ModelTier, ToolDef, RouteResult, MCQOption,
)
from tier_engine.detector import detect_tier, get_tier_config
from tier_engine.embeddings import TFIDFEmbedder, SentenceEmbedder, cosine_similarity

logger = logging.getLogger("tier.router")

MCQ_LABELS = ["A", "B", "C", "D", "E"]


class ToolRouter:
    """
    Routes intents to tools based on embedding similarity and model tier.

    The core value: instead of showing 57 tools to a 3B model,
    show 4 MCQ options ranked by relevance.
    """

    def __init__(self, use_sentence_transformer: bool = False):
        self._tools: dict[str, ToolDef] = {}
        self._tfidf = TFIDFEmbedder()
        self._st: Optional[SentenceEmbedder] = None
        self._use_st = use_sentence_transformer
        self._fitted = False

        if use_sentence_transformer:
            self._st = SentenceEmbedder()
            if not self._st.available:
                self._st = None
                self._use_st = False

    def register_tool(self, tool: ToolDef):
        """Register a tool for routing."""
        self._tools[tool.name] = tool
        self._fitted = False  # need re-fit

    def register_tools(self, tools: list[ToolDef]):
        """Register multiple tools."""
        for t in tools:
            self._tools[t.name] = t
        self._fitted = False

    def unregister_tool(self, name: str):
        self._tools.pop(name, None)
        self._fitted = False

    def _ensure_fitted(self):
        """Fit embeddings if needed."""
        if self._fitted or not self._tools:
            return

        descriptions = [
            f"{t.name}: {t.description}" for t in self._tools.values()
        ]

        if self._use_st and self._st:
            embeddings = self._st.embed_batch(descriptions)
            for tool, emb in zip(self._tools.values(), embeddings):
                tool.embedding = emb
        else:
            self._tfidf.fit(descriptions)
            for tool, desc in zip(self._tools.values(), descriptions):
                tool.embedding = self._tfidf.embed(desc)

        self._fitted = True
        logger.info("Fitted %d tools for routing", len(self._tools))

    def route(
        self,
        intent: str,
        model_name: str = "",
        tier_override: Optional[str] = None,
    ) -> RouteResult:
        """
        Route an intent to matching tools.

        Returns a RouteResult with format appropriate for the model tier.
        """
        self._ensure_fitted()

        # Detect tier
        tier = detect_tier(model_name, {"tier": tier_override} if tier_override else None)
        config = get_tier_config(tier)

        # Embed intent
        if self._use_st and self._st:
            intent_emb = self._st.embed(intent)
        else:
            intent_emb = self._tfidf.embed(intent)

        # Score all tools
        scored = []
        for tool in self._tools.values():
            if tool.embedding:
                sim = cosine_similarity(intent_emb, tool.embedding)
                # Boost by usage (slight preference for proven tools)
                boost = min(tool.usage_count * 0.001, 0.1)
                scored.append((tool, sim + boost))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Limit based on tier
        max_tools = config["max_tools"] or len(scored)
        top = scored[:max_tools]

        result = RouteResult(
            tier=tier.value,
            format=config["format"],
            tools=[t.name for t, _ in top],
            scores=[round(s, 4) for _, s in top],
        )

        # Format based on tier
        if config["format"] == "mcq":
            result.mcq_options = self._format_mcq(top, config)
        elif config["format"] == "direct" and top:
            result.selected_tool = top[0][0].name
            result.format = "direct"

        return result

    def _format_mcq(self, candidates: list, config: dict) -> dict:
        """Format top candidates as MCQ for small models."""
        max_len = config["description_length"]
        options = {}

        for i, (tool, score) in enumerate(candidates[:len(MCQ_LABELS)]):
            label = MCQ_LABELS[i]
            desc = tool.short_description or tool.description
            if max_len and len(desc) > max_len:
                desc = desc[:max_len - 3] + "..."
            options[label] = {
                "tool": tool.name,
                "description": desc,
                "score": round(score, 3),
            }

        # Add "none of the above"
        if len(options) < len(MCQ_LABELS):
            options[MCQ_LABELS[len(options)]] = {
                "tool": "_none",
                "description": "None of the above",
                "score": 0,
            }

        return options

    def format_for_prompt(self, result: RouteResult) -> str:
        """Format a RouteResult as text suitable for an LLM prompt."""
        if result.format == "mcq" and result.mcq_options:
            lines = ["Pick the best tool for this task:\n"]
            for label, opt in result.mcq_options.items():
                lines.append(f"  {label}) {opt['tool']} — {opt['description']}")
            lines.append("\nRespond with just the letter (A, B, C, or D).")
            return "\n".join(lines)

        elif result.format == "condensed":
            lines = ["Available tools (ranked by relevance):\n"]
            for name, score in zip(result.tools, result.scores):
                tool = self._tools.get(name)
                if tool:
                    desc = tool.short_description or tool.description[:100]
                    lines.append(f"  - {name}: {desc}")
            return "\n".join(lines)

        elif result.format == "ranked":
            lines = ["Tools ranked by relevance:\n"]
            for name, score in zip(result.tools, result.scores):
                tool = self._tools.get(name)
                if tool:
                    lines.append(f"  [{score:.2f}] {name}: {tool.description}")
                    if tool.parameters:
                        params = ", ".join(tool.parameters.keys())
                        lines.append(f"         params: {params}")
            return "\n".join(lines)

        else:  # full
            lines = []
            for name in result.tools:
                tool = self._tools.get(name)
                if tool:
                    lines.append(f"{name}: {tool.description}")
            return "\n".join(lines)

    def resolve_mcq(self, choice: str, route_result: RouteResult) -> Optional[str]:
        """Resolve an MCQ choice (A/B/C/D) to a tool name."""
        if not route_result.mcq_options:
            return None
        choice = choice.strip().upper()
        opt = route_result.mcq_options.get(choice)
        if opt and opt["tool"] != "_none":
            return opt["tool"]
        return None

    def record_usage(self, tool_name: str, success: bool = True):
        """Record tool usage for ranking boost."""
        tool = self._tools.get(tool_name)
        if tool:
            tool.usage_count += 1
            if not success:
                tool.success_rate = (
                    tool.success_rate * (tool.usage_count - 1) + (1 if success else 0)
                ) / tool.usage_count

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
