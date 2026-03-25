"""
Tier Engine — Tier-based tool routing for AI agents.

Makes OpenClaw usable on any model size: 0.5B to 400B+.
Small models get MCQ, medium get condensed, large get full tools.
"""

import os
import time
import json
import uuid
import sqlite3
import logging
import threading
from typing import Optional, Callable, Any

from tier_engine.router import ToolRouter
from tier_engine.detector import detect_tier, get_tier_config
from tier_engine.models import ModelTier, ToolDef, RouteResult
from tier_engine.errors import ToolNotFoundError

logger = logging.getLogger("tier")

DEFAULT_DB_PATH = "./tier.db"

# Common OpenClaw tools with descriptions
DEFAULT_TOOLS = [
    ToolDef(name="file_read", description="Read a file from disk", short_description="Read file", category="filesystem", parameters={"path": "string"}),
    ToolDef(name="file_write", description="Write content to a file on disk", short_description="Write file", category="filesystem", parameters={"path": "string", "content": "string"}),
    ToolDef(name="file_delete", description="Delete a file from disk", short_description="Delete file", category="filesystem", parameters={"path": "string"}),
    ToolDef(name="grep", description="Search for text patterns in files using regex", short_description="Search in files", category="search", parameters={"pattern": "string", "path": "string"}),
    ToolDef(name="glob", description="Find files matching a pattern like *.py or src/**/*.ts", short_description="Find files", category="search", parameters={"pattern": "string"}),
    ToolDef(name="shell_exec", description="Execute a shell command and return output", short_description="Run command", category="system", parameters={"command": "string"}),
    ToolDef(name="web_search", description="Search the web for information", short_description="Search web", category="web", parameters={"query": "string"}),
    ToolDef(name="web_fetch", description="Fetch content from a URL and extract text", short_description="Fetch URL", category="web", parameters={"url": "string"}),
    ToolDef(name="git_status", description="Show the working tree status of a git repository", short_description="Git status", category="git", parameters={}),
    ToolDef(name="git_diff", description="Show changes between commits or working tree", short_description="Git diff", category="git", parameters={}),
    ToolDef(name="git_log", description="Show commit history log", short_description="Git log", category="git", parameters={"limit": "integer"}),
    ToolDef(name="git_commit", description="Create a new git commit with a message", short_description="Git commit", category="git", parameters={"message": "string"}),
    ToolDef(name="image_generate", description="Generate an image from a text description", short_description="Generate image", category="media", parameters={"prompt": "string"}),
    ToolDef(name="code_run", description="Execute Python code in a sandboxed environment", short_description="Run code", category="code", parameters={"code": "string"}),
    ToolDef(name="http_request", description="Make an HTTP request to an API endpoint", short_description="HTTP request", category="api", parameters={"method": "string", "url": "string", "body": "object"}),
    ToolDef(name="database_query", description="Execute a SQL query against a database", short_description="SQL query", category="data", parameters={"query": "string"}),
    ToolDef(name="calendar_event", description="Create a calendar event with date and time", short_description="Create event", category="productivity", parameters={"title": "string", "date": "string"}),
    ToolDef(name="send_email", description="Send an email to a recipient", short_description="Send email", category="communication", parameters={"to": "string", "subject": "string", "body": "string"}),
    ToolDef(name="send_message", description="Send a message to a chat channel", short_description="Send message", category="communication", parameters={"channel": "string", "text": "string"}),
    ToolDef(name="memory_store", description="Store information in long-term memory", short_description="Remember this", category="memory", parameters={"content": "string"}),
    ToolDef(name="memory_recall", description="Recall information from long-term memory", short_description="Remember about", category="memory", parameters={"query": "string"}),
    ToolDef(name="summarize", description="Summarize a long text into key points", short_description="Summarize text", category="text", parameters={"text": "string"}),
    ToolDef(name="translate", description="Translate text from one language to another", short_description="Translate", category="text", parameters={"text": "string", "target_lang": "string"}),
    ToolDef(name="screenshot", description="Take a screenshot of the current screen", short_description="Screenshot", category="media", parameters={}),
    ToolDef(name="clipboard_read", description="Read the current clipboard contents", short_description="Read clipboard", category="system", parameters={}),
]


class TierEngine:
    """
    Tier-based tool routing engine.

    Registers tools, detects model capability, and routes intents
    to the right tools in the right format for the model tier.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._lock = threading.Lock()

        # Initialize router
        use_st = self.config.get("use_sentence_transformer", False)
        self.router = ToolRouter(use_sentence_transformer=use_st)

        # SQLite for usage stats
        db_path = self.config.get("db_path", os.environ.get("TIER_DB_PATH", DEFAULT_DB_PATH))
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        # Load default tools
        if self.config.get("load_defaults", True):
            self.router.register_tools(DEFAULT_TOOLS)

        # Load custom tools from DB
        self._load_custom_tools()

        logger.info("TierEngine initialized (tools=%d, db=%s)", self.router.tool_count, db_path)

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS custom_tools (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                short_description TEXT,
                parameters TEXT,
                category TEXT DEFAULT 'custom',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL,
                model_name TEXT,
                tier TEXT,
                tool_selected TEXT,
                tool_executed TEXT,
                success INTEGER,
                duration_ms INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_tool ON usage_log(tool_executed, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_usage_tier ON usage_log(tier, created_at DESC);
        """)
        self._conn.commit()

    def _load_custom_tools(self):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM custom_tools")
        for row in cur.fetchall():
            self.router.register_tool(ToolDef(
                name=row["name"],
                description=row["description"],
                short_description=row["short_description"] or "",
                parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                category=row["category"] or "custom",
            ))

    # ── Public API ─────────────────────────────────────────────────────

    def route(self, intent: str, model_name: str = "") -> RouteResult:
        """Route an intent to matching tools based on model tier."""
        return self.router.route(intent, model_name)

    def route_and_format(self, intent: str, model_name: str = "") -> dict:
        """Route and return formatted prompt text."""
        result = self.router.route(intent, model_name)
        prompt_text = self.router.format_for_prompt(result)

        return {
            "tier": result.tier,
            "format": result.format,
            "tools": result.tools,
            "scores": result.scores,
            "prompt": prompt_text,
            "mcq_options": result.mcq_options,
        }

    def resolve_mcq(self, choice: str, route_result_tools: list, route_result_mcq: dict) -> Optional[str]:
        """Resolve MCQ choice to tool name."""
        result = RouteResult(mcq_options=route_result_mcq, tools=route_result_tools)
        return self.router.resolve_mcq(choice, result)

    def register_tool(
        self,
        name: str,
        description: str,
        short_description: str = "",
        parameters: Optional[dict] = None,
        category: str = "custom",
    ) -> bool:
        """Register a custom tool."""
        tool = ToolDef(
            name=name,
            description=description,
            short_description=short_description,
            parameters=parameters or {},
            category=category,
        )
        self.router.register_tool(tool)

        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO custom_tools
                   (name, description, short_description, parameters, category, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, description, short_description, json.dumps(parameters or {}), category, now),
            )
            self._conn.commit()
        return True

    def record_usage(
        self,
        intent: str,
        model_name: str,
        tier: str,
        tool_selected: str,
        tool_executed: str = "",
        success: bool = True,
        duration_ms: int = 0,
    ):
        """Record tool usage for analytics and ranking."""
        self.router.record_usage(tool_executed or tool_selected, success)

        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        with self._lock:
            self._conn.execute(
                """INSERT INTO usage_log
                   (intent, model_name, tier, tool_selected, tool_executed, success, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (intent, model_name, tier, tool_selected, tool_executed or tool_selected,
                 int(success), duration_ms, now),
            )
            self._conn.commit()

    def detect_tier(self, model_name: str) -> dict:
        """Detect model tier and return config."""
        tier = detect_tier(model_name)
        config = get_tier_config(tier)
        return {"tier": tier.value, "model": model_name, **config}

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        cur = self._conn.cursor()

        cur.execute("SELECT COUNT(*) as c FROM usage_log")
        total_routes = cur.fetchone()["c"]

        cur.execute("SELECT tier, COUNT(*) as c FROM usage_log GROUP BY tier")
        by_tier = {row["tier"]: row["c"] for row in cur.fetchall()}

        cur.execute(
            "SELECT tool_executed, COUNT(*) as c FROM usage_log GROUP BY tool_executed ORDER BY c DESC LIMIT 10"
        )
        top_tools = {row["tool_executed"]: row["c"] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) as c FROM usage_log WHERE success = 1")
        success_count = cur.fetchone()["c"]

        return {
            "registered_tools": self.router.tool_count,
            "total_routes": total_routes,
            "routes_by_tier": by_tier,
            "top_tools": top_tools,
            "success_rate": round(success_count / max(total_routes, 1), 3),
        }

    def health_check(self) -> dict:
        return {
            "healthy": True,
            "engine": "tier",
            "tools": self.router.tool_count,
            "tiers": ["S", "M", "L", "XL"],
        }

    # ── Lifecycle ──────────────────────────────────────────────────────

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
