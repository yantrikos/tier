#!/usr/bin/env python3
"""
Tier-Based Adaptive Tool Routing — Benchmark Harness

Measures tool selection accuracy, token usage, and latency across model sizes
with baseline (all tools) vs tier-adapted (filtered/MCQ) presentation.

For the whitepaper: "Tier-Based Adaptive Tool Routing for
Capability-Heterogeneous AI Agents"
"""

import json
import time
import sys
import os
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone

import urllib.request

try:
    import yantrikdb
    YANTRIKDB_AVAILABLE = True
except ImportError:
    YANTRIKDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# YantrikDB Semantic Tool Ranker
# ---------------------------------------------------------------------------

class SemanticToolRanker:
    """Uses YantrikDB's vector index to rank tools by semantic similarity."""

    def __init__(self, tools: list):
        if not YANTRIKDB_AVAILABLE:
            raise RuntimeError("yantrikdb not installed")
        from sentence_transformers import SentenceTransformer
        self.db = yantrikdb.YantrikDB(db_path="/tmp/tier_benchmark_tools.db")
        self.db.set_embedder(SentenceTransformer("all-MiniLM-L6-v2"))
        self._index_tools(tools)
        self._tool_map = {t["name"]: t for t in tools}

    def _index_tools(self, tools: list):
        """Index all tool descriptions into YantrikDB."""
        for t in tools:
            # Combine name, category, short and full descriptions for rich embedding
            text = f"{t['name'].replace('_', ' ')}: {t['description_full']} Category: {t['category']}"
            try:
                self.db.record(
                    text=text,
                    memory_type="semantic",
                    importance=0.5,
                    metadata={"tool_name": t["name"], "category": t["category"]},
                )
            except Exception:
                pass  # Skip if already indexed

    def rank(self, query: str, top_k: int = 4) -> list:
        """Rank tools by semantic similarity to query. Returns list of tool dicts."""
        results = self.db.recall(query=query, top_k=top_k * 2)  # fetch extra in case of dupes
        ranked = []
        seen = set()
        for r in results:
            metadata = r.get("metadata") or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            tool_name = metadata.get("tool_name")
            if tool_name and tool_name in self._tool_map and tool_name not in seen:
                ranked.append(self._tool_map[tool_name])
                seen.add(tool_name)
                if len(ranked) >= top_k:
                    break
        return ranked

    def close(self):
        self.db.close()


# Global ranker instance (initialized lazily)
_semantic_ranker = None

def get_semantic_ranker(tools: list) -> SemanticToolRanker:
    global _semantic_ranker
    if _semantic_ranker is None and YANTRIKDB_AVAILABLE:
        print("  [YantrikDB] Indexing tools for semantic ranking...")
        _semantic_ranker = SemanticToolRanker(tools)
        print(f"  [YantrikDB] Indexed {len(tools)} tools")
    return _semantic_ranker


def semantic_rank_tools(query: str, tools: list, top_k: int = 4) -> list:
    """Rank tools using YantrikDB semantic search. Falls back to keyword if unavailable."""
    ranker = get_semantic_ranker(tools)
    if ranker:
        ranked = ranker.rank(query, top_k=top_k)
        if ranked:
            return ranked
    # Fallback to keyword matching
    return keyword_rank_tools(query, tools, top_k)


def keyword_rank_tools(query: str, tools: list, top_k: int = 4) -> list:
    """Rank tools using keyword matching (fallback)."""
    scored = []
    msg_lower = query.lower()
    for t in tools:
        score = 0
        for word in t["name"].replace("_", " ").split():
            if word in msg_lower:
                score += 3
        for word in t["description_short"].lower().split():
            if word in msg_lower:
                score += 1
        for word in t["description_full"].lower().split():
            if word in msg_lower:
                score += 0.5
        if t["category"].lower() in msg_lower:
            score += 2
        scored.append((score, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_k]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_OLLAMA = "http://localhost:11434"
REMOTE_OLLAMA = "http://192.168.4.35:11434"

MODELS = {
    # Tier S — sub-4B
    "gemma3:270m":        {"tier": "S", "params": "270M",  "host": REMOTE_OLLAMA},
    "qwen3.5:0.8b":      {"tier": "S", "params": "0.8B",  "host": REMOTE_OLLAMA},
    "qwen2.5:1.5b":      {"tier": "S", "params": "1.5B",  "host": LOCAL_OLLAMA},
    "qwen3.5:2b":         {"tier": "S", "params": "2B",    "host": REMOTE_OLLAMA},
    "granite4:3b":        {"tier": "S", "params": "3B",    "host": REMOTE_OLLAMA},
    # Tier M — 4B-14B
    "qwen3.5:4b":         {"tier": "M", "params": "4B",    "host": REMOTE_OLLAMA},
    "qwen2.5:7b-instruct-q4_K_M": {"tier": "M", "params": "7B", "host": REMOTE_OLLAMA},
    "qwen3.5:9b":         {"tier": "M", "params": "9B",    "host": REMOTE_OLLAMA},
    "qwen2.5:14b-instruct-q4_K_M": {"tier": "M", "params": "14B", "host": REMOTE_OLLAMA},
    # Tier L — 14B-32B
    "gpt-oss:20b":        {"tier": "L", "params": "20B",   "host": LOCAL_OLLAMA},
    "qwen3.5:27b-nothink": {"tier": "L", "params": "27B",  "host": REMOTE_OLLAMA},
    "qwen2.5-coder:32b-instruct-q4_K_M": {"tier": "L", "params": "32B", "host": REMOTE_OLLAMA},
    # Tier XL — 35B+
    "qwen3.5:35b":        {"tier": "XL", "params": "35B",  "host": REMOTE_OLLAMA},
}

# ---------------------------------------------------------------------------
# Tool Registry — 30 realistic tools across 6 categories
# ---------------------------------------------------------------------------

TOOLS = [
    # =========================================================================
    # Filesystem (10)
    # =========================================================================
    {"name": "file_read",       "category": "filesystem", "description_short": "Read file", "description_full": "Read the contents of a file from the local filesystem. Supports text files with optional line numbering, offset, and encoding control.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "encoding": "string", "line_numbers": "boolean", "offset": "integer", "limit": "integer"}},
    {"name": "file_write",      "category": "filesystem", "description_short": "Write file", "description_full": "Write content to a file on the local filesystem. Creates parent directories if needed. Overwrites existing content.", "params_minimal": {"path": "string", "content": "string"}, "params_full": {"path": "string", "content": "string", "encoding": "string", "create_dirs": "boolean", "backup": "boolean"}},
    {"name": "file_delete",     "category": "filesystem", "description_short": "Delete file", "description_full": "Delete a file or empty directory from the local filesystem. Supports recursive deletion with confirmation.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "recursive": "boolean", "force": "boolean"}},
    {"name": "file_list",       "category": "filesystem", "description_short": "List directory", "description_full": "List files and directories in a given path. Returns names, sizes, types, and modification times.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "recursive": "boolean", "pattern": "string", "include_hidden": "boolean", "sort_by": "string"}},
    {"name": "file_search",     "category": "filesystem", "description_short": "Search files", "description_full": "Search for files by name pattern or glob across the filesystem. Returns matching paths with metadata.", "params_minimal": {"pattern": "string"}, "params_full": {"pattern": "string", "path": "string", "max_depth": "integer", "type": "string", "size_filter": "string"}},
    {"name": "file_move",       "category": "filesystem", "description_short": "Move file", "description_full": "Move or rename a file or directory. Creates destination directories if needed.", "params_minimal": {"source": "string", "destination": "string"}, "params_full": {"source": "string", "destination": "string", "overwrite": "boolean"}},
    {"name": "file_copy",       "category": "filesystem", "description_short": "Copy file", "description_full": "Copy a file or directory to a new location. Supports recursive copy for directories.", "params_minimal": {"source": "string", "destination": "string"}, "params_full": {"source": "string", "destination": "string", "recursive": "boolean", "overwrite": "boolean"}},
    {"name": "file_info",       "category": "filesystem", "description_short": "File info", "description_full": "Get detailed metadata about a file: size, permissions, owner, modification time, MIME type.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "follow_symlinks": "boolean"}},
    {"name": "file_chmod",      "category": "filesystem", "description_short": "Set permissions", "description_full": "Change file or directory permissions using numeric or symbolic mode notation.", "params_minimal": {"path": "string", "mode": "string"}, "params_full": {"path": "string", "mode": "string", "recursive": "boolean"}},
    {"name": "file_watch",      "category": "filesystem", "description_short": "Watch for changes", "description_full": "Watch a file or directory for changes. Triggers callback on create, modify, or delete events.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "events": "array", "recursive": "boolean", "debounce_ms": "integer"}},

    # =========================================================================
    # Code & Git (12)
    # =========================================================================
    {"name": "grep",            "category": "code", "description_short": "Search code", "description_full": "Search file contents using regex patterns. Supports context lines, file type filtering, and multiple output modes.", "params_minimal": {"pattern": "string"}, "params_full": {"pattern": "string", "path": "string", "glob": "string", "context": "integer", "case_insensitive": "boolean", "max_results": "integer"}},
    {"name": "code_edit",       "category": "code", "description_short": "Edit code", "description_full": "Replace a specific string in a source file with a new string. Finds exact match and performs the substitution.", "params_minimal": {"file": "string", "old": "string", "new": "string"}, "params_full": {"file": "string", "old": "string", "new": "string", "replace_all": "boolean"}},
    {"name": "code_format",     "category": "code", "description_short": "Format code", "description_full": "Auto-format source code using language-appropriate formatters (black, prettier, gofmt, rustfmt).", "params_minimal": {"file": "string"}, "params_full": {"file": "string", "language": "string", "config": "string"}},
    {"name": "code_lint",       "category": "code", "description_short": "Lint code", "description_full": "Run linting checks on source code. Returns warnings and errors with line numbers and suggested fixes.", "params_minimal": {"file": "string"}, "params_full": {"file": "string", "language": "string", "fix": "boolean", "rules": "array"}},
    {"name": "code_symbols",    "category": "code", "description_short": "List symbols", "description_full": "Extract function, class, and variable definitions from source code. Returns symbol tree with line numbers.", "params_minimal": {"file": "string"}, "params_full": {"file": "string", "language": "string", "kind": "string"}},
    {"name": "run_command",     "category": "code", "description_short": "Run shell command", "description_full": "Execute a shell command and return its stdout, stderr, and exit code. Supports timeout and working directory.", "params_minimal": {"command": "string"}, "params_full": {"command": "string", "cwd": "string", "timeout": "integer", "env": "object"}},
    {"name": "run_test",        "category": "code", "description_short": "Run tests", "description_full": "Run test suites using the project's test framework (pytest, jest, go test). Returns pass/fail summary.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "framework": "string", "filter": "string", "verbose": "boolean", "coverage": "boolean"}},
    {"name": "git_status",      "category": "code", "description_short": "Git status", "description_full": "Show the working tree status of a git repository. Lists modified, staged, and untracked files.", "params_minimal": {}, "params_full": {"short": "boolean", "branch": "boolean"}},
    {"name": "git_commit",      "category": "code", "description_short": "Git commit", "description_full": "Create a git commit with a message. Optionally stage files before committing.", "params_minimal": {"message": "string"}, "params_full": {"message": "string", "files": "array", "all": "boolean", "amend": "boolean"}},
    {"name": "git_diff",        "category": "code", "description_short": "Git diff", "description_full": "Show changes between commits, working tree, and staging area. Supports file filtering and stat-only mode.", "params_minimal": {}, "params_full": {"ref": "string", "file": "string", "staged": "boolean", "stat": "boolean"}},
    {"name": "git_log",         "category": "code", "description_short": "Git log", "description_full": "Show commit history with messages, authors, and dates. Supports filtering by author, path, and date range.", "params_minimal": {}, "params_full": {"limit": "integer", "author": "string", "path": "string", "since": "string", "oneline": "boolean"}},
    {"name": "git_branch",      "category": "code", "description_short": "Git branch", "description_full": "List, create, or switch branches. Supports creating from a specific ref and deleting merged branches.", "params_minimal": {}, "params_full": {"name": "string", "create": "boolean", "delete": "boolean", "switch": "boolean", "from_ref": "string"}},

    # =========================================================================
    # Web & API (10)
    # =========================================================================
    {"name": "web_fetch",       "category": "web", "description_short": "Fetch webpage", "description_full": "Fetch content from a URL and return it as clean markdown text. Strips ads, navigation, and boilerplate.", "params_minimal": {"url": "string"}, "params_full": {"url": "string", "selector": "string", "timeout": "integer", "headers": "object"}},
    {"name": "web_search",      "category": "web", "description_short": "Web search", "description_full": "Search the web using a search engine and return the top results with titles, URLs, and snippets.", "params_minimal": {"query": "string"}, "params_full": {"query": "string", "num_results": "integer", "site": "string", "date_range": "string"}},
    {"name": "web_screenshot",  "category": "web", "description_short": "Screenshot page", "description_full": "Take a screenshot of a webpage and return it as a PNG image. Supports viewport size and full-page capture.", "params_minimal": {"url": "string"}, "params_full": {"url": "string", "width": "integer", "height": "integer", "full_page": "boolean"}},
    {"name": "api_request",     "category": "web", "description_short": "HTTP request", "description_full": "Make an HTTP request (GET, POST, PUT, DELETE) to an API endpoint. Returns status code, headers, and body.", "params_minimal": {"url": "string", "method": "string"}, "params_full": {"url": "string", "method": "string", "headers": "object", "body": "string", "auth": "string", "timeout": "integer"}},
    {"name": "download_file",   "category": "web", "description_short": "Download file", "description_full": "Download a file from a URL and save it to a local path. Supports progress tracking and resume.", "params_minimal": {"url": "string", "path": "string"}, "params_full": {"url": "string", "path": "string", "overwrite": "boolean", "timeout": "integer"}},
    {"name": "rss_read",        "category": "web", "description_short": "Read RSS feed", "description_full": "Parse an RSS or Atom feed URL and return the latest entries with titles, links, and summaries.", "params_minimal": {"url": "string"}, "params_full": {"url": "string", "limit": "integer", "since": "string"}},
    {"name": "dns_lookup",      "category": "web", "description_short": "DNS lookup", "description_full": "Resolve a domain name to IP addresses. Supports A, AAAA, MX, TXT, CNAME, and NS record types.", "params_minimal": {"domain": "string"}, "params_full": {"domain": "string", "type": "string", "server": "string"}},
    {"name": "ping",            "category": "web", "description_short": "Ping host", "description_full": "Send ICMP ping to a host and report latency, packet loss, and reachability status.", "params_minimal": {"host": "string"}, "params_full": {"host": "string", "count": "integer", "timeout": "integer"}},
    {"name": "ssl_check",       "category": "web", "description_short": "Check SSL cert", "description_full": "Check the SSL/TLS certificate of a domain. Returns expiry date, issuer, chain validity, and grade.", "params_minimal": {"domain": "string"}, "params_full": {"domain": "string", "port": "integer"}},
    {"name": "whois_lookup",    "category": "web", "description_short": "WHOIS lookup", "description_full": "Query WHOIS records for a domain. Returns registrar, creation date, expiry, and nameservers.", "params_minimal": {"domain": "string"}, "params_full": {"domain": "string"}},

    # =========================================================================
    # Data & Database (10)
    # =========================================================================
    {"name": "json_query",      "category": "data", "description_short": "Query JSON", "description_full": "Query and transform JSON data using JMESPath or JSONPath expressions. Supports filtering, projection, and aggregation.", "params_minimal": {"data": "string", "query": "string"}, "params_full": {"data": "string", "query": "string", "language": "string", "pretty": "boolean"}},
    {"name": "csv_read",        "category": "data", "description_short": "Read CSV", "description_full": "Read and parse a CSV file. Returns structured data with column names, row count, and optional filtering.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "delimiter": "string", "columns": "array", "filter": "string", "limit": "integer"}},
    {"name": "csv_write",       "category": "data", "description_short": "Write CSV", "description_full": "Write structured data to a CSV file. Supports custom delimiters, headers, and append mode.", "params_minimal": {"path": "string", "data": "array"}, "params_full": {"path": "string", "data": "array", "delimiter": "string", "headers": "array", "append": "boolean"}},
    {"name": "db_query",        "category": "data", "description_short": "Query database", "description_full": "Execute a SQL query against a connected database (SQLite, PostgreSQL). Returns rows as JSON.", "params_minimal": {"query": "string"}, "params_full": {"query": "string", "database": "string", "params": "array", "limit": "integer", "timeout": "integer"}},
    {"name": "db_schema",       "category": "data", "description_short": "Database schema", "description_full": "List tables, columns, types, and indexes of a connected database. Supports SQLite and PostgreSQL.", "params_minimal": {}, "params_full": {"database": "string", "table": "string"}},
    {"name": "calc",            "category": "data", "description_short": "Calculate", "description_full": "Evaluate a mathematical expression or perform unit conversions. Supports arithmetic, trigonometry, and statistics.", "params_minimal": {"expression": "string"}, "params_full": {"expression": "string", "precision": "integer", "format": "string"}},
    {"name": "data_transform",  "category": "data", "description_short": "Transform data", "description_full": "Transform data between formats: JSON to CSV, XML to JSON, YAML to JSON, and more. Supports mapping and filtering.", "params_minimal": {"input": "string", "from": "string", "to": "string"}, "params_full": {"input": "string", "from": "string", "to": "string", "mapping": "object", "filter": "string"}},
    {"name": "regex_extract",   "category": "data", "description_short": "Regex extract", "description_full": "Extract data from text using regular expressions. Returns all matches with named capture groups.", "params_minimal": {"text": "string", "pattern": "string"}, "params_full": {"text": "string", "pattern": "string", "flags": "string", "limit": "integer"}},
    {"name": "hash_compute",    "category": "data", "description_short": "Compute hash", "description_full": "Compute cryptographic hash of text or file. Supports MD5, SHA-1, SHA-256, SHA-512, and BLAKE2.", "params_minimal": {"input": "string"}, "params_full": {"input": "string", "algorithm": "string", "file": "string"}},
    {"name": "base64_convert",  "category": "data", "description_short": "Base64 encode/decode", "description_full": "Encode or decode data using Base64. Supports standard and URL-safe variants.", "params_minimal": {"input": "string", "action": "string"}, "params_full": {"input": "string", "action": "string", "url_safe": "boolean"}},

    # =========================================================================
    # Communication & Messaging (10)
    # =========================================================================
    {"name": "send_message",    "category": "communication", "description_short": "Send message", "description_full": "Send a message to a user or channel via configured messaging platform (Slack, Telegram, Discord, etc.).", "params_minimal": {"to": "string", "text": "string"}, "params_full": {"to": "string", "text": "string", "channel": "string", "format": "string", "reply_to": "string"}},
    {"name": "send_email",      "category": "communication", "description_short": "Send email", "description_full": "Compose and send an email. Supports HTML body, attachments, CC/BCC, and reply threading.", "params_minimal": {"to": "string", "subject": "string", "body": "string"}, "params_full": {"to": "string", "subject": "string", "body": "string", "cc": "array", "bcc": "array", "attachments": "array", "html": "boolean"}},
    {"name": "calendar_event",  "category": "communication", "description_short": "Create event", "description_full": "Create a calendar event with title, date, time, duration, and optional attendees and location.", "params_minimal": {"title": "string", "date": "string"}, "params_full": {"title": "string", "date": "string", "time": "string", "duration": "integer", "location": "string", "attendees": "array", "reminder": "integer"}},
    {"name": "calendar_list",   "category": "communication", "description_short": "List events", "description_full": "List upcoming calendar events. Supports date range filtering and calendar selection.", "params_minimal": {}, "params_full": {"from": "string", "to": "string", "calendar": "string", "limit": "integer"}},
    {"name": "set_reminder",    "category": "communication", "description_short": "Set reminder", "description_full": "Set a reminder for a specific time or relative delay. Delivers via configured notification channel.", "params_minimal": {"text": "string", "when": "string"}, "params_full": {"text": "string", "when": "string", "repeat": "string", "channel": "string", "priority": "string"}},
    {"name": "read_inbox",      "category": "communication", "description_short": "Read inbox", "description_full": "Read recent messages from email inbox or messaging platform. Supports filtering by sender, subject, and date.", "params_minimal": {}, "params_full": {"limit": "integer", "sender": "string", "subject": "string", "unread_only": "boolean", "since": "string"}},
    {"name": "contact_lookup",  "category": "communication", "description_short": "Lookup contact", "description_full": "Search contacts by name, email, or phone. Returns matching entries with all available fields.", "params_minimal": {"query": "string"}, "params_full": {"query": "string", "field": "string", "limit": "integer"}},
    {"name": "translate",       "category": "communication", "description_short": "Translate text", "description_full": "Translate text between languages. Auto-detects source language if not specified.", "params_minimal": {"text": "string", "to": "string"}, "params_full": {"text": "string", "to": "string", "from": "string"}},
    {"name": "summarize_thread","category": "communication", "description_short": "Summarize thread", "description_full": "Summarize a message thread or email chain. Returns key points, decisions, and action items.", "params_minimal": {"thread_id": "string"}, "params_full": {"thread_id": "string", "channel": "string", "max_length": "integer"}},
    {"name": "notify",          "category": "communication", "description_short": "Send notification", "description_full": "Push a notification to the user's device or desktop. Supports priority levels and action buttons.", "params_minimal": {"title": "string", "body": "string"}, "params_full": {"title": "string", "body": "string", "priority": "string", "actions": "array", "sound": "boolean"}},

    # =========================================================================
    # System & Infrastructure (10)
    # =========================================================================
    {"name": "system_info",     "category": "system", "description_short": "System info", "description_full": "Get system information: CPU usage, memory, disk space, OS version, uptime, and running processes.", "params_minimal": {}, "params_full": {"include_processes": "boolean", "include_network": "boolean"}},
    {"name": "process_list",    "category": "system", "description_short": "List processes", "description_full": "List running processes with PID, name, CPU%, memory usage, and command line arguments.", "params_minimal": {}, "params_full": {"sort_by": "string", "filter": "string", "limit": "integer"}},
    {"name": "process_kill",    "category": "system", "description_short": "Kill process", "description_full": "Terminate a running process by PID or name. Supports graceful and forceful termination.", "params_minimal": {"pid": "integer"}, "params_full": {"pid": "integer", "name": "string", "signal": "string", "force": "boolean"}},
    {"name": "env_get",         "category": "system", "description_short": "Get env var", "description_full": "Get the value of an environment variable. Returns null if not set.", "params_minimal": {"name": "string"}, "params_full": {"name": "string", "default": "string"}},
    {"name": "env_set",         "category": "system", "description_short": "Set env var", "description_full": "Set an environment variable for the current session. Does not persist across restarts.", "params_minimal": {"name": "string", "value": "string"}, "params_full": {"name": "string", "value": "string", "persist": "boolean"}},
    {"name": "cron_schedule",   "category": "system", "description_short": "Schedule task", "description_full": "Create a scheduled task using cron syntax. Supports one-time and recurring schedules with timezone.", "params_minimal": {"schedule": "string", "command": "string"}, "params_full": {"schedule": "string", "command": "string", "name": "string", "timezone": "string", "enabled": "boolean"}},
    {"name": "cron_list",       "category": "system", "description_short": "List scheduled tasks", "description_full": "List all scheduled cron tasks with their next run time, status, and last execution result.", "params_minimal": {}, "params_full": {"status": "string"}},
    {"name": "service_status",  "category": "system", "description_short": "Service status", "description_full": "Check the status of a system service (systemd, launchd). Returns running state, PID, and logs.", "params_minimal": {"name": "string"}, "params_full": {"name": "string", "logs": "boolean", "lines": "integer"}},
    {"name": "disk_usage",      "category": "system", "description_short": "Disk usage", "description_full": "Show disk usage for a path or all mounted volumes. Returns total, used, free space and percentage.", "params_minimal": {}, "params_full": {"path": "string", "human_readable": "boolean"}},
    {"name": "network_info",    "category": "system", "description_short": "Network info", "description_full": "Show network interfaces, IP addresses, active connections, and listening ports.", "params_minimal": {}, "params_full": {"interface": "string", "connections": "boolean", "listening": "boolean"}},

    # =========================================================================
    # DevOps & Cloud (10)
    # =========================================================================
    {"name": "docker_ps",       "category": "devops", "description_short": "List containers", "description_full": "List Docker containers with status, ports, image, and resource usage. Supports filtering.", "params_minimal": {}, "params_full": {"all": "boolean", "filter": "string", "format": "string"}},
    {"name": "docker_logs",     "category": "devops", "description_short": "Container logs", "description_full": "Fetch logs from a Docker container. Supports tail, follow, timestamps, and since filters.", "params_minimal": {"container": "string"}, "params_full": {"container": "string", "tail": "integer", "since": "string", "timestamps": "boolean"}},
    {"name": "docker_exec",     "category": "devops", "description_short": "Exec in container", "description_full": "Execute a command inside a running Docker container. Returns stdout and exit code.", "params_minimal": {"container": "string", "command": "string"}, "params_full": {"container": "string", "command": "string", "user": "string", "workdir": "string"}},
    {"name": "docker_build",    "category": "devops", "description_short": "Build image", "description_full": "Build a Docker image from a Dockerfile. Supports build args, tags, and multi-stage builds.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "tag": "string", "file": "string", "args": "object", "no_cache": "boolean"}},
    {"name": "k8s_get",         "category": "devops", "description_short": "Get K8s resource", "description_full": "Get Kubernetes resources (pods, services, deployments). Supports namespace and label filtering.", "params_minimal": {"kind": "string"}, "params_full": {"kind": "string", "name": "string", "namespace": "string", "labels": "string", "output": "string"}},
    {"name": "k8s_logs",        "category": "devops", "description_short": "Pod logs", "description_full": "Fetch logs from a Kubernetes pod. Supports container selection, tail, and previous container logs.", "params_minimal": {"pod": "string"}, "params_full": {"pod": "string", "namespace": "string", "container": "string", "tail": "integer", "previous": "boolean"}},
    {"name": "ssh_exec",        "category": "devops", "description_short": "SSH command", "description_full": "Execute a command on a remote host via SSH. Returns stdout, stderr, and exit code.", "params_minimal": {"host": "string", "command": "string"}, "params_full": {"host": "string", "command": "string", "user": "string", "key": "string", "port": "integer", "timeout": "integer"}},
    {"name": "deploy",          "category": "devops", "description_short": "Deploy app", "description_full": "Deploy an application to the configured target environment. Supports rolling updates and rollback.", "params_minimal": {"app": "string", "version": "string"}, "params_full": {"app": "string", "version": "string", "environment": "string", "strategy": "string", "dry_run": "boolean"}},
    {"name": "terraform_plan",  "category": "devops", "description_short": "Terraform plan", "description_full": "Run terraform plan to preview infrastructure changes. Returns added, changed, and destroyed resources.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "var_file": "string", "target": "string", "out": "string"}},
    {"name": "ansible_run",     "category": "devops", "description_short": "Run Ansible", "description_full": "Execute an Ansible playbook against target hosts. Returns task results and changed/failed summary.", "params_minimal": {"playbook": "string"}, "params_full": {"playbook": "string", "inventory": "string", "limit": "string", "tags": "string", "check": "boolean"}},

    # =========================================================================
    # AI & Memory (8)
    # =========================================================================
    {"name": "memory_store",    "category": "ai", "description_short": "Store memory", "description_full": "Store a fact, decision, or preference in persistent cognitive memory. Supports importance and domain tagging.", "params_minimal": {"text": "string"}, "params_full": {"text": "string", "importance": "number", "domain": "string", "source": "string"}},
    {"name": "memory_recall",   "category": "ai", "description_short": "Recall memory", "description_full": "Search cognitive memory by semantic similarity. Returns relevant memories with confidence scores.", "params_minimal": {"query": "string"}, "params_full": {"query": "string", "limit": "integer", "domain": "string", "min_score": "number"}},
    {"name": "memory_forget",   "category": "ai", "description_short": "Forget memory", "description_full": "Delete a specific memory by ID. Removes it from recall results permanently.", "params_minimal": {"id": "string"}, "params_full": {"id": "string", "reason": "string"}},
    {"name": "embed_text",      "category": "ai", "description_short": "Generate embedding", "description_full": "Generate a vector embedding for text using the configured embedding model. Returns float array.", "params_minimal": {"text": "string"}, "params_full": {"text": "string", "model": "string"}},
    {"name": "classify_text",   "category": "ai", "description_short": "Classify text", "description_full": "Classify text into predefined categories using zero-shot or few-shot classification.", "params_minimal": {"text": "string", "labels": "array"}, "params_full": {"text": "string", "labels": "array", "multi_label": "boolean"}},
    {"name": "sentiment",       "category": "ai", "description_short": "Sentiment analysis", "description_full": "Analyze the sentiment of text. Returns positive/negative/neutral with confidence score.", "params_minimal": {"text": "string"}, "params_full": {"text": "string", "granularity": "string"}},
    {"name": "ocr_image",       "category": "ai", "description_short": "OCR image", "description_full": "Extract text from an image using optical character recognition. Supports multiple languages.", "params_minimal": {"path": "string"}, "params_full": {"path": "string", "language": "string", "format": "string"}},
    {"name": "tts_speak",       "category": "ai", "description_short": "Text to speech", "description_full": "Convert text to speech audio. Supports multiple voices, speeds, and output formats.", "params_minimal": {"text": "string"}, "params_full": {"text": "string", "voice": "string", "speed": "number", "format": "string"}},
]

# ---------------------------------------------------------------------------
# Test Prompts — 50 prompts with expected tool + basic param validation
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # Filesystem
    {"prompt": "Read the file config.yaml", "expected_tool": "file_read", "expected_params": ["path"], "category": "filesystem"},
    {"prompt": "Show me what's in /etc/hosts", "expected_tool": "file_read", "expected_params": ["path"], "category": "filesystem"},
    {"prompt": "Write 'hello world' to output.txt", "expected_tool": "file_write", "expected_params": ["path", "content"], "category": "filesystem"},
    {"prompt": "Save this text to a new file called notes.md: Meeting at 3pm", "expected_tool": "file_write", "expected_params": ["path", "content"], "category": "filesystem"},
    {"prompt": "Delete the temporary file /tmp/cache.json", "expected_tool": "file_delete", "expected_params": ["path"], "category": "filesystem"},
    {"prompt": "List all files in the src directory", "expected_tool": "file_list", "expected_params": ["path"], "category": "filesystem"},
    {"prompt": "What files are in the current directory?", "expected_tool": "file_list", "expected_params": [], "category": "filesystem"},
    {"prompt": "Find all Python files in this project", "expected_tool": "file_search", "expected_params": ["pattern"], "category": "filesystem"},
    {"prompt": "Search for files named README", "expected_tool": "file_search", "expected_params": ["pattern"], "category": "filesystem"},
    {"prompt": "Rename data.csv to data_backup.csv", "expected_tool": "file_move", "expected_params": ["source", "destination"], "category": "filesystem"},

    # Code
    {"prompt": "Search for TODO comments in the codebase", "expected_tool": "grep", "expected_params": ["pattern"], "category": "code"},
    {"prompt": "Find all occurrences of 'import os' in Python files", "expected_tool": "grep", "expected_params": ["pattern"], "category": "code"},
    {"prompt": "Change the variable name 'old_name' to 'new_name' in app.py", "expected_tool": "code_edit", "expected_params": ["file", "old", "new"], "category": "code"},
    {"prompt": "What's the git status of this repo?", "expected_tool": "git_status", "expected_params": [], "category": "code"},
    {"prompt": "Commit these changes with message 'fix: resolve null pointer'", "expected_tool": "git_commit", "expected_params": ["message"], "category": "code"},
    {"prompt": "Run the test suite with pytest", "expected_tool": "run_command", "expected_params": ["command"], "category": "code"},
    {"prompt": "Execute npm install", "expected_tool": "run_command", "expected_params": ["command"], "category": "code"},
    {"prompt": "Check if the server is running on port 8080", "expected_tool": "run_command", "expected_params": ["command"], "category": "code"},

    # Web
    {"prompt": "Fetch the content of https://example.com", "expected_tool": "web_fetch", "expected_params": ["url"], "category": "web"},
    {"prompt": "Read the documentation page at https://docs.python.org/3/", "expected_tool": "web_fetch", "expected_params": ["url"], "category": "web"},
    {"prompt": "Search the web for Python FastAPI tutorial", "expected_tool": "web_search", "expected_params": ["query"], "category": "web"},
    {"prompt": "Google for best practices in REST API design", "expected_tool": "web_search", "expected_params": ["query"], "category": "web"},
    {"prompt": "Take a screenshot of https://github.com", "expected_tool": "web_screenshot", "expected_params": ["url"], "category": "web"},
    {"prompt": "Make a GET request to https://api.github.com/users/octocat", "expected_tool": "api_request", "expected_params": ["url"], "category": "web"},
    {"prompt": "POST to https://httpbin.org/post with body {'key': 'value'}", "expected_tool": "api_request", "expected_params": ["url", "method"], "category": "web"},
    {"prompt": "Download the file at https://example.com/data.zip and save it to ~/Downloads/", "expected_tool": "download_file", "expected_params": ["url", "path"], "category": "web"},

    # Data
    {"prompt": "Parse the JSON in response.json and extract the 'users' array", "expected_tool": "json_query", "expected_params": ["data", "query"], "category": "data"},
    {"prompt": "Read the CSV file sales_data.csv", "expected_tool": "csv_read", "expected_params": ["path"], "category": "data"},
    {"prompt": "Query the database: SELECT * FROM users WHERE active = true", "expected_tool": "db_query", "expected_params": ["query"], "category": "data"},
    {"prompt": "How many rows are in the orders table?", "expected_tool": "db_query", "expected_params": ["query"], "category": "data"},
    {"prompt": "Calculate 15% of 2499.99", "expected_tool": "calc", "expected_params": ["expression"], "category": "data"},
    {"prompt": "What is the square root of 144?", "expected_tool": "calc", "expected_params": ["expression"], "category": "data"},
    {"prompt": "Convert this JSON to CSV format", "expected_tool": "data_transform", "expected_params": ["from", "to"], "category": "data"},

    # Communication
    {"prompt": "Send a message to Alice saying the deployment is complete", "expected_tool": "send_message", "expected_params": ["to", "text"], "category": "communication"},
    {"prompt": "Message the team channel: servers are back online", "expected_tool": "send_message", "expected_params": ["to", "text"], "category": "communication"},
    {"prompt": "Send an email to bob@company.com about the quarterly report", "expected_tool": "send_email", "expected_params": ["to", "subject"], "category": "communication"},
    {"prompt": "Create a meeting for tomorrow at 2pm called Sprint Planning", "expected_tool": "calendar_event", "expected_params": ["title", "date"], "category": "communication"},
    {"prompt": "Schedule a team sync for next Monday at 10am", "expected_tool": "calendar_event", "expected_params": ["title", "date"], "category": "communication"},
    {"prompt": "Remind me to review the PR in 2 hours", "expected_tool": "set_reminder", "expected_params": ["text", "when"], "category": "communication"},
    {"prompt": "Set a reminder for Friday: submit expense report", "expected_tool": "set_reminder", "expected_params": ["text", "when"], "category": "communication"},
    {"prompt": "Check my email inbox", "expected_tool": "read_inbox", "expected_params": [], "category": "communication"},
    {"prompt": "Show me unread messages from the last hour", "expected_tool": "read_inbox", "expected_params": [], "category": "communication"},

    # System
    {"prompt": "How much disk space is available?", "expected_tool": "system_info", "expected_params": [], "category": "system"},
    {"prompt": "Show me CPU and memory usage", "expected_tool": "system_info", "expected_params": [], "category": "system"},
    {"prompt": "Kill the process with PID 12345", "expected_tool": "process_kill", "expected_params": ["pid"], "category": "system"},
    {"prompt": "What is the DATABASE_URL environment variable?", "expected_tool": "env_get", "expected_params": ["name"], "category": "system"},
    {"prompt": "Get the value of OPENAI_API_KEY", "expected_tool": "env_get", "expected_params": ["name"], "category": "system"},
    {"prompt": "Schedule a backup every day at midnight", "expected_tool": "cron_schedule", "expected_params": ["schedule", "command"], "category": "system"},
    {"prompt": "Run the cleanup script every Sunday at 3am", "expected_tool": "cron_schedule", "expected_params": ["schedule", "command"], "category": "system"},
]

# ---------------------------------------------------------------------------
# Presentation Strategies
# ---------------------------------------------------------------------------

def build_baseline_prompt(tools: list, user_message: str) -> str:
    """Baseline: present ALL tools with full descriptions."""
    tool_block = []
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_full"].items())
        tool_block.append(f"- {t['name']}: {t['description_full']}\n  Parameters: {{{params}}}")
    tool_text = "\n".join(tool_block)

    return f"""You are an AI assistant with access to tools. Pick the single best tool for the user's request and respond with JSON only.

Available tools:
{tool_text}

Respond with ONLY this JSON format, no other text:
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_message}"""


def build_tier_s_prompt(tools: list, user_message: str, top_k: int = 4) -> str:
    """Tier S: MCQ with top-K tools by semantic similarity (YantrikDB)."""
    top = semantic_rank_tools(user_message, tools, top_k=top_k)

    options = []
    labels = ["A", "B", "C", "D", "E"]
    for i, t in enumerate(top):
        params = ", ".join(t["params_minimal"].keys())
        options.append(f"{labels[i]}) {t['name']}: {t['description_short']} ({params})")

    options_text = "\n".join(options)

    return f"""Pick the best tool. Reply with the letter and parameters as JSON.

{options_text}

Reply format: {{"choice": "A", "params": {{"param": "value"}}}}

User: {user_message}"""


def build_tier_l_prompt(tools: list, user_message: str) -> str:
    """Tier L: ALL tools but with medium-length descriptions (not full, not short)."""
    tool_block = []
    for t in tools:
        # Use short description + minimal params — keeps all tools visible
        params = ", ".join(f"{k}: {v}" for k, v in t["params_minimal"].items())
        tool_block.append(f"- {t['name']}: {t['description_short']}. Params: {{{params}}}")
    tool_text = "\n".join(tool_block)

    return f"""You are an AI assistant with access to tools. Pick the single best tool for the user's request and respond with JSON only.

Available tools:
{tool_text}

Respond with ONLY this JSON format, no other text:
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_message}"""


def build_tier_m_prompt(tools: list, user_message: str, top_k: int = 8) -> str:
    """Tier M: condensed descriptions, top-K tools by semantic similarity."""
    top = semantic_rank_tools(user_message, tools, top_k=top_k)

    tool_block = []
    for t in top:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_minimal"].items())
        tool_block.append(f"- {t['name']}: {t['description_short']} ({params})")
    tool_text = "\n".join(tool_block)

    return f"""You have these tools. Pick the best one for the user's request.

Tools:
{tool_text}

Reply with JSON only: {{"tool": "tool_name", "params": {{"param": "value"}}}}

User: {user_message}"""


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def ollama_generate(host: str, model: str, prompt: str, timeout: int = 120) -> dict:
    """Call Ollama generate API. Returns parsed response."""
    url = f"{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
        }
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        elapsed = time.monotonic() - t0
        return {
            "response": body.get("response", ""),
            "prompt_eval_count": body.get("prompt_eval_count", 0),
            "eval_count": body.get("eval_count", 0),
            "total_duration_ns": body.get("total_duration", 0),
            "elapsed_s": elapsed,
            "error": None,
        }
    except Exception as e:
        return {
            "response": "",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "total_duration_ns": 0,
            "elapsed_s": 0,
            "error": str(e),
        }


def extract_tool_call(response_text: str, mode: str = "direct") -> dict:
    """Parse tool call from model response. Returns {tool, params, raw}."""
    text = response_text.strip()

    # Try to find JSON in the response
    # Look for the first { ... } block
    start = text.find("{")
    if start == -1:
        return {"tool": None, "params": {}, "raw": text, "parse_error": "no_json"}

    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"tool": None, "params": {}, "raw": text, "parse_error": "invalid_json"}

    if mode == "mcq":
        # MCQ mode: choice field maps to tool
        choice = parsed.get("choice", "")
        return {"tool": choice, "params": parsed.get("params", {}), "raw": text, "parse_error": None}
    else:
        tool = parsed.get("tool", parsed.get("name", None))
        params = parsed.get("params", parsed.get("parameters", parsed.get("arguments", {})))
        return {"tool": tool, "params": params if isinstance(params, dict) else {}, "raw": text, "parse_error": None}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_result(expected_tool: str, expected_params: list, result: dict,
                 mode: str = "direct", mcq_options: list = None) -> dict:
    """Score a single result. Returns metrics dict."""
    tool = result.get("tool")
    params = result.get("params", {})

    # Tool selection accuracy
    if mode == "mcq" and mcq_options:
        # Map choice letter back to tool name
        labels = ["A", "B", "C", "D", "E"]
        if tool and tool.upper() in labels:
            idx = labels.index(tool.upper())
            if idx < len(mcq_options):
                tool = mcq_options[idx]
            else:
                tool = None

    tool_correct = (tool == expected_tool) if tool else False

    # Param accuracy: check if expected params are present
    if expected_params:
        params_present = sum(1 for p in expected_params if p in (params or {}))
        param_accuracy = params_present / len(expected_params)
    else:
        param_accuracy = 1.0  # No params expected

    return {
        "tool_correct": tool_correct,
        "tool_selected": tool,
        "param_accuracy": param_accuracy,
        "parse_error": result.get("parse_error"),
        "format_valid": result.get("parse_error") is None and tool is not None,
    }


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model: str
    tier: str
    params: str
    strategy: str  # "baseline" or "tier_adapted"
    prompt_idx: int
    user_prompt: str
    expected_tool: str
    selected_tool: Optional[str]
    tool_correct: bool
    param_accuracy: float
    format_valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: Optional[str]


def run_benchmark(models: dict = None, prompts: list = None, output_path: str = None):
    """Run the full benchmark suite."""
    if models is None:
        models = MODELS
    if prompts is None:
        prompts = TEST_PROMPTS
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "results.jsonl")

    results = []
    total = len(models) * len(prompts) * 2  # 2 strategies per model
    done = 0

    print(f"\n{'='*70}")
    print(f"  TIER-BASED TOOL ROUTING BENCHMARK")
    print(f"  Models: {len(models)}  |  Prompts: {len(prompts)}  |  Total runs: {total}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*70}\n")

    for model_name, model_info in models.items():
        tier = model_info["tier"]
        host = model_info["host"]
        params_str = model_info["params"]

        print(f"\n--- {model_name} (Tier {tier}, {params_str}) @ {host} ---")

        for pidx, test in enumerate(prompts):
            user_msg = test["prompt"]
            expected = test["expected_tool"]
            expected_params = test["expected_params"]

            # === Strategy 1: Baseline (all tools, full descriptions) ===
            baseline_prompt = build_baseline_prompt(TOOLS, user_msg)
            resp = ollama_generate(host, model_name, baseline_prompt)

            if resp["error"]:
                br = BenchmarkResult(
                    model=model_name, tier=tier, params=params_str,
                    strategy="baseline", prompt_idx=pidx, user_prompt=user_msg,
                    expected_tool=expected, selected_tool=None, tool_correct=False,
                    param_accuracy=0.0, format_valid=False,
                    prompt_tokens=0, completion_tokens=0, latency_s=0,
                    error=resp["error"]
                )
            else:
                parsed = extract_tool_call(resp["response"])
                scored = score_result(expected, expected_params, parsed)
                br = BenchmarkResult(
                    model=model_name, tier=tier, params=params_str,
                    strategy="baseline", prompt_idx=pidx, user_prompt=user_msg,
                    expected_tool=expected, selected_tool=scored["tool_selected"],
                    tool_correct=scored["tool_correct"],
                    param_accuracy=scored["param_accuracy"],
                    format_valid=scored["format_valid"],
                    prompt_tokens=resp["prompt_eval_count"],
                    completion_tokens=resp["eval_count"],
                    latency_s=resp["elapsed_s"],
                    error=None
                )
            results.append(br)
            done += 1

            # === Strategy 2: Tier-adapted ===
            if tier == "S":
                adapted_prompt = build_tier_s_prompt(TOOLS, user_msg, top_k=4)
                adapt_mode = "mcq"
                # Build MCQ options list for scoring (must match what build_tier_s_prompt used)
                mcq_opts = [t["name"] for t in semantic_rank_tools(user_msg, TOOLS, top_k=4)]
            elif tier == "M":
                adapted_prompt = build_tier_m_prompt(TOOLS, user_msg, top_k=8)
                adapt_mode = "direct"
                mcq_opts = None
            elif tier == "L":
                # L tier: full tool set but with shorter descriptions
                adapted_prompt = build_tier_l_prompt(TOOLS, user_msg)
                adapt_mode = "direct"
                mcq_opts = None
            else:
                # XL tier: identical to baseline — large models handle full tools
                adapted_prompt = build_baseline_prompt(TOOLS, user_msg)
                adapt_mode = "direct"
                mcq_opts = None

            resp2 = ollama_generate(host, model_name, adapted_prompt)

            if resp2["error"]:
                ar = BenchmarkResult(
                    model=model_name, tier=tier, params=params_str,
                    strategy="tier_adapted", prompt_idx=pidx, user_prompt=user_msg,
                    expected_tool=expected, selected_tool=None, tool_correct=False,
                    param_accuracy=0.0, format_valid=False,
                    prompt_tokens=0, completion_tokens=0, latency_s=0,
                    error=resp2["error"]
                )
            else:
                parsed2 = extract_tool_call(resp2["response"], mode=adapt_mode)
                scored2 = score_result(expected, expected_params, parsed2,
                                       mode=adapt_mode, mcq_options=mcq_opts)
                ar = BenchmarkResult(
                    model=model_name, tier=tier, params=params_str,
                    strategy="tier_adapted", prompt_idx=pidx, user_prompt=user_msg,
                    expected_tool=expected, selected_tool=scored2["tool_selected"],
                    tool_correct=scored2["tool_correct"],
                    param_accuracy=scored2["param_accuracy"],
                    format_valid=scored2["format_valid"],
                    prompt_tokens=resp2["prompt_eval_count"],
                    completion_tokens=resp2["eval_count"],
                    latency_s=resp2["elapsed_s"],
                    error=None
                )
            results.append(ar)
            done += 1

            # Progress
            b_icon = "+" if br.tool_correct else "-"
            a_icon = "+" if ar.tool_correct else "-"
            sys.stdout.write(f"\r  [{done}/{total}] {model_name}: baseline={b_icon} tier={a_icon} | {user_msg[:40]}...")
            sys.stdout.flush()

        print()

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    print(f"\nResults written to {output_path}")
    print(f"Total results: {len(results)}")

    # Print summary
    print_summary(results)

    return results


def print_summary(results: list):
    """Print a summary table of benchmark results."""
    print(f"\n{'='*90}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*90}")

    # Group by model + strategy
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if isinstance(r, BenchmarkResult):
            groups[(r.model, r.tier, r.params, r.strategy)].append(r)
        else:
            d = r
            groups[(d["model"], d["tier"], d["params"], d["strategy"])].append(d)

    # Header
    print(f"\n{'Model':<35} {'Tier':>4} {'Strategy':<14} {'Accuracy':>8} {'Format':>7} {'Tokens':>7} {'Latency':>8}")
    print("-" * 90)

    prev_model = None
    for (model, tier, params, strategy), items in sorted(groups.items()):
        if prev_model and prev_model != model:
            print("-" * 90)
        prev_model = model

        if isinstance(items[0], BenchmarkResult):
            n = len(items)
            acc = sum(1 for i in items if i.tool_correct) / n * 100
            fmt = sum(1 for i in items if i.format_valid) / n * 100
            tokens = statistics.mean([i.prompt_tokens for i in items if i.prompt_tokens > 0]) if any(i.prompt_tokens > 0 for i in items) else 0
            lat = statistics.mean([i.latency_s for i in items if i.latency_s > 0]) if any(i.latency_s > 0 for i in items) else 0
        else:
            n = len(items)
            acc = sum(1 for i in items if i["tool_correct"]) / n * 100
            fmt = sum(1 for i in items if i["format_valid"]) / n * 100
            tokens = statistics.mean([i["prompt_tokens"] for i in items if i["prompt_tokens"] > 0]) if any(i["prompt_tokens"] > 0 for i in items) else 0
            lat = statistics.mean([i["latency_s"] for i in items if i["latency_s"] > 0]) if any(i["latency_s"] > 0 for i in items) else 0

        label = f"{model} ({params})"
        print(f"{label:<35} {tier:>4} {strategy:<14} {acc:>7.1f}% {fmt:>6.1f}% {tokens:>6.0f}t {lat:>7.2f}s")

    # Aggregate by tier + strategy
    print(f"\n{'='*90}")
    print(f"  AGGREGATE BY TIER")
    print(f"{'='*90}")
    print(f"\n{'Tier':<6} {'Strategy':<14} {'Accuracy':>8} {'Format':>7} {'Avg Tokens':>10} {'Avg Latency':>11}")
    print("-" * 60)

    tier_groups = defaultdict(list)
    for r in results:
        if isinstance(r, BenchmarkResult):
            tier_groups[(r.tier, r.strategy)].append(r)
        else:
            tier_groups[(r["tier"], r["strategy"])].append(r)

    for (tier, strategy), items in sorted(tier_groups.items()):
        if isinstance(items[0], BenchmarkResult):
            n = len(items)
            acc = sum(1 for i in items if i.tool_correct) / n * 100
            fmt = sum(1 for i in items if i.format_valid) / n * 100
            tokens = statistics.mean([i.prompt_tokens for i in items if i.prompt_tokens > 0]) if any(i.prompt_tokens > 0 for i in items) else 0
            lat = statistics.mean([i.latency_s for i in items if i.latency_s > 0]) if any(i.latency_s > 0 for i in items) else 0
        else:
            n = len(items)
            acc = sum(1 for i in items if i["tool_correct"]) / n * 100
            fmt = sum(1 for i in items if i["format_valid"]) / n * 100
            tokens = statistics.mean([i["prompt_tokens"] for i in items if i["prompt_tokens"] > 0]) if any(i["prompt_tokens"] > 0 for i in items) else 0
            lat = statistics.mean([i["latency_s"] for i in items if i["latency_s"] > 0]) if any(i["latency_s"] > 0 for i in items) else 0

        print(f"{tier:<6} {strategy:<14} {acc:>7.1f}% {fmt:>6.1f}% {tokens:>9.0f}t {lat:>10.2f}s")

    # Key metric: improvement
    print(f"\n{'='*90}")
    print(f"  KEY FINDINGS")
    print(f"{'='*90}")

    for tier_name in ["S", "M", "L", "XL"]:
        baseline = [r for r in results if (r.tier if isinstance(r, BenchmarkResult) else r["tier"]) == tier_name
                     and (r.strategy if isinstance(r, BenchmarkResult) else r["strategy"]) == "baseline"]
        adapted = [r for r in results if (r.tier if isinstance(r, BenchmarkResult) else r["tier"]) == tier_name
                    and (r.strategy if isinstance(r, BenchmarkResult) else r["strategy"]) == "tier_adapted"]
        if baseline and adapted:
            b_acc = sum(1 for i in baseline if (i.tool_correct if isinstance(i, BenchmarkResult) else i["tool_correct"])) / len(baseline) * 100
            a_acc = sum(1 for i in adapted if (i.tool_correct if isinstance(i, BenchmarkResult) else i["tool_correct"])) / len(adapted) * 100
            b_tok = statistics.mean([(i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) for i in baseline if (i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) > 0]) if any((i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) > 0 for i in baseline) else 1
            a_tok = statistics.mean([(i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) for i in adapted if (i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) > 0]) if any((i.prompt_tokens if isinstance(i, BenchmarkResult) else i["prompt_tokens"]) > 0 for i in adapted) else 1
            delta_acc = a_acc - b_acc
            token_reduction = (1 - a_tok / b_tok) * 100 if b_tok > 0 else 0
            arrow = "+" if delta_acc >= 0 else ""
            print(f"\n  Tier {tier_name}: accuracy {arrow}{delta_acc:.1f}pp ({b_acc:.1f}% -> {a_acc:.1f}%), tokens {token_reduction:.0f}% reduction")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tier-Based Tool Routing Benchmark")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--quick", action="store_true", help="Quick run: 2 models, 10 prompts")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    args = parser.parse_args()

    if args.quick:
        # Quick mode: 1 small + 1 large, 10 prompts
        quick_models = {
            "qwen2.5:1.5b": MODELS["qwen2.5:1.5b"],
            "gpt-oss:20b": MODELS["gpt-oss:20b"],
        }
        run_benchmark(models=quick_models, prompts=TEST_PROMPTS[:10], output_path=args.output)
    elif args.models:
        selected = {k: v for k, v in MODELS.items() if k in args.models}
        run_benchmark(models=selected, output_path=args.output)
    else:
        run_benchmark(output_path=args.output)
