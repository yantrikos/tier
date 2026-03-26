#!/usr/bin/env python3
"""
Tier-Based Adaptive Tool Routing — Benchmark Harness v3

Uses Ollama's NATIVE tool calling API (/api/chat with tools parameter)
instead of text-injected prompts. This matches how YantrikOS and
production agents actually use tool calling.

Strategies:
1. baseline: all 80 tools via native API
2. semantic_8: YantrikDB top-8 via native API
3. semantic_4: YantrikDB top-4 via native API
4. family_oracle: correct family tools via native API (upper bound)
5. family_detected: YantrikDB-detected family via native API
"""

import json
import time
import sys
import os
import statistics
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timezone
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_OLLAMA = "http://localhost:11434"
REMOTE_OLLAMA = "http://192.168.4.35:11434"

MODELS = {
    "qwen2.5:1.5b":       {"tier": "Tiny",   "params": "1.5B",  "host": LOCAL_OLLAMA},
    "qwen3.5:9b":          {"tier": "Medium", "params": "9B",    "host": REMOTE_OLLAMA},
    "gpt-oss:20b":         {"tier": "Large",  "params": "20B",   "host": LOCAL_OLLAMA},
    "qwen3.5:35b":         {"tier": "Large",  "params": "35B",   "host": REMOTE_OLLAMA},
}

# ---------------------------------------------------------------------------
# YantrikDB Semantic Ranker
# ---------------------------------------------------------------------------

try:
    import yantrikdb
    from sentence_transformers import SentenceTransformer
    HAS_YANTRIKDB = True
except ImportError:
    HAS_YANTRIKDB = False

class SemanticRanker:
    def __init__(self, tools):
        preindex = "/tmp/tier_v3_preindex.db"
        need_index = not os.path.exists(preindex)
        self.db = yantrikdb.YantrikDB(preindex)
        self.db.set_embedder(SentenceTransformer("all-MiniLM-L6-v2"))
        self._tool_map = {t["name"]: t for t in tools}
        if need_index:
            print("  [YantrikDB] Indexing tools...")
            for t in tools:
                text = f"{t['name'].replace('_', ' ')}: {t['full']} Category: {t['family']}"
                self.db.record(text=text, memory_type="semantic", importance=0.5,
                               metadata={"tool_name": t["name"], "family": t["family"]})
            print(f"  [YantrikDB] Indexed {len(tools)} tools")
        else:
            print("  [YantrikDB] Using pre-indexed DB")

    def rank(self, query, top_k=8):
        results = self.db.recall(query=query, top_k=top_k * 2)
        ranked, seen = [], set()
        for r in results:
            meta = r.get("metadata") or {}
            name = meta.get("tool_name") if isinstance(meta, dict) else None
            if name and name in self._tool_map and name not in seen:
                ranked.append(self._tool_map[name])
                seen.add(name)
                if len(ranked) >= top_k:
                    break
        return ranked

    def detect_family(self, query):
        results = self.db.recall(query=query, top_k=5)
        votes = {}
        for r in results:
            meta = r.get("metadata") or {}
            fam = meta.get("family") if isinstance(meta, dict) else None
            if fam:
                votes[fam] = votes.get(fam, 0) + r.get("score", 1.0)
        return max(votes, key=votes.get) if votes else None

_ranker = None
def get_ranker(tools):
    global _ranker
    if _ranker is None:
        _ranker = SemanticRanker(tools)
    return _ranker

# ---------------------------------------------------------------------------
# Tool Registry — 80 tools across 8 families
# ---------------------------------------------------------------------------

TOOLS = [
    # Filesystem (10)
    {"name": "file_read",       "family": "filesystem", "short": "Read file contents", "full": "Read the contents of a file from the local filesystem. Supports text files with optional line numbering, offset, and encoding control.","params": {"path": "string"}},
    {"name": "file_write",      "family": "filesystem", "short": "Write file", "full": "Write content to a file on the local filesystem. Creates parent directories if needed.","params": {"path": "string", "content": "string"}},
    {"name": "file_delete",     "family": "filesystem", "short": "Delete file", "full": "Delete a file or empty directory from the local filesystem.","params": {"path": "string"}},
    {"name": "file_list",       "family": "filesystem", "short": "List directory contents", "full": "List files and directories in a given path. Returns names, sizes, types, and modification times.","params": {"path": "string"}},
    {"name": "file_search",     "family": "filesystem", "short": "Search files by name pattern", "full": "Search for files by name pattern or glob across the filesystem.","params": {"pattern": "string"}},
    {"name": "file_move",       "family": "filesystem", "short": "Move or rename file", "full": "Move or rename a file or directory.","params": {"source": "string", "destination": "string"}},
    {"name": "file_copy",       "family": "filesystem", "short": "Copy file", "full": "Copy a file or directory to a new location.","params": {"source": "string", "destination": "string"}},
    {"name": "file_info",       "family": "filesystem", "short": "Get file metadata", "full": "Get detailed metadata about a file: size, permissions, owner, modification time.","params": {"path": "string"}},
    {"name": "file_chmod",      "family": "filesystem", "short": "Change file permissions", "full": "Change file or directory permissions.","params": {"path": "string", "mode": "string"}},
    {"name": "file_watch",      "family": "filesystem", "short": "Watch file for changes", "full": "Watch a file or directory for changes and trigger on events.","params": {"path": "string"}},
    # Code (12)
    {"name": "grep",            "family": "code", "short": "Search file contents with regex", "full": "Search file contents using regex patterns. Supports context lines and file type filtering.","params": {"pattern": "string"}},
    {"name": "code_edit",       "family": "code", "short": "Edit source code", "full": "Replace a specific string in a source file with a new string.","params": {"file": "string", "old": "string", "new": "string"}},
    {"name": "code_format",     "family": "code", "short": "Auto-format code", "full": "Auto-format source code using language-appropriate formatters.","params": {"file": "string"}},
    {"name": "code_lint",       "family": "code", "short": "Lint source code", "full": "Run linting checks on source code. Returns warnings and errors.","params": {"file": "string"}},
    {"name": "code_symbols",    "family": "code", "short": "List code symbols and definitions", "full": "Extract function, class, and variable definitions from source code.","params": {"file": "string"}},
    {"name": "run_command",     "family": "code", "short": "Execute shell command", "full": "Execute a shell command and return stdout, stderr, and exit code.","params": {"command": "string"}},
    {"name": "run_test",        "family": "code", "short": "Run test suite", "full": "Run test suites using the project's test framework.","params": {"path": "string"}},
    {"name": "git_status",      "family": "code", "short": "Show git working tree status", "full": "Show the working tree status of a git repository.","params": {}},
    {"name": "git_commit",      "family": "code", "short": "Create git commit", "full": "Create a git commit with a message.","params": {"message": "string"}},
    {"name": "git_diff",        "family": "code", "short": "Show git changes", "full": "Show changes between commits or working tree.","params": {}},
    {"name": "git_log",         "family": "code", "short": "Show git commit history", "full": "Show commit history with messages and authors.","params": {}},
    {"name": "git_branch",      "family": "code", "short": "Manage git branches", "full": "List, create, or switch branches.","params": {}},
    # Web (10)
    {"name": "web_fetch",       "family": "web", "short": "Fetch webpage as markdown", "full": "Fetch content from a URL and return clean markdown.","params": {"url": "string"}},
    {"name": "web_search",      "family": "web", "short": "Search the web", "full": "Search the web and return top results with titles and snippets.","params": {"query": "string"}},
    {"name": "web_screenshot",  "family": "web", "short": "Screenshot webpage", "full": "Take a screenshot of a webpage as PNG.","params": {"url": "string"}},
    {"name": "api_request",     "family": "web", "short": "Make HTTP API request", "full": "Make an HTTP request to an API endpoint.","params": {"url": "string", "method": "string"}},
    {"name": "download_file",   "family": "web", "short": "Download file from URL", "full": "Download a file from a URL and save locally.","params": {"url": "string", "path": "string"}},
    {"name": "rss_read",        "family": "web", "short": "Read RSS feed", "full": "Parse an RSS or Atom feed URL and return entries.","params": {"url": "string"}},
    {"name": "dns_lookup",      "family": "web", "short": "DNS lookup", "full": "Resolve a domain name to IP addresses.","params": {"domain": "string"}},
    {"name": "ping",            "family": "web", "short": "Ping host", "full": "Send ICMP ping to a host and report latency.","params": {"host": "string"}},
    {"name": "ssl_check",       "family": "web", "short": "Check SSL certificate", "full": "Check SSL/TLS certificate of a domain.","params": {"domain": "string"}},
    {"name": "whois_lookup",    "family": "web", "short": "WHOIS domain lookup", "full": "Query WHOIS records for a domain.","params": {"domain": "string"}},
    # Data (10)
    {"name": "json_query",      "family": "data", "short": "Query JSON data", "full": "Query and transform JSON data using JMESPath expressions.","params": {"data": "string", "query": "string"}},
    {"name": "csv_read",        "family": "data", "short": "Read CSV file", "full": "Read and parse a CSV file into structured data.","params": {"path": "string"}},
    {"name": "csv_write",       "family": "data", "short": "Write CSV file", "full": "Write structured data to a CSV file.","params": {"path": "string", "data": "string"}},
    {"name": "db_query",        "family": "data", "short": "Execute SQL query", "full": "Execute a SQL query against SQLite or PostgreSQL.","params": {"query": "string"}},
    {"name": "db_schema",       "family": "data", "short": "Show database schema", "full": "List tables, columns, and indexes of a database.","params": {}},
    {"name": "calc",            "family": "data", "short": "Calculate math expression", "full": "Evaluate a mathematical expression or unit conversion.","params": {"expression": "string"}},
    {"name": "data_transform",  "family": "data", "short": "Transform data between formats", "full": "Transform data between formats: JSON, CSV, XML, YAML.","params": {"input": "string", "from_fmt": "string", "to_fmt": "string"}},
    {"name": "regex_extract",   "family": "data", "short": "Extract with regex", "full": "Extract data from text using regular expressions.","params": {"text": "string", "pattern": "string"}},
    {"name": "hash_compute",    "family": "data", "short": "Compute hash", "full": "Compute cryptographic hash of text or file.","params": {"input": "string"}},
    {"name": "base64_convert",  "family": "data", "short": "Base64 encode or decode", "full": "Encode or decode data using Base64.","params": {"input": "string", "action": "string"}},
    # Communication (10)
    {"name": "send_message",    "family": "communication", "short": "Send chat message", "full": "Send a message to a user or channel via messaging platform.","params": {"to": "string", "text": "string"}},
    {"name": "send_email",      "family": "communication", "short": "Send email", "full": "Compose and send an email with optional attachments.","params": {"to": "string", "subject": "string", "body": "string"}},
    {"name": "calendar_event",  "family": "communication", "short": "Create calendar event", "full": "Create a calendar event with title, date, and time.","params": {"title": "string", "date": "string"}},
    {"name": "calendar_list",   "family": "communication", "short": "List calendar events", "full": "List upcoming calendar events.","params": {}},
    {"name": "set_reminder",    "family": "communication", "short": "Set reminder", "full": "Set a reminder for a specific time or delay.","params": {"text": "string", "when": "string"}},
    {"name": "read_inbox",      "family": "communication", "short": "Read inbox messages", "full": "Read recent messages from email or messaging platform.","params": {}},
    {"name": "contact_lookup",  "family": "communication", "short": "Lookup contact info", "full": "Search contacts by name, email, or phone.","params": {"query": "string"}},
    {"name": "translate",       "family": "communication", "short": "Translate text", "full": "Translate text between languages.","params": {"text": "string", "to": "string"}},
    {"name": "summarize_thread","family": "communication", "short": "Summarize message thread", "full": "Summarize a message thread or email chain.","params": {"thread_id": "string"}},
    {"name": "notify",          "family": "communication", "short": "Send push notification", "full": "Push a notification to the user's device.","params": {"title": "string", "body": "string"}},
    # System (10)
    {"name": "system_info",     "family": "system", "short": "Get system info", "full": "Get CPU, memory, disk, OS version, and uptime.","params": {}},
    {"name": "process_list",    "family": "system", "short": "List running processes", "full": "List running processes with PID, name, CPU, and memory.","params": {}},
    {"name": "process_kill",    "family": "system", "short": "Kill process by PID", "full": "Terminate a running process by PID.","params": {"pid": "integer"}},
    {"name": "env_get",         "family": "system", "short": "Get environment variable", "full": "Get the value of an environment variable.","params": {"name": "string"}},
    {"name": "env_set",         "family": "system", "short": "Set environment variable", "full": "Set an environment variable for the session.","params": {"name": "string", "value": "string"}},
    {"name": "cron_schedule",   "family": "system", "short": "Create scheduled task", "full": "Create a scheduled task using cron syntax.","params": {"schedule": "string", "command": "string"}},
    {"name": "cron_list",       "family": "system", "short": "List scheduled tasks", "full": "List all scheduled cron tasks.","params": {}},
    {"name": "service_status",  "family": "system", "short": "Check service status", "full": "Check status of a system service.","params": {"name": "string"}},
    {"name": "disk_usage",      "family": "system", "short": "Show disk usage", "full": "Show disk usage for a path or all volumes.","params": {}},
    {"name": "network_info",    "family": "system", "short": "Show network info", "full": "Show network interfaces, IPs, and connections.","params": {}},
    # DevOps (10)
    {"name": "docker_ps",       "family": "devops", "short": "List Docker containers", "full": "List Docker containers with status and ports.","params": {}},
    {"name": "docker_logs",     "family": "devops", "short": "Get container logs", "full": "Fetch logs from a Docker container.","params": {"container": "string"}},
    {"name": "docker_exec",     "family": "devops", "short": "Execute in container", "full": "Execute a command inside a Docker container.","params": {"container": "string", "command": "string"}},
    {"name": "docker_build",    "family": "devops", "short": "Build Docker image", "full": "Build a Docker image from a Dockerfile.","params": {"path": "string"}},
    {"name": "k8s_get",         "family": "devops", "short": "Get Kubernetes resource", "full": "Get Kubernetes resources (pods, services, deployments).","params": {"kind": "string"}},
    {"name": "k8s_logs",        "family": "devops", "short": "Get pod logs", "full": "Fetch logs from a Kubernetes pod.","params": {"pod": "string"}},
    {"name": "ssh_exec",        "family": "devops", "short": "Run SSH command", "full": "Execute a command on a remote host via SSH.","params": {"host": "string", "command": "string"}},
    {"name": "deploy",          "family": "devops", "short": "Deploy application", "full": "Deploy an application to the target environment.","params": {"app": "string", "version": "string"}},
    {"name": "terraform_plan",  "family": "devops", "short": "Terraform plan", "full": "Run terraform plan to preview infrastructure changes.","params": {"path": "string"}},
    {"name": "ansible_run",     "family": "devops", "short": "Run Ansible playbook", "full": "Execute an Ansible playbook.","params": {"playbook": "string"}},
    # AI (8)
    {"name": "memory_store",    "family": "ai", "short": "Store in memory", "full": "Store a fact or preference in persistent cognitive memory.","params": {"text": "string"}},
    {"name": "memory_recall",   "family": "ai", "short": "Recall from memory", "full": "Search cognitive memory by semantic similarity.","params": {"query": "string"}},
    {"name": "memory_forget",   "family": "ai", "short": "Forget memory", "full": "Delete a specific memory by ID.","params": {"id": "string"}},
    {"name": "embed_text",      "family": "ai", "short": "Generate text embedding", "full": "Generate a vector embedding for text.","params": {"text": "string"}},
    {"name": "classify_text",   "family": "ai", "short": "Classify text", "full": "Classify text into predefined categories.","params": {"text": "string", "labels": "string"}},
    {"name": "sentiment",       "family": "ai", "short": "Analyze sentiment", "full": "Analyze sentiment of text (positive/negative/neutral).","params": {"text": "string"}},
    {"name": "ocr_image",       "family": "ai", "short": "OCR image to text", "full": "Extract text from an image using OCR.","params": {"path": "string"}},
    {"name": "tts_speak",       "family": "ai", "short": "Text to speech", "full": "Convert text to speech audio.","params": {"text": "string"}},
]

# ---------------------------------------------------------------------------
# Test Prompts — 50 prompts
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    {"prompt": "Read the file config.yaml", "tool": "file_read", "family": "filesystem"},
    {"prompt": "Show me what's in /etc/hosts", "tool": "file_read", "family": "filesystem"},
    {"prompt": "Write 'hello world' to output.txt", "tool": "file_write", "family": "filesystem"},
    {"prompt": "Save this text to notes.md: Meeting at 3pm", "tool": "file_write", "family": "filesystem"},
    {"prompt": "Delete the temporary file /tmp/cache.json", "tool": "file_delete", "family": "filesystem"},
    {"prompt": "List all files in the src directory", "tool": "file_list", "family": "filesystem"},
    {"prompt": "What files are in the current directory?", "tool": "file_list", "family": "filesystem"},
    {"prompt": "Find all Python files in this project", "tool": "file_search", "family": "filesystem"},
    {"prompt": "Search for files named README", "tool": "file_search", "family": "filesystem"},
    {"prompt": "Rename data.csv to data_backup.csv", "tool": "file_move", "family": "filesystem"},
    {"prompt": "Search for TODO comments in the codebase", "tool": "grep", "family": "code"},
    {"prompt": "Find all occurrences of 'import os' in Python files", "tool": "grep", "family": "code"},
    {"prompt": "Change 'old_name' to 'new_name' in app.py", "tool": "code_edit", "family": "code"},
    {"prompt": "What's the git status?", "tool": "git_status", "family": "code"},
    {"prompt": "Commit with message 'fix: resolve null pointer'", "tool": "git_commit", "family": "code"},
    {"prompt": "Run the test suite with pytest", "tool": "run_command", "family": "code"},
    {"prompt": "Execute npm install", "tool": "run_command", "family": "code"},
    {"prompt": "Check if port 8080 is in use", "tool": "run_command", "family": "code"},
    {"prompt": "Fetch the content of https://example.com", "tool": "web_fetch", "family": "web"},
    {"prompt": "Read https://docs.python.org/3/", "tool": "web_fetch", "family": "web"},
    {"prompt": "Search the web for Python FastAPI tutorial", "tool": "web_search", "family": "web"},
    {"prompt": "Google for REST API best practices", "tool": "web_search", "family": "web"},
    {"prompt": "Take a screenshot of https://github.com", "tool": "web_screenshot", "family": "web"},
    {"prompt": "GET https://api.github.com/users/octocat", "tool": "api_request", "family": "web"},
    {"prompt": "POST to https://httpbin.org/post with {'key':'value'}", "tool": "api_request", "family": "web"},
    {"prompt": "Download https://example.com/data.zip to Downloads", "tool": "download_file", "family": "web"},
    {"prompt": "Parse response.json and extract the users array", "tool": "json_query", "family": "data"},
    {"prompt": "Read the CSV file sales_data.csv", "tool": "csv_read", "family": "data"},
    {"prompt": "SELECT * FROM users WHERE active = true", "tool": "db_query", "family": "data"},
    {"prompt": "How many rows in the orders table?", "tool": "db_query", "family": "data"},
    {"prompt": "Calculate 15% of 2499.99", "tool": "calc", "family": "data"},
    {"prompt": "What is the square root of 144?", "tool": "calc", "family": "data"},
    {"prompt": "Convert this JSON to CSV format", "tool": "data_transform", "family": "data"},
    {"prompt": "Send Alice: the deployment is complete", "tool": "send_message", "family": "communication"},
    {"prompt": "Message #team: servers are back online", "tool": "send_message", "family": "communication"},
    {"prompt": "Email bob@company.com about quarterly report", "tool": "send_email", "family": "communication"},
    {"prompt": "Create meeting tomorrow 2pm: Sprint Planning", "tool": "calendar_event", "family": "communication"},
    {"prompt": "Schedule team sync next Monday 10am", "tool": "calendar_event", "family": "communication"},
    {"prompt": "Remind me to review the PR in 2 hours", "tool": "set_reminder", "family": "communication"},
    {"prompt": "Set reminder Friday: submit expense report", "tool": "set_reminder", "family": "communication"},
    {"prompt": "Check my email inbox", "tool": "read_inbox", "family": "communication"},
    {"prompt": "Show unread messages from last hour", "tool": "read_inbox", "family": "communication"},
    {"prompt": "How much disk space is available?", "tool": "disk_usage", "family": "system"},
    {"prompt": "Show CPU and memory usage", "tool": "system_info", "family": "system"},
    {"prompt": "Kill process with PID 12345", "tool": "process_kill", "family": "system"},
    {"prompt": "What is the DATABASE_URL environment variable?", "tool": "env_get", "family": "system"},
    {"prompt": "Get the value of OPENAI_API_KEY", "tool": "env_get", "family": "system"},
    {"prompt": "Schedule a backup every day at midnight", "tool": "cron_schedule", "family": "system"},
    {"prompt": "Run cleanup script every Sunday 3am", "tool": "cron_schedule", "family": "system"},
    {"prompt": "Check if nginx service is running", "tool": "service_status", "family": "system"},
]

# ---------------------------------------------------------------------------
# Native Tool Calling API
# ---------------------------------------------------------------------------

def to_native_tools(tool_list):
    """Convert tool dicts to Ollama native tool format."""
    native = []
    for t in tool_list:
        props = {}
        required = []
        for k, v in t["params"].items():
            props[k] = {"type": v if v in ("string", "integer", "boolean", "number") else "string", "description": k}
            required.append(k)
        native.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["short"],
                "parameters": {"type": "object", "properties": props, "required": required[:2]}
            }
        })
    return native


def call_native(host, model, prompt, tools, timeout=120):
    """Call Ollama /api/chat with native tools. Returns (tool_name, tokens, latency)."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256}
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        elapsed = time.monotonic() - t0
        msg = data.get("message", {})
        calls = msg.get("tool_calls", [])
        tool_name = calls[0]["function"]["name"] if calls else None
        tokens = data.get("prompt_eval_count", 0)
        return tool_name, tokens, elapsed, None
    except Exception as e:
        return None, 0, 0, str(e)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_baseline(tools, prompt, model_info, ranker):
    """All 80 tools via native API."""
    return to_native_tools(tools), "baseline"

def strategy_semantic_8(tools, prompt, model_info, ranker):
    """YantrikDB top-8 via native API."""
    ranked = ranker.rank(prompt, top_k=8)
    return to_native_tools(ranked), "semantic_8"

def strategy_semantic_4(tools, prompt, model_info, ranker):
    """YantrikDB top-4 via native API."""
    ranked = ranker.rank(prompt, top_k=4)
    return to_native_tools(ranked), "semantic_4"

def strategy_family_oracle(tools, prompt, model_info, ranker, expected_family=None):
    """Correct family tools via native API (upper bound)."""
    if expected_family:
        family_tools = [t for t in tools if t["family"] == expected_family]
    else:
        family_tools = tools
    return to_native_tools(family_tools), "family_oracle"

def strategy_family_detected(tools, prompt, model_info, ranker):
    """YantrikDB-detected family via native API."""
    detected = ranker.detect_family(prompt)
    if detected:
        family_tools = [t for t in tools if t["family"] == detected]
        if family_tools:
            return to_native_tools(family_tools), "family_detected"
    return to_native_tools(tools), "family_detected"

STRATEGIES = ["baseline", "semantic_8", "semantic_4", "family_oracle", "family_detected"]

# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    model: str
    tier: str
    params_str: str
    strategy: str
    prompt_idx: int
    user_prompt: str
    expected_tool: str
    expected_family: str
    selected_tool: Optional[str]
    tool_correct: bool
    family_correct: bool
    prompt_tokens: int
    latency_s: float
    error: Optional[str]


def run_benchmark(models=None, prompts=None, strategies=None, output_path=None):
    if models is None: models = MODELS
    if prompts is None: prompts = TEST_PROMPTS
    if strategies is None: strategies = STRATEGIES
    if output_path is None: output_path = os.path.join(os.path.dirname(__file__), "results_v3.jsonl")

    ranker = get_ranker(TOOLS) if HAS_YANTRIKDB else None
    results = []
    total = len(models) * len(prompts) * len(strategies)
    done = 0

    print(f"\n{'='*80}")
    print(f"  TIER BENCHMARK v3 — NATIVE TOOL CALLING")
    print(f"  Models: {len(models)} | Prompts: {len(prompts)} | Strategies: {len(strategies)} | Total: {total}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}\n")

    for model_name, model_info in models.items():
        tier = model_info["tier"]
        host = model_info["host"]
        print(f"\n--- {model_name} ({tier}, {model_info['params']}) @ {host} ---")

        for strat_name in strategies:
            correct = 0
            total_tokens = 0
            total_latency = 0

            for pidx, test in enumerate(prompts):
                if strat_name == "baseline":
                    native_tools, _ = strategy_baseline(TOOLS, test["prompt"], model_info, ranker)
                elif strat_name == "semantic_8":
                    native_tools, _ = strategy_semantic_8(TOOLS, test["prompt"], model_info, ranker)
                elif strat_name == "semantic_4":
                    native_tools, _ = strategy_semantic_4(TOOLS, test["prompt"], model_info, ranker)
                elif strat_name == "family_oracle":
                    native_tools, _ = strategy_family_oracle(TOOLS, test["prompt"], model_info, ranker, test["family"])
                elif strat_name == "family_detected":
                    native_tools, _ = strategy_family_detected(TOOLS, test["prompt"], model_info, ranker)
                else:
                    continue

                selected, tokens, latency, error = call_native(host, model_name, test["prompt"], native_tools)
                tool_correct = (selected == test["tool"])
                tool_obj = next((t for t in TOOLS if t["name"] == selected), None)
                family_correct = (tool_obj["family"] == test["family"]) if tool_obj else False

                if tool_correct: correct += 1
                total_tokens += tokens
                total_latency += latency

                results.append(Result(
                    model=model_name, tier=tier, params_str=model_info["params"],
                    strategy=strat_name, prompt_idx=pidx, user_prompt=test["prompt"],
                    expected_tool=test["tool"], expected_family=test["family"],
                    selected_tool=selected, tool_correct=tool_correct,
                    family_correct=family_correct,
                    prompt_tokens=tokens, latency_s=latency, error=error
                ))
                done += 1

            n = len(prompts)
            avg_tok = total_tokens // n if n else 0
            avg_lat = total_latency / n if n else 0
            print(f"  {strat_name:<18} {correct:>2}/{n} = {correct/n*100:5.1f}%  avg {avg_tok:>5}t  {avg_lat:.2f}s")

    # Write
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\nResults: {output_path} ({len(results)} records)")

    print_summary(results)
    return results


def print_summary(results):
    print(f"\n{'='*100}")
    print(f"  FINAL RESULTS — NATIVE TOOL CALLING")
    print(f"{'='*100}")

    models_seen = []
    for r in results:
        key = (r.model, r.tier, r.params_str)
        if key not in models_seen: models_seen.append(key)
    strats = STRATEGIES

    # Accuracy table
    print(f"\n  TOOL SELECTION ACCURACY")
    header = f"  {'Model':<28} {'Tier':<8}"
    for s in strats: header += f" {s:>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model, tier, params in models_seen:
        row = f"  {model} ({params})"
        row = f"{row:<28} {tier:<8}"
        for s in strats:
            items = [r for r in results if r.model == model and r.strategy == s]
            acc = sum(1 for i in items if i.tool_correct) / len(items) * 100 if items else 0
            row += f" {acc:>15.1f}%"
        print(row)

    # Token table
    print(f"\n  PROMPT TOKENS (avg)")
    header = f"  {'Model':<28} {'Tier':<8}"
    for s in strats: header += f" {s:>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model, tier, params in models_seen:
        row = f"  {model} ({params})"
        row = f"{row:<28} {tier:<8}"
        for s in strats:
            items = [r for r in results if r.model == model and r.strategy == s and r.prompt_tokens > 0]
            avg = statistics.mean([i.prompt_tokens for i in items]) if items else 0
            row += f" {avg:>14.0f}t"
        print(row)

    # Key findings
    print(f"\n{'='*100}")
    print(f"  KEY FINDINGS")
    print(f"{'='*100}")
    for model, tier, params in models_seen:
        base = [r for r in results if r.model == model and r.strategy == "baseline"]
        best_strat = None
        best_acc = 0
        for s in strats:
            if s == "baseline": continue
            items = [r for r in results if r.model == model and r.strategy == s]
            acc = sum(1 for i in items if i.tool_correct) / len(items) * 100 if items else 0
            if acc > best_acc:
                best_acc = acc
                best_strat = s
        base_acc = sum(1 for i in base if i.tool_correct) / len(base) * 100 if base else 0
        base_tok = statistics.mean([i.prompt_tokens for i in base if i.prompt_tokens > 0]) if any(i.prompt_tokens > 0 for i in base) else 0
        best_items = [r for r in results if r.model == model and r.strategy == best_strat] if best_strat else []
        best_tok = statistics.mean([i.prompt_tokens for i in best_items if i.prompt_tokens > 0]) if best_items and any(i.prompt_tokens > 0 for i in best_items) else 0
        delta = best_acc - base_acc
        tok_save = (1 - best_tok / base_tok) * 100 if base_tok > 0 else 0
        sign = "+" if delta >= 0 else ""
        print(f"\n  {model} ({params}, {tier}):")
        print(f"    Best strategy: {best_strat} ({sign}{delta:.1f}pp vs baseline)")
        print(f"    Accuracy: {base_acc:.1f}% → {best_acc:.1f}%")
        print(f"    Tokens: {base_tok:.0f} → {best_tok:.0f} ({tok_save:.0f}% reduction)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.quick:
        sel = {"qwen2.5:1.5b": MODELS["qwen2.5:1.5b"], "gpt-oss:20b": MODELS["gpt-oss:20b"]}
        run_benchmark(models=sel, prompts=TEST_PROMPTS[:10], output_path=args.output)
    elif args.models:
        sel = {k: v for k, v in MODELS.items() if k in args.models}
        run_benchmark(models=sel, output_path=args.output)
    else:
        run_benchmark(output_path=args.output)
