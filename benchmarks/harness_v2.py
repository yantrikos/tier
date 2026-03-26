#!/usr/bin/env python3
"""
Tier-Based Adaptive Tool Routing — Benchmark Harness v2

Tests 4 strategies across model sizes:
1. Baseline: all tools, full descriptions
2. Semantic: YantrikDB-ranked top-K tools
3. Family: detect tool category first, then show only that family
4. Full Tier: family routing + tier-adapted format (MCQ/JSON/Native) + slot mode

Based on ModelCapabilityProfile from yantrik-ml/capability.rs
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

try:
    import yantrikdb
    from sentence_transformers import SentenceTransformer
    YANTRIKDB_AVAILABLE = True
except ImportError:
    YANTRIKDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_OLLAMA = "http://localhost:11434"
REMOTE_OLLAMA = "http://192.168.4.35:11434"

MODELS = {
    "qwen2.5:1.5b":      {"tier": "Tiny",   "params": "1.5B",  "host": LOCAL_OLLAMA,
                           "max_tools": 3, "call_mode": "mcq", "slot_mode": "kv", "family_route": False},
    "qwen3.5:9b":         {"tier": "Medium", "params": "9B",    "host": REMOTE_OLLAMA,
                           "max_tools": 8, "call_mode": "json", "slot_mode": "json", "family_route": True},
    "gpt-oss:20b":        {"tier": "Large",  "params": "20B",   "host": LOCAL_OLLAMA,
                           "max_tools": 50, "call_mode": "native", "slot_mode": "json", "family_route": False},
    "qwen3.5:35b":        {"tier": "Large",  "params": "35B",   "host": REMOTE_OLLAMA,
                           "max_tools": 50, "call_mode": "native", "slot_mode": "json", "family_route": False},
}

# ---------------------------------------------------------------------------
# Tool Registry — 80 tools across 8 categories (families)
# ---------------------------------------------------------------------------

TOOL_FAMILIES = {
    "filesystem": "File and directory operations — read, write, delete, search, copy, move, watch files",
    "code": "Code editing, searching, formatting, testing, and git version control",
    "web": "Web fetching, searching, API requests, downloads, DNS, SSL, and network tools",
    "data": "Data querying, transformation, CSV, JSON, databases, regex, hashing, and encoding",
    "communication": "Messaging, email, calendar, reminders, contacts, translation, and notifications",
    "system": "System monitoring, processes, environment, services, networking, and scheduling",
    "devops": "Docker, Kubernetes, SSH, deployment, Terraform, and Ansible operations",
    "ai": "Memory storage/recall, embeddings, classification, sentiment, OCR, and text-to-speech",
}

TOOLS = [
    # Filesystem (10)
    {"name": "file_read",       "family": "filesystem", "short": "Read file", "full": "Read the contents of a file from the local filesystem. Supports text files with optional line numbering, offset, and encoding control.", "params_kv": "PATH: <file path>", "params_json": {"path": "string"}, "params_full": {"path": "string", "encoding": "string", "line_numbers": "boolean", "offset": "integer", "limit": "integer"}},
    {"name": "file_write",      "family": "filesystem", "short": "Write file", "full": "Write content to a file on the local filesystem. Creates parent directories if needed.", "params_kv": "PATH: <file path>\nCONTENT: <text>", "params_json": {"path": "string", "content": "string"}, "params_full": {"path": "string", "content": "string", "encoding": "string", "create_dirs": "boolean"}},
    {"name": "file_delete",     "family": "filesystem", "short": "Delete file", "full": "Delete a file or empty directory from the local filesystem.", "params_kv": "PATH: <file path>", "params_json": {"path": "string"}, "params_full": {"path": "string", "recursive": "boolean", "force": "boolean"}},
    {"name": "file_list",       "family": "filesystem", "short": "List directory", "full": "List files and directories in a given path. Returns names, sizes, types, and modification times.", "params_kv": "PATH: <directory>", "params_json": {"path": "string"}, "params_full": {"path": "string", "recursive": "boolean", "pattern": "string", "include_hidden": "boolean"}},
    {"name": "file_search",     "family": "filesystem", "short": "Search files by name", "full": "Search for files by name pattern or glob across the filesystem.", "params_kv": "PATTERN: <glob>", "params_json": {"pattern": "string"}, "params_full": {"pattern": "string", "path": "string", "max_depth": "integer", "type": "string"}},
    {"name": "file_move",       "family": "filesystem", "short": "Move/rename file", "full": "Move or rename a file or directory.", "params_kv": "FROM: <source>\nTO: <dest>", "params_json": {"source": "string", "destination": "string"}, "params_full": {"source": "string", "destination": "string", "overwrite": "boolean"}},
    {"name": "file_copy",       "family": "filesystem", "short": "Copy file", "full": "Copy a file or directory to a new location.", "params_kv": "FROM: <source>\nTO: <dest>", "params_json": {"source": "string", "destination": "string"}, "params_full": {"source": "string", "destination": "string", "recursive": "boolean"}},
    {"name": "file_info",       "family": "filesystem", "short": "File metadata", "full": "Get detailed metadata about a file: size, permissions, owner, modification time.", "params_kv": "PATH: <file>", "params_json": {"path": "string"}, "params_full": {"path": "string", "follow_symlinks": "boolean"}},
    {"name": "file_chmod",      "family": "filesystem", "short": "Set permissions", "full": "Change file or directory permissions.", "params_kv": "PATH: <file>\nMODE: <permissions>", "params_json": {"path": "string", "mode": "string"}, "params_full": {"path": "string", "mode": "string", "recursive": "boolean"}},
    {"name": "file_watch",      "family": "filesystem", "short": "Watch for changes", "full": "Watch a file or directory for changes and trigger on events.", "params_kv": "PATH: <file or dir>", "params_json": {"path": "string"}, "params_full": {"path": "string", "events": "array", "recursive": "boolean"}},

    # Code (12)
    {"name": "grep",            "family": "code", "short": "Search code content", "full": "Search file contents using regex patterns. Supports context lines and file type filtering.", "params_kv": "PATTERN: <regex>", "params_json": {"pattern": "string"}, "params_full": {"pattern": "string", "path": "string", "glob": "string", "context": "integer", "case_insensitive": "boolean"}},
    {"name": "code_edit",       "family": "code", "short": "Edit code", "full": "Replace a specific string in a source file with a new string.", "params_kv": "FILE: <path>\nOLD: <text>\nNEW: <text>", "params_json": {"file": "string", "old": "string", "new": "string"}, "params_full": {"file": "string", "old": "string", "new": "string", "replace_all": "boolean"}},
    {"name": "code_format",     "family": "code", "short": "Format code", "full": "Auto-format source code using language-appropriate formatters.", "params_kv": "FILE: <path>", "params_json": {"file": "string"}, "params_full": {"file": "string", "language": "string"}},
    {"name": "code_lint",       "family": "code", "short": "Lint code", "full": "Run linting checks on source code. Returns warnings and errors.", "params_kv": "FILE: <path>", "params_json": {"file": "string"}, "params_full": {"file": "string", "language": "string", "fix": "boolean"}},
    {"name": "code_symbols",    "family": "code", "short": "List code symbols", "full": "Extract function, class, and variable definitions from source code.", "params_kv": "FILE: <path>", "params_json": {"file": "string"}, "params_full": {"file": "string", "language": "string", "kind": "string"}},
    {"name": "run_command",     "family": "code", "short": "Run shell command", "full": "Execute a shell command and return stdout, stderr, and exit code.", "params_kv": "COMMAND: <shell command>", "params_json": {"command": "string"}, "params_full": {"command": "string", "cwd": "string", "timeout": "integer", "env": "object"}},
    {"name": "run_test",        "family": "code", "short": "Run tests", "full": "Run test suites using the project's test framework.", "params_kv": "PATH: <test path>", "params_json": {"path": "string"}, "params_full": {"path": "string", "framework": "string", "filter": "string", "verbose": "boolean"}},
    {"name": "git_status",      "family": "code", "short": "Git status", "full": "Show the working tree status of a git repository.", "params_kv": "", "params_json": {}, "params_full": {"short": "boolean", "branch": "boolean"}},
    {"name": "git_commit",      "family": "code", "short": "Git commit", "full": "Create a git commit with a message.", "params_kv": "MESSAGE: <commit msg>", "params_json": {"message": "string"}, "params_full": {"message": "string", "files": "array", "all": "boolean"}},
    {"name": "git_diff",        "family": "code", "short": "Git diff", "full": "Show changes between commits or working tree.", "params_kv": "", "params_json": {}, "params_full": {"ref": "string", "file": "string", "staged": "boolean"}},
    {"name": "git_log",         "family": "code", "short": "Git log", "full": "Show commit history with messages and authors.", "params_kv": "", "params_json": {}, "params_full": {"limit": "integer", "author": "string", "path": "string"}},
    {"name": "git_branch",      "family": "code", "short": "Git branch", "full": "List, create, or switch branches.", "params_kv": "", "params_json": {}, "params_full": {"name": "string", "create": "boolean", "delete": "boolean"}},

    # Web (10)
    {"name": "web_fetch",       "family": "web", "short": "Fetch webpage", "full": "Fetch content from a URL and return clean markdown.", "params_kv": "URL: <url>", "params_json": {"url": "string"}, "params_full": {"url": "string", "selector": "string", "timeout": "integer"}},
    {"name": "web_search",      "family": "web", "short": "Web search", "full": "Search the web and return top results with titles and snippets.", "params_kv": "QUERY: <search terms>", "params_json": {"query": "string"}, "params_full": {"query": "string", "num_results": "integer", "site": "string"}},
    {"name": "web_screenshot",  "family": "web", "short": "Screenshot page", "full": "Take a screenshot of a webpage as PNG.", "params_kv": "URL: <url>", "params_json": {"url": "string"}, "params_full": {"url": "string", "width": "integer", "full_page": "boolean"}},
    {"name": "api_request",     "family": "web", "short": "HTTP request", "full": "Make an HTTP request to an API endpoint.", "params_kv": "URL: <url>\nMETHOD: <GET/POST/PUT/DELETE>", "params_json": {"url": "string", "method": "string"}, "params_full": {"url": "string", "method": "string", "headers": "object", "body": "string"}},
    {"name": "download_file",   "family": "web", "short": "Download file", "full": "Download a file from a URL and save locally.", "params_kv": "URL: <url>\nPATH: <save path>", "params_json": {"url": "string", "path": "string"}, "params_full": {"url": "string", "path": "string", "overwrite": "boolean"}},
    {"name": "rss_read",        "family": "web", "short": "Read RSS feed", "full": "Parse an RSS or Atom feed URL and return entries.", "params_kv": "URL: <feed url>", "params_json": {"url": "string"}, "params_full": {"url": "string", "limit": "integer"}},
    {"name": "dns_lookup",      "family": "web", "short": "DNS lookup", "full": "Resolve a domain name to IP addresses.", "params_kv": "DOMAIN: <domain>", "params_json": {"domain": "string"}, "params_full": {"domain": "string", "type": "string"}},
    {"name": "ping",            "family": "web", "short": "Ping host", "full": "Send ICMP ping to a host and report latency.", "params_kv": "HOST: <hostname>", "params_json": {"host": "string"}, "params_full": {"host": "string", "count": "integer"}},
    {"name": "ssl_check",       "family": "web", "short": "Check SSL cert", "full": "Check SSL/TLS certificate of a domain.", "params_kv": "DOMAIN: <domain>", "params_json": {"domain": "string"}, "params_full": {"domain": "string", "port": "integer"}},
    {"name": "whois_lookup",    "family": "web", "short": "WHOIS lookup", "full": "Query WHOIS records for a domain.", "params_kv": "DOMAIN: <domain>", "params_json": {"domain": "string"}, "params_full": {"domain": "string"}},

    # Data (10)
    {"name": "json_query",      "family": "data", "short": "Query JSON", "full": "Query and transform JSON data using JMESPath expressions.", "params_kv": "DATA: <json>\nQUERY: <jmespath>", "params_json": {"data": "string", "query": "string"}, "params_full": {"data": "string", "query": "string", "pretty": "boolean"}},
    {"name": "csv_read",        "family": "data", "short": "Read CSV", "full": "Read and parse a CSV file into structured data.", "params_kv": "PATH: <file>", "params_json": {"path": "string"}, "params_full": {"path": "string", "delimiter": "string", "limit": "integer"}},
    {"name": "csv_write",       "family": "data", "short": "Write CSV", "full": "Write structured data to a CSV file.", "params_kv": "PATH: <file>\nDATA: <rows>", "params_json": {"path": "string", "data": "array"}, "params_full": {"path": "string", "data": "array", "headers": "array"}},
    {"name": "db_query",        "family": "data", "short": "Query database", "full": "Execute a SQL query against SQLite or PostgreSQL.", "params_kv": "QUERY: <sql>", "params_json": {"query": "string"}, "params_full": {"query": "string", "database": "string", "limit": "integer"}},
    {"name": "db_schema",       "family": "data", "short": "Database schema", "full": "List tables, columns, and indexes of a database.", "params_kv": "", "params_json": {}, "params_full": {"database": "string", "table": "string"}},
    {"name": "calc",            "family": "data", "short": "Calculate", "full": "Evaluate a mathematical expression or unit conversion.", "params_kv": "EXPRESSION: <math>", "params_json": {"expression": "string"}, "params_full": {"expression": "string", "precision": "integer"}},
    {"name": "data_transform",  "family": "data", "short": "Transform data", "full": "Transform data between formats: JSON, CSV, XML, YAML.", "params_kv": "INPUT: <data>\nFROM: <format>\nTO: <format>", "params_json": {"input": "string", "from": "string", "to": "string"}, "params_full": {"input": "string", "from": "string", "to": "string", "mapping": "object"}},
    {"name": "regex_extract",   "family": "data", "short": "Regex extract", "full": "Extract data from text using regular expressions.", "params_kv": "TEXT: <text>\nPATTERN: <regex>", "params_json": {"text": "string", "pattern": "string"}, "params_full": {"text": "string", "pattern": "string", "limit": "integer"}},
    {"name": "hash_compute",    "family": "data", "short": "Compute hash", "full": "Compute cryptographic hash of text or file.", "params_kv": "INPUT: <text>", "params_json": {"input": "string"}, "params_full": {"input": "string", "algorithm": "string"}},
    {"name": "base64_convert",  "family": "data", "short": "Base64 encode/decode", "full": "Encode or decode data using Base64.", "params_kv": "INPUT: <text>\nACTION: <encode/decode>", "params_json": {"input": "string", "action": "string"}, "params_full": {"input": "string", "action": "string"}},

    # Communication (10)
    {"name": "send_message",    "family": "communication", "short": "Send message", "full": "Send a message to a user or channel via messaging platform.", "params_kv": "TO: <recipient>\nTEXT: <message>", "params_json": {"to": "string", "text": "string"}, "params_full": {"to": "string", "text": "string", "channel": "string"}},
    {"name": "send_email",      "family": "communication", "short": "Send email", "full": "Compose and send an email with optional attachments.", "params_kv": "TO: <email>\nSUBJECT: <subject>\nBODY: <text>", "params_json": {"to": "string", "subject": "string", "body": "string"}, "params_full": {"to": "string", "subject": "string", "body": "string", "cc": "array", "attachments": "array"}},
    {"name": "calendar_event",  "family": "communication", "short": "Create event", "full": "Create a calendar event with title, date, and time.", "params_kv": "TITLE: <event>\nDATE: <date>", "params_json": {"title": "string", "date": "string"}, "params_full": {"title": "string", "date": "string", "time": "string", "duration": "integer", "attendees": "array"}},
    {"name": "calendar_list",   "family": "communication", "short": "List events", "full": "List upcoming calendar events.", "params_kv": "", "params_json": {}, "params_full": {"from": "string", "to": "string", "limit": "integer"}},
    {"name": "set_reminder",    "family": "communication", "short": "Set reminder", "full": "Set a reminder for a specific time or delay.", "params_kv": "TEXT: <reminder>\nWHEN: <time>", "params_json": {"text": "string", "when": "string"}, "params_full": {"text": "string", "when": "string", "repeat": "string"}},
    {"name": "read_inbox",      "family": "communication", "short": "Read inbox", "full": "Read recent messages from email or messaging platform.", "params_kv": "", "params_json": {}, "params_full": {"limit": "integer", "unread_only": "boolean"}},
    {"name": "contact_lookup",  "family": "communication", "short": "Lookup contact", "full": "Search contacts by name, email, or phone.", "params_kv": "QUERY: <name or email>", "params_json": {"query": "string"}, "params_full": {"query": "string", "field": "string"}},
    {"name": "translate",       "family": "communication", "short": "Translate text", "full": "Translate text between languages.", "params_kv": "TEXT: <text>\nTO: <language>", "params_json": {"text": "string", "to": "string"}, "params_full": {"text": "string", "to": "string", "from": "string"}},
    {"name": "summarize_thread","family": "communication", "short": "Summarize thread", "full": "Summarize a message thread or email chain.", "params_kv": "THREAD: <thread id>", "params_json": {"thread_id": "string"}, "params_full": {"thread_id": "string", "channel": "string"}},
    {"name": "notify",          "family": "communication", "short": "Send notification", "full": "Push a notification to the user's device.", "params_kv": "TITLE: <title>\nBODY: <text>", "params_json": {"title": "string", "body": "string"}, "params_full": {"title": "string", "body": "string", "priority": "string"}},

    # System (10)
    {"name": "system_info",     "family": "system", "short": "System info", "full": "Get CPU, memory, disk, OS version, and uptime.", "params_kv": "", "params_json": {}, "params_full": {"include_processes": "boolean"}},
    {"name": "process_list",    "family": "system", "short": "List processes", "full": "List running processes with PID, name, CPU, and memory.", "params_kv": "", "params_json": {}, "params_full": {"sort_by": "string", "limit": "integer"}},
    {"name": "process_kill",    "family": "system", "short": "Kill process", "full": "Terminate a running process by PID.", "params_kv": "PID: <process id>", "params_json": {"pid": "integer"}, "params_full": {"pid": "integer", "signal": "string", "force": "boolean"}},
    {"name": "env_get",         "family": "system", "short": "Get env var", "full": "Get the value of an environment variable.", "params_kv": "NAME: <var name>", "params_json": {"name": "string"}, "params_full": {"name": "string", "default": "string"}},
    {"name": "env_set",         "family": "system", "short": "Set env var", "full": "Set an environment variable for the session.", "params_kv": "NAME: <var>\nVALUE: <value>", "params_json": {"name": "string", "value": "string"}, "params_full": {"name": "string", "value": "string"}},
    {"name": "cron_schedule",   "family": "system", "short": "Schedule task", "full": "Create a scheduled task using cron syntax.", "params_kv": "SCHEDULE: <cron>\nCOMMAND: <cmd>", "params_json": {"schedule": "string", "command": "string"}, "params_full": {"schedule": "string", "command": "string", "name": "string"}},
    {"name": "cron_list",       "family": "system", "short": "List scheduled tasks", "full": "List all scheduled cron tasks.", "params_kv": "", "params_json": {}, "params_full": {"status": "string"}},
    {"name": "service_status",  "family": "system", "short": "Service status", "full": "Check status of a system service.", "params_kv": "NAME: <service>", "params_json": {"name": "string"}, "params_full": {"name": "string", "logs": "boolean"}},
    {"name": "disk_usage",      "family": "system", "short": "Disk usage", "full": "Show disk usage for a path or all volumes.", "params_kv": "", "params_json": {}, "params_full": {"path": "string"}},
    {"name": "network_info",    "family": "system", "short": "Network info", "full": "Show network interfaces, IPs, and connections.", "params_kv": "", "params_json": {}, "params_full": {"interface": "string", "connections": "boolean"}},

    # DevOps (10)
    {"name": "docker_ps",       "family": "devops", "short": "List containers", "full": "List Docker containers with status and ports.", "params_kv": "", "params_json": {}, "params_full": {"all": "boolean", "filter": "string"}},
    {"name": "docker_logs",     "family": "devops", "short": "Container logs", "full": "Fetch logs from a Docker container.", "params_kv": "CONTAINER: <name>", "params_json": {"container": "string"}, "params_full": {"container": "string", "tail": "integer"}},
    {"name": "docker_exec",     "family": "devops", "short": "Exec in container", "full": "Execute a command inside a Docker container.", "params_kv": "CONTAINER: <name>\nCOMMAND: <cmd>", "params_json": {"container": "string", "command": "string"}, "params_full": {"container": "string", "command": "string", "user": "string"}},
    {"name": "docker_build",    "family": "devops", "short": "Build image", "full": "Build a Docker image from a Dockerfile.", "params_kv": "PATH: <dir>", "params_json": {"path": "string"}, "params_full": {"path": "string", "tag": "string", "no_cache": "boolean"}},
    {"name": "k8s_get",         "family": "devops", "short": "Get K8s resource", "full": "Get Kubernetes resources (pods, services, deployments).", "params_kv": "KIND: <resource type>", "params_json": {"kind": "string"}, "params_full": {"kind": "string", "name": "string", "namespace": "string"}},
    {"name": "k8s_logs",        "family": "devops", "short": "Pod logs", "full": "Fetch logs from a Kubernetes pod.", "params_kv": "POD: <pod name>", "params_json": {"pod": "string"}, "params_full": {"pod": "string", "namespace": "string", "container": "string"}},
    {"name": "ssh_exec",        "family": "devops", "short": "SSH command", "full": "Execute a command on a remote host via SSH.", "params_kv": "HOST: <hostname>\nCOMMAND: <cmd>", "params_json": {"host": "string", "command": "string"}, "params_full": {"host": "string", "command": "string", "user": "string", "key": "string"}},
    {"name": "deploy",          "family": "devops", "short": "Deploy app", "full": "Deploy an application to the target environment.", "params_kv": "APP: <name>\nVERSION: <ver>", "params_json": {"app": "string", "version": "string"}, "params_full": {"app": "string", "version": "string", "environment": "string"}},
    {"name": "terraform_plan",  "family": "devops", "short": "Terraform plan", "full": "Run terraform plan to preview infrastructure changes.", "params_kv": "PATH: <dir>", "params_json": {"path": "string"}, "params_full": {"path": "string", "var_file": "string"}},
    {"name": "ansible_run",     "family": "devops", "short": "Run Ansible", "full": "Execute an Ansible playbook.", "params_kv": "PLAYBOOK: <file>", "params_json": {"playbook": "string"}, "params_full": {"playbook": "string", "inventory": "string"}},

    # AI (8)
    {"name": "memory_store",    "family": "ai", "short": "Store memory", "full": "Store a fact or preference in persistent cognitive memory.", "params_kv": "TEXT: <what to remember>", "params_json": {"text": "string"}, "params_full": {"text": "string", "importance": "number", "domain": "string"}},
    {"name": "memory_recall",   "family": "ai", "short": "Recall memory", "full": "Search cognitive memory by semantic similarity.", "params_kv": "QUERY: <search>", "params_json": {"query": "string"}, "params_full": {"query": "string", "limit": "integer", "domain": "string"}},
    {"name": "memory_forget",   "family": "ai", "short": "Forget memory", "full": "Delete a specific memory by ID.", "params_kv": "ID: <memory id>", "params_json": {"id": "string"}, "params_full": {"id": "string"}},
    {"name": "embed_text",      "family": "ai", "short": "Generate embedding", "full": "Generate a vector embedding for text.", "params_kv": "TEXT: <text>", "params_json": {"text": "string"}, "params_full": {"text": "string", "model": "string"}},
    {"name": "classify_text",   "family": "ai", "short": "Classify text", "full": "Classify text into predefined categories.", "params_kv": "TEXT: <text>\nLABELS: <categories>", "params_json": {"text": "string", "labels": "array"}, "params_full": {"text": "string", "labels": "array"}},
    {"name": "sentiment",       "family": "ai", "short": "Sentiment analysis", "full": "Analyze sentiment of text (positive/negative/neutral).", "params_kv": "TEXT: <text>", "params_json": {"text": "string"}, "params_full": {"text": "string"}},
    {"name": "ocr_image",       "family": "ai", "short": "OCR image", "full": "Extract text from an image using OCR.", "params_kv": "PATH: <image file>", "params_json": {"path": "string"}, "params_full": {"path": "string", "language": "string"}},
    {"name": "tts_speak",       "family": "ai", "short": "Text to speech", "full": "Convert text to speech audio.", "params_kv": "TEXT: <text>", "params_json": {"text": "string"}, "params_full": {"text": "string", "voice": "string"}},
]

# ---------------------------------------------------------------------------
# Test Prompts — 50 prompts with expected tool + family
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    # Filesystem (10)
    {"prompt": "Read the file config.yaml", "tool": "file_read", "family": "filesystem", "params": ["path"]},
    {"prompt": "Show me what's in /etc/hosts", "tool": "file_read", "family": "filesystem", "params": ["path"]},
    {"prompt": "Write 'hello world' to output.txt", "tool": "file_write", "family": "filesystem", "params": ["path", "content"]},
    {"prompt": "Save this text to a new file called notes.md: Meeting at 3pm", "tool": "file_write", "family": "filesystem", "params": ["path", "content"]},
    {"prompt": "Delete the temporary file /tmp/cache.json", "tool": "file_delete", "family": "filesystem", "params": ["path"]},
    {"prompt": "List all files in the src directory", "tool": "file_list", "family": "filesystem", "params": ["path"]},
    {"prompt": "What files are in the current directory?", "tool": "file_list", "family": "filesystem", "params": []},
    {"prompt": "Find all Python files in this project", "tool": "file_search", "family": "filesystem", "params": ["pattern"]},
    {"prompt": "Search for files named README", "tool": "file_search", "family": "filesystem", "params": ["pattern"]},
    {"prompt": "Rename data.csv to data_backup.csv", "tool": "file_move", "family": "filesystem", "params": ["source", "destination"]},
    # Code (8)
    {"prompt": "Search for TODO comments in the codebase", "tool": "grep", "family": "code", "params": ["pattern"]},
    {"prompt": "Find all occurrences of 'import os' in Python files", "tool": "grep", "family": "code", "params": ["pattern"]},
    {"prompt": "Change the variable name 'old_name' to 'new_name' in app.py", "tool": "code_edit", "family": "code", "params": ["file", "old", "new"]},
    {"prompt": "What's the git status of this repo?", "tool": "git_status", "family": "code", "params": []},
    {"prompt": "Commit these changes with message 'fix: resolve null pointer'", "tool": "git_commit", "family": "code", "params": ["message"]},
    {"prompt": "Run the test suite with pytest", "tool": "run_command", "family": "code", "params": ["command"]},
    {"prompt": "Execute npm install", "tool": "run_command", "family": "code", "params": ["command"]},
    {"prompt": "Check if the server is running on port 8080", "tool": "run_command", "family": "code", "params": ["command"]},
    # Web (8)
    {"prompt": "Fetch the content of https://example.com", "tool": "web_fetch", "family": "web", "params": ["url"]},
    {"prompt": "Read the documentation page at https://docs.python.org/3/", "tool": "web_fetch", "family": "web", "params": ["url"]},
    {"prompt": "Search the web for Python FastAPI tutorial", "tool": "web_search", "family": "web", "params": ["query"]},
    {"prompt": "Google for best practices in REST API design", "tool": "web_search", "family": "web", "params": ["query"]},
    {"prompt": "Take a screenshot of https://github.com", "tool": "web_screenshot", "family": "web", "params": ["url"]},
    {"prompt": "Make a GET request to https://api.github.com/users/octocat", "tool": "api_request", "family": "web", "params": ["url"]},
    {"prompt": "POST to https://httpbin.org/post with body {'key': 'value'}", "tool": "api_request", "family": "web", "params": ["url", "method"]},
    {"prompt": "Download the file at https://example.com/data.zip to Downloads", "tool": "download_file", "family": "web", "params": ["url", "path"]},
    # Data (7)
    {"prompt": "Parse the JSON in response.json and extract the users array", "tool": "json_query", "family": "data", "params": ["data", "query"]},
    {"prompt": "Read the CSV file sales_data.csv", "tool": "csv_read", "family": "data", "params": ["path"]},
    {"prompt": "Query the database: SELECT * FROM users WHERE active = true", "tool": "db_query", "family": "data", "params": ["query"]},
    {"prompt": "How many rows are in the orders table?", "tool": "db_query", "family": "data", "params": ["query"]},
    {"prompt": "Calculate 15% of 2499.99", "tool": "calc", "family": "data", "params": ["expression"]},
    {"prompt": "What is the square root of 144?", "tool": "calc", "family": "data", "params": ["expression"]},
    {"prompt": "Convert this JSON to CSV format", "tool": "data_transform", "family": "data", "params": ["from", "to"]},
    # Communication (9)
    {"prompt": "Send a message to Alice saying the deployment is complete", "tool": "send_message", "family": "communication", "params": ["to", "text"]},
    {"prompt": "Message the team channel: servers are back online", "tool": "send_message", "family": "communication", "params": ["to", "text"]},
    {"prompt": "Send an email to bob@company.com about the quarterly report", "tool": "send_email", "family": "communication", "params": ["to", "subject"]},
    {"prompt": "Create a meeting for tomorrow at 2pm called Sprint Planning", "tool": "calendar_event", "family": "communication", "params": ["title", "date"]},
    {"prompt": "Schedule a team sync for next Monday at 10am", "tool": "calendar_event", "family": "communication", "params": ["title", "date"]},
    {"prompt": "Remind me to review the PR in 2 hours", "tool": "set_reminder", "family": "communication", "params": ["text", "when"]},
    {"prompt": "Set a reminder for Friday: submit expense report", "tool": "set_reminder", "family": "communication", "params": ["text", "when"]},
    {"prompt": "Check my email inbox", "tool": "read_inbox", "family": "communication", "params": []},
    {"prompt": "Show me unread messages from the last hour", "tool": "read_inbox", "family": "communication", "params": []},
    # System (8)
    {"prompt": "How much disk space is available?", "tool": "disk_usage", "family": "system", "params": []},
    {"prompt": "Show me CPU and memory usage", "tool": "system_info", "family": "system", "params": []},
    {"prompt": "Kill the process with PID 12345", "tool": "process_kill", "family": "system", "params": ["pid"]},
    {"prompt": "What is the DATABASE_URL environment variable?", "tool": "env_get", "family": "system", "params": ["name"]},
    {"prompt": "Get the value of OPENAI_API_KEY", "tool": "env_get", "family": "system", "params": ["name"]},
    {"prompt": "Schedule a backup every day at midnight", "tool": "cron_schedule", "family": "system", "params": ["schedule", "command"]},
    {"prompt": "Run the cleanup script every Sunday at 3am", "tool": "cron_schedule", "family": "system", "params": ["schedule", "command"]},
    {"prompt": "Check if nginx is running", "tool": "service_status", "family": "system", "params": ["name"]},
]

# ---------------------------------------------------------------------------
# YantrikDB Semantic Ranker
# ---------------------------------------------------------------------------

class SemanticRanker:
    def __init__(self, tools):
        import os
        # Use pre-indexed DB if available
        preindex = "/tmp/tier_v2_preindex.db"
        db_path = preindex if os.path.exists(preindex) else "/tmp/tier_v2_tools.db"
        self.db = yantrikdb.YantrikDB(db_path)
        self.db.set_embedder(SentenceTransformer("all-MiniLM-L6-v2"))
        self._tool_map = {t["name"]: t for t in tools}
        if not os.path.exists(preindex):
            for t in tools:
                text = f"{t['name'].replace('_', ' ')}: {t['full']} Category: {t['family']}"
                try:
                    self.db.record(text=text, memory_type="semantic", importance=0.5,
                                   metadata={"tool_name": t["name"], "family": t["family"]})
                except:
                    pass

    def rank(self, query, top_k=4):
        results = self.db.recall(query=query, top_k=top_k * 2)
        ranked, seen = [], set()
        for r in results:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            name = meta.get("tool_name") if isinstance(meta, dict) else None
            if name and name in self._tool_map and name not in seen:
                ranked.append(self._tool_map[name])
                seen.add(name)
                if len(ranked) >= top_k:
                    break
        return ranked

    def detect_family(self, query):
        """Use embeddings to detect which tool family the query belongs to."""
        results = self.db.recall(query=query, top_k=3)
        family_votes = {}
        for r in results:
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try: meta = json.loads(meta)
                except: meta = {}
            fam = meta.get("family") if isinstance(meta, dict) else None
            if fam:
                family_votes[fam] = family_votes.get(fam, 0) + r.get("score", 1.0)
        if family_votes:
            return max(family_votes, key=family_votes.get)
        return None

_ranker = None
def get_ranker():
    global _ranker
    if _ranker is None:
        print("  [YantrikDB] Indexing 80 tools...")
        _ranker = SemanticRanker(TOOLS)
        print("  [YantrikDB] Ready")
    return _ranker

# ---------------------------------------------------------------------------
# Prompt Builders — 4 Strategies
# ---------------------------------------------------------------------------

def strategy_baseline(tools, user_msg, model_info):
    """Strategy 1: ALL tools, full descriptions. Current OpenClaw behavior."""
    lines = []
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_full"].items())
        lines.append(f"- {t['name']}: {t['full']}\n  Parameters: {{{params}}}")
    return f"""You are an AI assistant with access to tools. Pick the single best tool for the user's request and respond with JSON only.

Available tools:
{chr(10).join(lines)}

Respond with ONLY this JSON format, no other text:
{{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}""", "direct", None


def strategy_semantic(tools, user_msg, model_info):
    """Strategy 2: YantrikDB-ranked top-K, short descriptions. Same K for all tiers for fair comparison."""
    top_k = 8  # Fixed K for fair comparison across tiers
    ranked = get_ranker().rank(user_msg, top_k=top_k)
    lines = []
    for t in ranked:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
        lines.append(f"- {t['name']}: {t['short']}. Params: {{{params}}}")
    return f"""You are an AI assistant. Pick the best tool and respond with JSON only.

Tools:
{chr(10).join(lines)}

Respond: {{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}""", "direct", None


def strategy_family(tools, user_msg, model_info):
    """Strategy 3: Detect family first, then show only that family's tools."""
    detected = get_ranker().detect_family(user_msg)
    family_tools = [t for t in tools if t["family"] == detected] if detected else tools
    # Fallback: if family detection failed or too few tools, use semantic
    if len(family_tools) < 2:
        return strategy_semantic(tools, user_msg, model_info)

    lines = []
    for t in family_tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
        lines.append(f"- {t['name']}: {t['short']} ({params})")

    family_label = (detected or "general").upper()
    return f"""You are an AI assistant. Pick the best tool and respond with JSON only.

{family_label} tools:
{chr(10).join(lines)}

Respond: {{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}""", "direct", None


def strategy_full_tier(tools, user_msg, model_info):
    """Strategy 4: Full tier-adapted (family + format + slot mode)."""
    call_mode = model_info.get("call_mode", "json")
    slot_mode = model_info.get("slot_mode", "json")
    use_family = model_info.get("family_route", False)
    max_tools = model_info.get("max_tools", 8)

    # Step 1: Get relevant tools (family route or semantic rank)
    if use_family:
        detected = get_ranker().detect_family(user_msg)
        candidates = [t for t in tools if t["family"] == detected] if detected else tools
        if len(candidates) < 2:
            candidates = [t for t in get_ranker().rank(user_msg, top_k=max_tools)]
    else:
        candidates = get_ranker().rank(user_msg, top_k=max_tools) if max_tools < len(tools) else tools

    candidates = candidates[:max_tools]

    # Step 2: Format based on call_mode
    if call_mode == "mcq":
        labels = "ABCDE"
        options = []
        mcq_map = []
        for i, t in enumerate(candidates):
            kv = t.get("params_kv", "")
            options.append(f"{labels[i]}) {t['name']}: {t['short']}")
            mcq_map.append(t["name"])
        options_text = "\n".join(options)

        if slot_mode == "kv":
            return f"""Pick the best tool. Reply with the letter and key-value parameters.

{options_text}

Reply format:
CHOICE: <letter>
{candidates[0].get('params_kv', 'PARAM: <value>') if candidates else ''}

User: {user_msg}""", "mcq_kv", mcq_map
        else:
            return f"""Pick the best tool. Reply with JSON.

{options_text}

Reply: {{"choice": "A", "params": {{"key": "value"}}}}

User: {user_msg}""", "mcq_json", mcq_map

    elif call_mode == "json":
        lines = []
        for t in candidates:
            params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
            lines.append(f"- {t['name']}: {t['short']} ({params})")
        return f"""Pick the best tool. Respond with JSON only.

Tools:
{chr(10).join(lines)}

Respond: {{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}""", "direct", None

    else:  # native — full tool set with short descriptions
        lines = []
        for t in candidates:
            params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
            lines.append(f"- {t['name']}: {t['short']}. Params: {{{params}}}")
        return f"""You are an AI assistant with access to tools. Pick the single best tool and respond with JSON only.

Available tools:
{chr(10).join(lines)}

Respond with ONLY: {{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}""", "direct", None


def strategy_discovery(tools, user_msg, model_info):
    """Strategy 5: Two-step discovery (YantrikOS pattern).

    Step 1: Model sees category summary + user prompt → picks a category
    Step 2: Model sees only that category's tools → picks the tool

    This simulates the discover_tools meta-tool flow.
    We simulate both steps in one LLM call using a structured prompt.
    """
    # Build category summary (what discover_tools() returns with no args)
    from collections import Counter
    cat_counts = Counter(t["family"] for t in tools)
    cat_lines = []
    for cat, count in sorted(cat_counts.items()):
        desc = TOOL_FAMILIES.get(cat, "")
        cat_lines.append(f"- {cat} ({count} tools): {desc}")
    cat_summary = "\n".join(cat_lines)

    # Use semantic detection to figure out what family the model SHOULD pick
    # so we can build the second-step prompt
    detected = get_ranker().detect_family(user_msg)

    if detected:
        family_tools = [t for t in tools if t["family"] == detected]
    else:
        # Fallback: use semantic ranking to find top tools
        family_tools = get_ranker().rank(user_msg, top_k=10)

    # Build the two-step prompt as a single call
    # Step 1 context: category summary
    # Step 2 context: the detected family's tools
    tool_lines = []
    for t in family_tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
        tool_lines.append(f"- {t['name']}: {t['short']} ({params})")
    tools_text = "\n".join(tool_lines)

    # List exact tool names for emphasis
    tool_names = [t["name"] for t in family_tools]

    prompt = f"""Pick a tool from this list. You MUST use one of these exact names: {', '.join(tool_names)}

{tools_text}

Respond with JSON only: {{"tool": "exact_tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}"""

    return prompt, "direct", None


def strategy_discovery_2step(tools, user_msg, model_info):
    """Strategy 6: True two-step discovery with TWO separate LLM calls.

    Call 1: "Which category?" → model picks a category
    Call 2: "Which tool from this category?" → model picks the tool

    This is what actually happens in YantrikOS with discover_tools.
    """
    # This strategy is special — it needs two LLM calls
    # We return a marker that the benchmark runner handles
    return None, "two_step", None


STRATEGIES = {
    "baseline": strategy_baseline,
    "family": strategy_family,
    "discovery_2step": strategy_discovery_2step,
}

# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def ollama_generate(host, model, prompt, timeout=120):
    url = f"{host}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False,
                          "options": {"temperature": 0.0, "num_predict": 256}}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        return {"response": body.get("response", ""), "prompt_eval_count": body.get("prompt_eval_count", 0),
                "eval_count": body.get("eval_count", 0), "elapsed_s": time.monotonic() - t0, "error": None}
    except Exception as e:
        return {"response": "", "prompt_eval_count": 0, "eval_count": 0, "elapsed_s": 0, "error": str(e)}

# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def parse_response(text, mode="direct", mcq_map=None):
    text = text.strip()

    if mode == "mcq_kv":
        # Parse CHOICE: A format
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("CHOICE:"):
                choice = line.split(":", 1)[1].strip().upper()
                if choice and choice[0] in "ABCDE" and mcq_map:
                    idx = ord(choice[0]) - ord("A")
                    if idx < len(mcq_map):
                        return {"tool": mcq_map[idx], "params": {}, "parse_error": None}
        # Fallback: try JSON
        return parse_json_response(text, mode, mcq_map)

    if mode == "mcq_json":
        parsed = parse_json_response(text, mode, mcq_map)
        if parsed["tool"] and mcq_map:
            choice = parsed["tool"]
            if choice.upper() in "ABCDE":
                idx = ord(choice.upper()[0]) - ord("A")
                if idx < len(mcq_map):
                    parsed["tool"] = mcq_map[idx]
        return parsed

    return parse_json_response(text, mode, mcq_map)


def parse_json_response(text, mode="direct", mcq_map=None):
    start = text.find("{")
    if start == -1:
        return {"tool": None, "params": {}, "parse_error": "no_json"}
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: end = i + 1; break
    try:
        parsed = json.loads(text[start:end])
    except:
        return {"tool": None, "params": {}, "parse_error": "invalid_json"}

    if "choice" in parsed and mcq_map:
        choice = str(parsed["choice"]).upper()
        if choice and choice[0] in "ABCDE":
            idx = ord(choice[0]) - ord("A")
            if idx < len(mcq_map):
                return {"tool": mcq_map[idx], "params": parsed.get("params", {}), "parse_error": None}

    tool = parsed.get("tool", parsed.get("name"))
    params = parsed.get("params", parsed.get("parameters", parsed.get("arguments", {})))
    return {"tool": tool, "params": params if isinstance(params, dict) else {}, "parse_error": None}

# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_two_step_discovery(host, model, model_info, tools, user_msg):
    """Execute a true two-step discovery flow with separate LLM calls."""
    tier = model_info.get("tier", "Medium")
    # Build category summary
    from collections import Counter
    cat_counts = Counter(t["family"] for t in tools)
    cat_lines = [f"- {cat} ({count} tools)" for cat, count in sorted(cat_counts.items())]

    # Step 1: Ask model to pick a category
    # Use MCQ format for tiny/small models, free text for larger
    tier = model_info.get("tier", "Medium")
    if tier in ("Tiny", "Small"):
        labels = "ABCDEFGH"
        cat_options = []
        cat_names = sorted(cat_counts.keys())
        for i, cat in enumerate(cat_names):
            cat_options.append(f"{labels[i]}) {cat}: {TOOL_FAMILIES.get(cat, '')[:60]}")
        step1_prompt = f"""Which category? Pick one letter.

{chr(10).join(cat_options)}

User: {user_msg}

Reply with ONLY the letter (A, B, C, etc.):"""
    else:
        step1_prompt = f"""You need to pick a tool for the user. First, choose the most relevant category.

Available categories:
{chr(10).join(cat_lines)}

Reply with ONLY the category name (one word, lowercase).

User: {user_msg}"""

    resp1 = ollama_generate(host, model, step1_prompt)
    if resp1["error"]:
        return resp1

    # Parse category from response
    raw_cat = resp1["response"].strip().split("\n")[0].strip().strip('"').strip("'").strip(".")

    # Handle MCQ letter response for tiny/small models
    cat_names = sorted(cat_counts.keys())
    labels = "ABCDEFGH"
    category = None

    # Check if response is a letter
    first_char = raw_cat.upper()[0] if raw_cat else ""
    if first_char in labels and tier in ("Tiny", "Small"):
        idx = labels.index(first_char)
        if idx < len(cat_names):
            category = cat_names[idx]

    # Fallback: check if response contains a category name
    if not category:
        for cat_name in TOOL_FAMILIES.keys():
            if cat_name in raw_cat.lower():
                category = cat_name
                break

    # Final fallback: use semantic detection
    if not category:
        category = get_ranker().detect_family(user_msg)

    # Step 2: Show tools from that category, ask to pick
    family_tools = [t for t in tools if t["family"] == category]
    if not family_tools:
        # Fallback: use semantic ranking
        family_tools = get_ranker().rank(user_msg, top_k=10)

    tool_lines = []
    for t in family_tools:
        params = ", ".join(f"{k}: {v}" for k, v in t["params_json"].items())
        tool_lines.append(f"- {t['name']}: {t['short']} ({params})")

    step2_prompt = f"""Pick the best tool from these {category} tools. Respond with JSON only.

{chr(10).join(tool_lines)}

Respond: {{"tool": "tool_name", "params": {{"param1": "value1"}}}}

User: {user_msg}"""

    resp2 = ollama_generate(host, model, step2_prompt)

    # Combine token counts from both steps
    if not resp2["error"]:
        resp2["prompt_eval_count"] = resp1.get("prompt_eval_count", 0) + resp2.get("prompt_eval_count", 0)
        resp2["eval_count"] = resp1.get("eval_count", 0) + resp2.get("eval_count", 0)
        resp2["elapsed_s"] = resp1.get("elapsed_s", 0) + resp2.get("elapsed_s", 0)

    return resp2


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
    param_accuracy: float
    format_valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: Optional[str]


def run_benchmark(models=None, prompts=None, strategies=None, output_path=None):
    if models is None: models = MODELS
    if prompts is None: prompts = TEST_PROMPTS
    if strategies is None: strategies = list(STRATEGIES.keys())
    if output_path is None: output_path = os.path.join(os.path.dirname(__file__), "results_v2.jsonl")

    results = []
    total = len(models) * len(prompts) * len(strategies)
    done = 0

    print(f"\n{'='*80}")
    print(f"  TIER-BASED TOOL ROUTING BENCHMARK v2")
    print(f"  Models: {len(models)} | Prompts: {len(prompts)} | Strategies: {len(strategies)} | Total: {total}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}\n")

    for model_name, model_info in models.items():
        tier = model_info["tier"]
        host = model_info["host"]

        print(f"\n--- {model_name} (Tier {tier}, {model_info['params']}) ---")

        for strat_name in strategies:
            strat_fn = STRATEGIES[strat_name]
            correct = 0

            for pidx, test in enumerate(prompts):
                result_tuple = strat_fn(TOOLS, test["prompt"], model_info)
                prompt_text, parse_mode, mcq_map = result_tuple

                # Two-step discovery: run category detection then tool selection
                if strat_name == "discovery_2step" and parse_mode == "two_step":
                    resp = run_two_step_discovery(host, model_name, model_info, TOOLS, test["prompt"])
                else:
                    resp = ollama_generate(host, model_name, prompt_text)

                if resp["error"]:
                    r = Result(model=model_name, tier=tier, params_str=model_info["params"],
                               strategy=strat_name, prompt_idx=pidx, user_prompt=test["prompt"],
                               expected_tool=test["tool"], expected_family=test["family"],
                               selected_tool=None, tool_correct=False, family_correct=False,
                               param_accuracy=0, format_valid=False,
                               prompt_tokens=0, completion_tokens=0, latency_s=0, error=resp["error"])
                else:
                    parsed = parse_response(resp["response"], parse_mode, mcq_map)
                    tool = parsed["tool"]
                    tool_correct = (tool == test["tool"])
                    # Check if at least the family is correct
                    tool_obj = next((t for t in TOOLS if t["name"] == tool), None)
                    family_correct = (tool_obj["family"] == test["family"]) if tool_obj else False

                    exp_params = test["params"]
                    if exp_params:
                        present = sum(1 for p in exp_params if p in (parsed.get("params") or {}))
                        param_acc = present / len(exp_params)
                    else:
                        param_acc = 1.0

                    if tool_correct: correct += 1

                    r = Result(model=model_name, tier=tier, params_str=model_info["params"],
                               strategy=strat_name, prompt_idx=pidx, user_prompt=test["prompt"],
                               expected_tool=test["tool"], expected_family=test["family"],
                               selected_tool=tool, tool_correct=tool_correct, family_correct=family_correct,
                               param_accuracy=param_acc,
                               format_valid=parsed["parse_error"] is None and tool is not None,
                               prompt_tokens=resp["prompt_eval_count"], completion_tokens=resp["eval_count"],
                               latency_s=resp["elapsed_s"], error=None)

                results.append(r)
                done += 1

            acc = correct / len(prompts) * 100
            print(f"  {strat_name:<12} {acc:5.1f}% ({correct}/{len(prompts)})")

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\nResults: {output_path} ({len(results)} records)")

    # Summary
    print_summary(results)
    return results


def print_summary(results):
    from collections import defaultdict
    print(f"\n{'='*100}")
    print(f"  BENCHMARK SUMMARY v2 — Tool Selection Accuracy")
    print(f"{'='*100}")

    # Table: model × strategy
    models_seen = []
    for r in results:
        key = (r.model, r.tier, r.params_str)
        if key not in models_seen: models_seen.append(key)
    strats = sorted(set(r.strategy for r in results))

    header = f"{'Model':<30} {'Tier':<8}"
    for s in strats:
        header += f" {s:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for model, tier, params in models_seen:
        row = f"{model} ({params})"
        row = f"{row:<30} {tier:<8}"
        for s in strats:
            items = [r for r in results if r.model == model and r.strategy == s]
            if items:
                acc = sum(1 for i in items if i.tool_correct) / len(items) * 100
                row += f" {acc:>11.1f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Token usage table
    print(f"\n{'='*100}")
    print(f"  TOKEN USAGE (avg prompt tokens)")
    print(f"{'='*100}")
    header = f"{'Model':<30} {'Tier':<8}"
    for s in strats:
        header += f" {s:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for model, tier, params in models_seen:
        row = f"{model} ({params})"
        row = f"{row:<30} {tier:<8}"
        for s in strats:
            items = [r for r in results if r.model == model and r.strategy == s and r.prompt_tokens > 0]
            if items:
                avg = statistics.mean([i.prompt_tokens for i in items])
                row += f" {avg:>10.0f}t"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Latency table
    print(f"\n{'='*100}")
    print(f"  LATENCY (avg seconds)")
    print(f"{'='*100}")
    header = f"{'Model':<30} {'Tier':<8}"
    for s in strats:
        header += f" {s:>12}"
    print(f"\n{header}")
    print("-" * len(header))

    for model, tier, params in models_seen:
        row = f"{model} ({params})"
        row = f"{row:<30} {tier:<8}"
        for s in strats:
            items = [r for r in results if r.model == model and r.strategy == s and r.latency_s > 0]
            if items:
                avg = statistics.mean([i.latency_s for i in items])
                row += f" {avg:>11.2f}s"
            else:
                row += f" {'N/A':>12}"
        print(row)

    # Family detection accuracy (for family and full_tier strategies)
    print(f"\n{'='*100}")
    print(f"  FAMILY DETECTION ACCURACY")
    print(f"{'='*100}")
    for model, tier, params in models_seen:
        for s in ["family", "full_tier"]:
            items = [r for r in results if r.model == model and r.strategy == s]
            if items:
                fam_acc = sum(1 for i in items if i.family_correct) / len(items) * 100
                print(f"  {model} ({params}) [{s}]: family correct {fam_acc:.1f}%")

    # Key findings
    print(f"\n{'='*100}")
    print(f"  KEY FINDINGS")
    print(f"{'='*100}")
    for model, tier, params in models_seen:
        base_items = [r for r in results if r.model == model and r.strategy == "baseline"]
        tier_items = [r for r in results if r.model == model and r.strategy == "full_tier"]
        if base_items and tier_items:
            b_acc = sum(1 for i in base_items if i.tool_correct) / len(base_items) * 100
            t_acc = sum(1 for i in tier_items if i.tool_correct) / len(tier_items) * 100
            b_tok = statistics.mean([i.prompt_tokens for i in base_items if i.prompt_tokens > 0]) if any(i.prompt_tokens > 0 for i in base_items) else 1
            t_tok = statistics.mean([i.prompt_tokens for i in tier_items if i.prompt_tokens > 0]) if any(i.prompt_tokens > 0 for i in tier_items) else 1
            delta = t_acc - b_acc
            tok_save = (1 - t_tok / b_tok) * 100 if b_tok > 0 else 0
            b_lat = statistics.mean([i.latency_s for i in base_items if i.latency_s > 0])
            t_lat = statistics.mean([i.latency_s for i in tier_items if i.latency_s > 0])
            speedup = b_lat / t_lat if t_lat > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"\n  {model} ({params}, Tier {tier}):")
            print(f"    Accuracy: {b_acc:.1f}% → {t_acc:.1f}% ({sign}{delta:.1f}pp)")
            print(f"    Tokens:   {b_tok:.0f} → {t_tok:.0f} ({tok_save:.0f}% reduction)")
            print(f"    Latency:  {b_lat:.2f}s → {t_lat:.2f}s ({speedup:.1f}x faster)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if args.quick:
        selected = {"qwen2.5:1.5b": MODELS["qwen2.5:1.5b"], "gpt-oss:20b": MODELS["gpt-oss:20b"]}
        run_benchmark(models=selected, prompts=TEST_PROMPTS[:10], strategies=args.strategies, output_path=args.output)
    elif args.models:
        selected = {k: v for k, v in MODELS.items() if k in args.models}
        run_benchmark(models=selected, strategies=args.strategies, output_path=args.output)
    else:
        run_benchmark(strategies=args.strategies, output_path=args.output)
