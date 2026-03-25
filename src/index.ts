/**
 * Tier — Durable Event Bus Plugin for OpenClaw
 *
 * Bridges to the Python TierEngine via persistent subprocess
 * for high-throughput event streaming (1000+ events/sec).
 */

import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { createInterface } from "node:readline";

const PLUGIN_DIR = path.resolve(__dirname, "..");
const BRIDGE_SCRIPT = path.join(PLUGIN_DIR, "tier_engine", "bridge.py");

interface BridgeResult {
  success: boolean;
  request_id?: string;
  error?: string;
  [key: string]: unknown;
}

// ── Persistent Subprocess Manager ─────────────────────────────────────

let _process: ChildProcess | null = null;
let _pendingRequests: Map<string, { resolve: Function; reject: Function }> =
  new Map();
let _requestCounter = 0;
let _ready = false;

function ensureProcess(): ChildProcess {
  if (_process && !_process.killed) {
    return _process;
  }

  if (!fs.existsSync(BRIDGE_SCRIPT)) {
    throw new Error(
      `bridge.py not found at ${BRIDGE_SCRIPT}. Install nexus-eventbus.`
    );
  }

  const proc = spawn("python3", [BRIDGE_SCRIPT, "--persistent"], {
    cwd: PLUGIN_DIR,
    stdio: ["pipe", "pipe", "pipe"],
  });

  const rl = createInterface({ input: proc.stdout! });

  rl.on("line", (line: string) => {
    try {
      const result: BridgeResult = JSON.parse(line);
      const rid = result.request_id;
      if (rid && _pendingRequests.has(rid)) {
        const { resolve } = _pendingRequests.get(rid)!;
        _pendingRequests.delete(rid);
        resolve(result);
      }
    } catch {
      // Ignore non-JSON output
    }
  });

  proc.stderr?.on("data", (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) {
      console.error(`[tier] ${msg}`);
    }
  });

  proc.on("close", (code: number | null) => {
    _process = null;
    _ready = false;
    // Reject all pending requests
    for (const [rid, { reject }] of _pendingRequests) {
      reject(new Error(`Tier bridge exited with code ${code}`));
    }
    _pendingRequests.clear();
  });

  _process = proc;
  _ready = true;
  return proc;
}

function sendCommand(
  command: string,
  args: Record<string, unknown> = {},
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return new Promise((resolve, reject) => {
    try {
      const proc = ensureProcess();
      const requestId = `r${++_requestCounter}`;
      const payload =
        JSON.stringify({ request_id: requestId, command, args, config }) + "\n";

      _pendingRequests.set(requestId, { resolve, reject });

      // Timeout after 30 seconds
      setTimeout(() => {
        if (_pendingRequests.has(requestId)) {
          _pendingRequests.delete(requestId);
          reject(new Error(`Tier bridge timeout for command: ${command}`));
        }
      }, 30000);

      proc.stdin!.write(payload);
    } catch (err) {
      reject(err);
    }
  });
}

function killProcess(): void {
  if (_process && !_process.killed) {
    _process.kill("SIGTERM");
    _process = null;
    _ready = false;
  }
}

// ── Plugin Lifecycle ──────────────────────────────────────────────────

export async function onStartup(context: {
  config: Record<string, unknown>;
}): Promise<void> {
  try {
    const result = await sendCommand("health_check", {}, context.config);
    if (result.healthy) {
      const stats = await sendCommand("stats", {}, context.config);
      console.log(
        `[tier] Event bus ready: ${stats.total_events ?? 0} events, ${stats.consumer_groups ?? 0} groups`
      );
    } else {
      console.error(`[tier] Health check failed: ${result.error}`);
    }
  } catch (err) {
    console.error(`[tier] Startup error: ${(err as Error).message}`);
  }
}

export async function onShutdown(): Promise<void> {
  killProcess();
  console.log("[tier] Event bus shut down");
}

// ── Public API ────────────────────────────────────────────────────────

export async function publish(
  eventType: string,
  payload: Record<string, unknown>,
  source: string,
  options: {
    correlationId?: string;
    causationId?: string;
    metadata?: Record<string, unknown>;
    partitionKey?: string;
    config?: Record<string, unknown>;
  } = {}
): Promise<BridgeResult> {
  return sendCommand(
    "publish",
    {
      event_type: eventType,
      payload,
      source,
      correlation_id: options.correlationId,
      causation_id: options.causationId,
      metadata: options.metadata,
      partition_key: options.partitionKey || "default",
    },
    options.config || {}
  );
}

export async function publishBatch(
  events: Array<{
    event_type: string;
    payload: Record<string, unknown>;
    source: string;
    correlation_id?: string;
    metadata?: Record<string, unknown>;
    partition_key?: string;
  }>,
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand("publish_batch", { events }, config);
}

export async function createGroup(
  groupId: string,
  description: string = "",
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "create_group",
    { group_id: groupId, description },
    config
  );
}

export async function subscribe(
  groupId: string,
  eventFilter: { event_types?: string[]; sources?: string[] },
  options: {
    maxRetries?: number;
    retryDelayMs?: number;
    priority?: number;
    config?: Record<string, unknown>;
  } = {}
): Promise<BridgeResult> {
  return sendCommand(
    "subscribe",
    {
      group_id: groupId,
      event_filter: eventFilter,
      max_retries: options.maxRetries ?? 3,
      retry_delay_ms: options.retryDelayMs ?? 1000,
      priority: options.priority ?? 0,
    },
    options.config || {}
  );
}

export async function poll(
  groupId: string,
  options: {
    limit?: number;
    partitionKey?: string;
    config?: Record<string, unknown>;
  } = {}
): Promise<BridgeResult> {
  return sendCommand(
    "poll",
    {
      group_id: groupId,
      limit: options.limit ?? 100,
      partition_key: options.partitionKey || "default",
    },
    options.config || {}
  );
}

export async function acknowledge(
  groupId: string,
  sequenceId: number,
  partitionKey: string = "default",
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "acknowledge",
    { group_id: groupId, sequence_id: sequenceId, partition_key: partitionKey },
    config
  );
}

export async function nack(
  groupId: string,
  eventId: string,
  subscriptionId: string,
  error: string,
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "nack",
    {
      group_id: groupId,
      event_id: eventId,
      subscription_id: subscriptionId,
      error,
    },
    config
  );
}

export async function replay(
  options: {
    fromSequence?: number;
    toSequence?: number;
    eventType?: string;
    source?: string;
    partitionKey?: string;
    since?: string;
    limit?: number;
    config?: Record<string, unknown>;
  } = {}
): Promise<BridgeResult> {
  return sendCommand(
    "replay",
    {
      from_sequence: options.fromSequence ?? 0,
      to_sequence: options.toSequence,
      event_type: options.eventType,
      source: options.source,
      partition_key: options.partitionKey,
      since: options.since,
      limit: options.limit ?? 1000,
    },
    options.config || {}
  );
}

export async function resetCheckpoint(
  groupId: string,
  toSequence: number = 0,
  partitionKey: string = "default",
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "reset_checkpoint",
    {
      group_id: groupId,
      partition_key: partitionKey,
      to_sequence: toSequence,
    },
    config
  );
}

export async function getLag(
  groupId: string,
  partitionKey: string = "default",
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "get_lag",
    { group_id: groupId, partition_key: partitionKey },
    config
  );
}

export async function registerSchema(
  eventType: string,
  version: string,
  schemaDef: Record<string, unknown>,
  description: string = "",
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "register_schema",
    { event_type: eventType, version, schema_def: schemaDef, description },
    config
  );
}

export async function getDlq(
  groupId?: string,
  limit: number = 50,
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand(
    "get_dlq",
    { group_id: groupId, limit },
    config
  );
}

export async function retryDlq(
  dlqId: number,
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand("retry_dlq", { dlq_id: dlqId }, config);
}

export async function resolveDlq(
  dlqId: number,
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand("resolve_dlq", { dlq_id: dlqId }, config);
}

export async function stats(
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand("stats", {}, config);
}

export async function healthCheck(
  config: Record<string, unknown> = {}
): Promise<BridgeResult> {
  return sendCommand("health_check", {}, config);
}

export async function compact(
  options: {
    beforeSequence?: number;
    beforeDate?: string;
    config?: Record<string, unknown>;
  } = {}
): Promise<BridgeResult> {
  return sendCommand(
    "compact",
    {
      before_sequence: options.beforeSequence,
      before_date: options.beforeDate,
    },
    options.config || {}
  );
}
