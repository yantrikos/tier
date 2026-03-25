// src/index.ts
import { spawn } from "node:child_process";
import path from "node:path";
import fs from "node:fs";
import { createInterface } from "node:readline";
var PLUGIN_DIR = path.resolve(__dirname, "..");
var BRIDGE_SCRIPT = path.join(PLUGIN_DIR, "tier_engine", "bridge.py");
var _process = null;
var _pendingRequests = /* @__PURE__ */ new Map();
var _requestCounter = 0;
var _ready = false;
function ensureProcess() {
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
    stdio: ["pipe", "pipe", "pipe"]
  });
  const rl = createInterface({ input: proc.stdout });
  rl.on("line", (line) => {
    try {
      const result = JSON.parse(line);
      const rid = result.request_id;
      if (rid && _pendingRequests.has(rid)) {
        const { resolve } = _pendingRequests.get(rid);
        _pendingRequests.delete(rid);
        resolve(result);
      }
    } catch {
    }
  });
  proc.stderr?.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) {
      console.error(`[tier] ${msg}`);
    }
  });
  proc.on("close", (code) => {
    _process = null;
    _ready = false;
    for (const [rid, { reject }] of _pendingRequests) {
      reject(new Error(`Tier bridge exited with code ${code}`));
    }
    _pendingRequests.clear();
  });
  _process = proc;
  _ready = true;
  return proc;
}
function sendCommand(command, args = {}, config = {}) {
  return new Promise((resolve, reject) => {
    try {
      const proc = ensureProcess();
      const requestId = `r${++_requestCounter}`;
      const payload = JSON.stringify({ request_id: requestId, command, args, config }) + "\n";
      _pendingRequests.set(requestId, { resolve, reject });
      setTimeout(() => {
        if (_pendingRequests.has(requestId)) {
          _pendingRequests.delete(requestId);
          reject(new Error(`Tier bridge timeout for command: ${command}`));
        }
      }, 3e4);
      proc.stdin.write(payload);
    } catch (err) {
      reject(err);
    }
  });
}
function killProcess() {
  if (_process && !_process.killed) {
    _process.kill("SIGTERM");
    _process = null;
    _ready = false;
  }
}
async function onStartup(context) {
  try {
    const result = await sendCommand("health_check", {}, context.config);
    if (result.healthy) {
      const stats2 = await sendCommand("stats", {}, context.config);
      console.log(
        `[tier] Event bus ready: ${stats2.total_events ?? 0} events, ${stats2.consumer_groups ?? 0} groups`
      );
    } else {
      console.error(`[tier] Health check failed: ${result.error}`);
    }
  } catch (err) {
    console.error(`[tier] Startup error: ${err.message}`);
  }
}
async function onShutdown() {
  killProcess();
  console.log("[tier] Event bus shut down");
}
async function publish(eventType, payload, source, options = {}) {
  return sendCommand(
    "publish",
    {
      event_type: eventType,
      payload,
      source,
      correlation_id: options.correlationId,
      causation_id: options.causationId,
      metadata: options.metadata,
      partition_key: options.partitionKey || "default"
    },
    options.config || {}
  );
}
async function publishBatch(events, config = {}) {
  return sendCommand("publish_batch", { events }, config);
}
async function createGroup(groupId, description = "", config = {}) {
  return sendCommand(
    "create_group",
    { group_id: groupId, description },
    config
  );
}
async function subscribe(groupId, eventFilter, options = {}) {
  return sendCommand(
    "subscribe",
    {
      group_id: groupId,
      event_filter: eventFilter,
      max_retries: options.maxRetries ?? 3,
      retry_delay_ms: options.retryDelayMs ?? 1e3,
      priority: options.priority ?? 0
    },
    options.config || {}
  );
}
async function poll(groupId, options = {}) {
  return sendCommand(
    "poll",
    {
      group_id: groupId,
      limit: options.limit ?? 100,
      partition_key: options.partitionKey || "default"
    },
    options.config || {}
  );
}
async function acknowledge(groupId, sequenceId, partitionKey = "default", config = {}) {
  return sendCommand(
    "acknowledge",
    { group_id: groupId, sequence_id: sequenceId, partition_key: partitionKey },
    config
  );
}
async function nack(groupId, eventId, subscriptionId, error, config = {}) {
  return sendCommand(
    "nack",
    {
      group_id: groupId,
      event_id: eventId,
      subscription_id: subscriptionId,
      error
    },
    config
  );
}
async function replay(options = {}) {
  return sendCommand(
    "replay",
    {
      from_sequence: options.fromSequence ?? 0,
      to_sequence: options.toSequence,
      event_type: options.eventType,
      source: options.source,
      partition_key: options.partitionKey,
      since: options.since,
      limit: options.limit ?? 1e3
    },
    options.config || {}
  );
}
async function resetCheckpoint(groupId, toSequence = 0, partitionKey = "default", config = {}) {
  return sendCommand(
    "reset_checkpoint",
    {
      group_id: groupId,
      partition_key: partitionKey,
      to_sequence: toSequence
    },
    config
  );
}
async function getLag(groupId, partitionKey = "default", config = {}) {
  return sendCommand(
    "get_lag",
    { group_id: groupId, partition_key: partitionKey },
    config
  );
}
async function registerSchema(eventType, version, schemaDef, description = "", config = {}) {
  return sendCommand(
    "register_schema",
    { event_type: eventType, version, schema_def: schemaDef, description },
    config
  );
}
async function getDlq(groupId, limit = 50, config = {}) {
  return sendCommand(
    "get_dlq",
    { group_id: groupId, limit },
    config
  );
}
async function retryDlq(dlqId, config = {}) {
  return sendCommand("retry_dlq", { dlq_id: dlqId }, config);
}
async function resolveDlq(dlqId, config = {}) {
  return sendCommand("resolve_dlq", { dlq_id: dlqId }, config);
}
async function stats(config = {}) {
  return sendCommand("stats", {}, config);
}
async function healthCheck(config = {}) {
  return sendCommand("health_check", {}, config);
}
async function compact(options = {}) {
  return sendCommand(
    "compact",
    {
      before_sequence: options.beforeSequence,
      before_date: options.beforeDate
    },
    options.config || {}
  );
}
export {
  acknowledge,
  compact,
  createGroup,
  getDlq,
  getLag,
  healthCheck,
  nack,
  onShutdown,
  onStartup,
  poll,
  publish,
  publishBatch,
  registerSchema,
  replay,
  resetCheckpoint,
  resolveDlq,
  retryDlq,
  stats,
  subscribe
};
