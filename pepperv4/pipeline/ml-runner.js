// ml-runner.js — Node.js integration layer for the local ML inference subprocess.
// Keeps a single persistent Python subprocess alive (lazy init) and communicates
// via newline-delimited JSON over stdin/stdout.

import { spawn } from 'child_process';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const INFER_SCRIPT = join(__dirname, '../ml/infer.py');
const PYTHON = process.env.PEPPER_PYTHON || 'python';
const CALL_TIMEOUT_MS = 10000;

let proc = null;
let stdoutBuffer = '';
// Queue of pending calls: { resolve, reject, timer }
const pendingQueue = [];

function startProcess() {
  proc = spawn(PYTHON, [INFER_SCRIPT], {
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: false,
  });

  proc.stdout.on('data', chunk => {
    stdoutBuffer += chunk.toString();
    const lines = stdoutBuffer.split('\n');
    stdoutBuffer = lines.pop(); // keep incomplete last chunk
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const pending = pendingQueue.shift();
      if (!pending) {
        process.stderr.write(`[ml-runner] Unexpected output: ${trimmed}\n`);
        continue;
      }
      clearTimeout(pending.timer);
      try {
        pending.resolve(JSON.parse(trimmed));
      } catch (e) {
        pending.reject(new Error(`[ml-runner] JSON parse failed: ${trimmed}`));
      }
    }
  });

  proc.stderr.on('data', chunk => {
    process.stderr.write(`[ml-runner] ${chunk}`);
  });

  proc.on('close', code => {
    process.stderr.write(`[ml-runner] subprocess exited (code ${code})\n`);
    proc = null;
    // Reject any pending calls
    while (pendingQueue.length > 0) {
      const pending = pendingQueue.shift();
      clearTimeout(pending.timer);
      pending.reject(new Error(`[ml-runner] subprocess exited unexpectedly (code ${code})`));
    }
  });

  proc.on('error', err => {
    process.stderr.write(`[ml-runner] spawn error: ${err.message}\n`);
    proc = null;
    while (pendingQueue.length > 0) {
      const pending = pendingQueue.shift();
      clearTimeout(pending.timer);
      pending.reject(err);
    }
  });
}

function ensureProcess() {
  if (!proc || proc.killed) {
    startProcess();
  }
}

function call(payload) {
  return new Promise((resolve, reject) => {
    ensureProcess();
    const timer = setTimeout(() => {
      // Remove from queue and reject
      const idx = pendingQueue.findIndex(p => p.timer === timer);
      if (idx !== -1) pendingQueue.splice(idx, 1);
      reject(new Error(`[ml-runner] timeout after ${CALL_TIMEOUT_MS}ms for task: ${payload.task}`));
    }, CALL_TIMEOUT_MS);

    pendingQueue.push({ resolve, reject, timer });
    try {
      proc.stdin.write(JSON.stringify(payload) + '\n');
    } catch (err) {
      clearTimeout(timer);
      pendingQueue.pop();
      reject(err);
    }
  });
}

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Phase A: classify the prompt into output type labels.
 * Returns a JSON string compatible with parseOutputSpec().
 */
export async function runPhaseA(prompt) {
  try {
    const result = await call({ task: 'phase_a', prompt });
    return JSON.stringify(result);
  } catch (err) {
    process.stderr.write(`[ml-runner] Phase A error: ${err.message}\n`);
    // Return a safe fallback that parseOutputSpec can handle
    return JSON.stringify({
      taskDescription: (prompt || '').slice(0, 500),
      outputType: 'text',
      outputLabels: { text: true, picture: false, command: false, presentation: false, specificFile: false, other: false },
      outputFormat: { type: 'inline_text', structure: 'direct answer', deliveryMethod: 'inline' },
      requiredDomains: [],
      complexity: 'simple',
      estimatedSteps: 1,
    });
  }
}

/**
 * Phase B: retrieve relevant memory files via TF-IDF cosine similarity.
 * inventory is the array from getFullInventory().
 * Returns a JSON string compatible with parseAuditResult().
 */
export async function runPhaseB(prompt, inventory) {
  try {
    const result = await call({ task: 'phase_b', prompt, inventory });
    return JSON.stringify(result);
  } catch (err) {
    process.stderr.write(`[ml-runner] Phase B error: ${err.message}\n`);
    return JSON.stringify({
      selectedMemories: [],
      missingMemories: [],
      toolsNeeded: [],
      notes: `ML retrieval failed: ${err.message}`,
    });
  }
}

/**
 * Gracefully shut down the Python subprocess (useful for tests / clean exit).
 */
export function shutdown() {
  if (proc && !proc.killed) {
    proc.stdin.end();
    proc = null;
  }
}
