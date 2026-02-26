// Clarification manager — handles needs_user_input → user answers → resumed flows.
// Persists pending clarification state to bot/memory/clarifications.json.

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const STORE_PATH = join(__dirname, '..', '..', 'pepperv1', 'backend', 'bot', 'memory', 'clarifications.json');

let store = null;

function load() {
  if (store) return store;
  try {
    if (existsSync(STORE_PATH)) {
      store = JSON.parse(readFileSync(STORE_PATH, 'utf-8'));
    } else {
      store = {};
    }
  } catch {
    store = {};
  }
  return store;
}

function save() {
  try {
    mkdirSync(dirname(STORE_PATH), { recursive: true });
    writeFileSync(STORE_PATH, JSON.stringify(store, null, 2), 'utf-8');
  } catch {}
}

export function get(key) {
  const data = load();
  return data[key] || null;
}

export function setPending(key, { originalPrompt, pendingQuestions, sessionId }) {
  load();
  store[key] = {
    originalPrompt,
    pendingQuestions,
    sessionId,
    answers: [],
    timestamp: Date.now(),
  };
  save();
}

export function appendAnswer(key, answer) {
  load();
  if (!store[key]) return;
  store[key].answers.push(answer);
  save();
}

export function buildAugmentedPrompt(entry) {
  const parts = [entry.originalPrompt];

  if (entry.answers.length > 0) {
    parts.push('\n\n[Previous clarification Q&A]:');
    for (let i = 0; i < entry.answers.length; i++) {
      const q = entry.pendingQuestions?.questions?.[i]?.question || `Question ${i + 1}`;
      parts.push(`Q: ${q}`);
      parts.push(`A: ${entry.answers[i]}`);
    }
    parts.push('\nPlease continue with the task using the above answers.');
  }

  return parts.join('\n');
}

export function clear(key) {
  load();
  delete store[key];
  save();
}
