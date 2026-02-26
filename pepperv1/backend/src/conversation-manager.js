import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONVERSATIONS_PATH = join(__dirname, '..', 'bot', 'memory', 'conversations.json');

// In-memory state: { "1": { sessionId, createdAt, lastActivity, label, platform }, ... }
let conversations = {};

export function loadConversations() {
  if (existsSync(CONVERSATIONS_PATH)) {
    try {
      conversations = JSON.parse(readFileSync(CONVERSATIONS_PATH, 'utf-8'));
    } catch {
      conversations = {};
    }
  }
}

function save() {
  writeFileSync(CONVERSATIONS_PATH, JSON.stringify(conversations, null, 2));
}

/**
 * Parse a message for a leading conversation number.
 * "1 build a site"  → { number: 1, command: 'message', body: 'build a site' }
 * "1 close"         → { number: 1, command: 'close', body: '' }
 * "hello"           → { number: null, command: 'message', body: 'hello' }
 */
export function parseMessage(text) {
  const closeMatch = text.match(/^(\d+)\s+close$/i);
  if (closeMatch) {
    return { number: parseInt(closeMatch[1], 10), command: 'close', body: '' };
  }

  const numberedStopMatch = text.match(/^(\d+)\s+stop$/i);
  if (numberedStopMatch) {
    return { number: parseInt(numberedStopMatch[1], 10), command: 'stop', body: '' };
  }

  if (/^stop$/i.test(text)) {
    return { number: null, command: 'stop', body: '' };
  }

  const numMatch = text.match(/^(\d+)\s+(.+)$/s);
  if (numMatch) {
    return { number: parseInt(numMatch[1], 10), command: 'message', body: numMatch[2] };
  }

  return { number: null, command: 'message', body: text };
}

export function resolveSession(number) {
  const conv = conversations[String(number)];
  return conv ? conv.sessionId : null;
}

export function getConversation(number) {
  return conversations[String(number)] || null;
}

export function getConversationMode(number) {
  const conv = conversations[String(number)];
  return conv?.mode || 'assistant';
}

export function createOrUpdateConversation(number, sessionId, body, platform, mode = 'assistant') {
  const key = String(number);
  const existing = conversations[key];
  if (existing) {
    existing.sessionId = sessionId;
    existing.lastActivity = new Date().toISOString();
  } else {
    conversations[key] = {
      sessionId,
      createdAt: new Date().toISOString(),
      lastActivity: new Date().toISOString(),
      label: body.slice(0, 50),
      platform,
      mode,
    };
  }
  save();
}

export function closeConversation(number) {
  const key = String(number);
  if (conversations[key]) {
    delete conversations[key];
    save();
    return true;
  }
  return false;
}

export function listConversations() {
  return Object.entries(conversations).map(([number, data]) => ({
    number: parseInt(number, 10),
    ...data,
  }));
}

// Load on import
loadConversations();
