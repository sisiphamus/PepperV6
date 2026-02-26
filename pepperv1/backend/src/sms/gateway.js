import { config } from '../config.js';
import { executeClaudePrompt, killProcess, clearClarificationState } from '../claude-bridge.js';
import { parseMessage, resolveSession, createOrUpdateConversation, closeConversation } from '../conversation-manager.js';
import { addToLogIndex, nextLogNumber } from '../index.js';
import { createRuntimeAwareProgress } from '../runtime-health.js';
import { extractImages } from '../transport-utils.js';
import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOGS_DIR = join(__dirname, '..', '..', 'bot', 'logs');

const SESSION_TIMEOUT_MS = 10 * 60 * 1000;
let io = null;

// Track active sessions per phone number for conversation continuity
// Map<phoneNumber, { sessionId, lastActivity }>
const chatSessions = new Map();

// Per-conversation message queue to prevent concurrent processing collisions
// Map<queueKey, Promise> — keyed by "conv:<number>" or "chat:<phoneNumber>"
const chatQueues = new Map();

// Track how many conversations are actively running (for cleanup gating)
let activeCount = 0;

function setSmsSocketIO(socketIO) {
  io = socketIO;
}

function emitLog(type, data) {
  const entry = { type, data, timestamp: new Date().toISOString() };
  io?.emit('log', entry);
  console.log(`[sms:${type}]`, JSON.stringify(data));
}

function isAllowed(phoneNumber) {
  if (!config.smsAllowedNumbers || config.smsAllowedNumbers.length === 0) return true;
  return config.smsAllowedNumbers.includes(phoneNumber);
}

function getQueueKey(phoneNumber, text) {
  let prompt = text;
  if (config.smsPrefix && prompt.startsWith(config.smsPrefix)) {
    prompt = prompt.slice(config.smsPrefix.length).trim();
  } else if (config.smsPrefix) {
    return null;
  }

  if (!prompt) return null;

  const parsed = parseMessage(prompt);
  if (parsed.number !== null) {
    return `conv:${parsed.number}`;
  }
  return `chat:${phoneNumber}`;
}

function formatQuestionsForSms(questionsPayload) {
  const questions = Array.isArray(questionsPayload?.questions) ? questionsPayload.questions : [];
  if (!questions.length) {
    return 'I need more detail before I continue. Reply with the missing info.';
  }
  return [
    'I need a few details before I continue:',
    ...questions.map((q, i) => `${i + 1}. ${q.question || 'Please clarify'}`),
    'Reply with your answers and I will continue.',
  ].join('\n');
}

async function sendSms(phoneNumber, text) {
  const url = config.smsGatewayUrl;
  if (!url) return;

  const auth = Buffer.from(`${config.smsGatewayUsername}:${config.smsGatewayPassword}`).toString('base64');

  // Chunk at 1600 chars for SMS
  const chunks = [];
  for (let i = 0; i < text.length; i += 1600) {
    chunks.push(text.slice(i, i + 1600));
  }

  for (const chunk of chunks) {
    try {
      const res = await fetch(`${url}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Basic ${auth}`,
        },
        body: JSON.stringify({ message: chunk, phoneNumbers: [phoneNumber] }),
      });
      if (!res.ok) {
        const body = await res.text();
        emitLog('send_error', { phoneNumber, status: res.status, body });
      }
    } catch (err) {
      emitLog('send_error', { phoneNumber, error: err.message });
    }
  }
}

async function registerWebhook(webhookUrl) {
  const url = config.smsGatewayUrl;
  const auth = Buffer.from(`${config.smsGatewayUsername}:${config.smsGatewayPassword}`).toString('base64');

  try {
    const res = await fetch(`${url}/webhooks`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Basic ${auth}`,
      },
      body: JSON.stringify({ url: webhookUrl, event: 'sms:received' }),
    });

    if (res.status === 409) {
      emitLog('webhook', { message: 'Webhook already registered', webhookUrl });
    } else if (res.ok) {
      emitLog('webhook', { message: 'Webhook registered', webhookUrl });
    } else {
      const body = await res.text();
      emitLog('webhook_error', { status: res.status, body });
    }
  } catch (err) {
    emitLog('webhook_error', { error: err.message });
  }
}

async function handleStopCommand(phoneNumber, number) {
  const processKey = number !== null ? `sms:conv:${number}` : `sms:chat:${phoneNumber}`;
  const killed = killProcess(processKey);
  clearClarificationState(processKey);
  if (killed) {
    const label = number !== null ? `conversation #${number}` : 'current conversation';
    await sendSms(phoneNumber, `Stopped ${label}.`);
  } else {
    const label = number !== null ? `conversation #${number}` : 'this chat';
    await sendSms(phoneNumber, `Nothing running for ${label}.`);
  }
}

function handleIncomingSms(payload) {
  const phoneNumber = payload.phoneNumber;
  const message = payload.message;
  if (!phoneNumber || !message) return;

  // Strip prefix if configured
  let text = message;
  if (config.smsPrefix && text.startsWith(config.smsPrefix)) {
    text = text.slice(config.smsPrefix.length).trim();
  } else if (config.smsPrefix) {
    return; // prefix configured but message doesn't match
  }

  if (!text) return;

  // Intercept stop commands before queueing
  const parsed = parseMessage(text);
  if (parsed.command === 'stop') {
    handleStopCommand(phoneNumber, parsed.number);
    return;
  }

  const key = getQueueKey(phoneNumber, message);
  if (!key) return;

  // Chain onto the existing queue for this key
  const prev = chatQueues.get(key) || Promise.resolve();
  const next = prev.then(() => processIncomingSms(phoneNumber, message)).catch(() => {});
  chatQueues.set(key, next);
}

async function processIncomingSms(phoneNumber, text) {
  // Check allowlist
  if (!isAllowed(phoneNumber)) {
    emitLog('blocked', { phoneNumber, reason: 'not in allowed list' });
    return;
  }

  // Strip prefix
  let prompt = text;
  if (config.smsPrefix && prompt.startsWith(config.smsPrefix)) {
    prompt = prompt.slice(config.smsPrefix.length).trim();
  } else if (config.smsPrefix) {
    return;
  }

  if (!prompt) return;

  // Handle /new command
  if (prompt.toLowerCase() === '/new') {
    chatSessions.delete(phoneNumber);
    clearClarificationState(`sms:chat:${phoneNumber}`);
    await sendSms(phoneNumber, 'Session cleared. Next message starts fresh.');
    return;
  }

  // Parse for numbered conversation prefix
  const parsed = parseMessage(prompt);

  // Handle close command
  if (parsed.command === 'close') {
    const closed = closeConversation(parsed.number);
    clearClarificationState(`sms:conv:${parsed.number}`);
    await sendSms(phoneNumber, closed
      ? `Conversation #${parsed.number} closed.`
      : `No active conversation #${parsed.number}.`);
    return;
  }

  // Determine session
  let resumeSessionId = null;
  if (parsed.number !== null) {
    resumeSessionId = resolveSession(parsed.number);
    emitLog('incoming', { phoneNumber, prompt: parsed.body, conversation: parsed.number, resuming: resumeSessionId });
  } else {
    const existing = chatSessions.get(phoneNumber);
    if (existing && (Date.now() - existing.lastActivity) < SESSION_TIMEOUT_MS) {
      resumeSessionId = existing.sessionId;
      emitLog('incoming', { phoneNumber, prompt: parsed.body, resuming: existing.sessionId });
    } else {
      emitLog('incoming', { phoneNumber, prompt: parsed.body });
    }
  }

  const processKey = parsed.number !== null ? `sms:conv:${parsed.number}` : `sms:chat:${phoneNumber}`;
  const convoLog = { sender: `sms_${phoneNumber}`, prompt: parsed.body, phoneNumber, conversationNumber: parsed.number, resumeSessionId, timestamp: new Date().toISOString(), events: [] };

  activeCount++;
  try {
    const progressWrapper = createRuntimeAwareProgress((type, data) => {
      emitLog(type, { phoneNumber, ...data });
      convoLog.events.push({ type, ...data });
    });
    const onProgress = progressWrapper.onProgress;
    convoLog.runtimeFingerprint = progressWrapper.health.bootFingerprint;
    convoLog.runtimeStaleDetected = progressWrapper.health.stale;
    convoLog.runtimeChangedFiles = progressWrapper.health.changedFiles;
    if (progressWrapper.health.stale) {
      emitLog('runtime_stale_code_detected', { phoneNumber, changedFiles: progressWrapper.health.changedFiles });
    }
    const execResult = await executeClaudePrompt(parsed.body, { onProgress, resumeSessionId, processKey, clarificationKey: processKey });
    if (execResult.status === 'needs_user_input') {
      convoLog.clarificationState = { status: 'needs_user_input', questions: execResult.questions };
      const clarificationText = formatQuestionsForSms(execResult.questions);
      await sendSms(phoneNumber, clarificationText);
      emitLog('clarification_requested', { phoneNumber, conversation: parsed.number });
      return;
    }
    const { response, sessionId, fullEvents } = execResult;

    if (sessionId) {
      if (parsed.number !== null) {
        createOrUpdateConversation(parsed.number, sessionId, parsed.body, 'sms');
      }
      chatSessions.set(phoneNumber, { sessionId, lastActivity: Date.now() });
    }

    convoLog.response = response;
    convoLog.sessionId = sessionId;
    convoLog.fullEvents = fullEvents;
    emitLog('response', { phoneNumber, prompt: parsed.body, responseLength: response.length, sessionId, conversation: parsed.number });
    const { cleanText: smsText } = extractImages(response);
    await sendSms(phoneNumber, smsText || response);
    emitLog('sent', { to: phoneNumber, responseLength: response.length });
  } catch (err) {
    if (err.stopped) {
      // Stop handler already sent a message
    } else if (resumeSessionId) {
      emitLog('resume_failed', { phoneNumber, error: err.message, fallback: 'fresh session' });
      chatSessions.delete(phoneNumber);
      try {
        const progressWrapper = createRuntimeAwareProgress((type, data) => {
          emitLog(type, { phoneNumber, ...data });
          convoLog.events.push({ type, ...data });
        });
        const onProgress = progressWrapper.onProgress;
        convoLog.runtimeFingerprint = progressWrapper.health.bootFingerprint;
        convoLog.runtimeStaleDetected = progressWrapper.health.stale;
        convoLog.runtimeChangedFiles = progressWrapper.health.changedFiles;
        if (progressWrapper.health.stale) {
          emitLog('runtime_stale_code_detected', { phoneNumber, changedFiles: progressWrapper.health.changedFiles });
        }
        const execResult = await executeClaudePrompt(parsed.body, { onProgress, processKey, clarificationKey: processKey });
        if (execResult.status === 'needs_user_input') {
          convoLog.clarificationState = { status: 'needs_user_input', questions: execResult.questions };
          const clarificationText = formatQuestionsForSms(execResult.questions);
          await sendSms(phoneNumber, clarificationText);
          emitLog('clarification_requested', { phoneNumber, conversation: parsed.number });
          return;
        }
        const { response, sessionId, fullEvents } = execResult;
        if (sessionId) {
          if (parsed.number !== null) {
            createOrUpdateConversation(parsed.number, sessionId, parsed.body, 'sms');
          }
          chatSessions.set(phoneNumber, { sessionId, lastActivity: Date.now() });
        }
        convoLog.response = response;
        convoLog.sessionId = sessionId;
        convoLog.fullEvents = fullEvents;
        emitLog('response', { phoneNumber, prompt: parsed.body, responseLength: response.length, sessionId, conversation: parsed.number });
        const { cleanText: retrySmsText } = extractImages(response);
        await sendSms(phoneNumber, retrySmsText || response);
        emitLog('sent', { to: phoneNumber, responseLength: response.length });
      } catch (retryErr) {
        convoLog.error = retryErr.message;
        emitLog('error', { phoneNumber, prompt: parsed.body, error: retryErr.message });
        await sendSms(phoneNumber, `Error: ${retryErr.message}`);
      }
    } else {
      convoLog.error = err.message;
      emitLog('error', { phoneNumber, prompt: parsed.body, error: err.message });
      await sendSms(phoneNumber, `Error: ${err.message}`);
    }
  } finally {
    activeCount--;
  }

  // Persist conversation log
  try {
    mkdirSync(LOGS_DIR, { recursive: true });
    const sanitized = phoneNumber.replace(/[^a-zA-Z0-9+]/g, '');
    const filename = `${nextLogNumber()}_sms_${sanitized}.json`;
    writeFileSync(join(LOGS_DIR, filename), JSON.stringify(convoLog, null, 2));
    addToLogIndex(filename, convoLog);
    io?.emit('conversation_update', { sessionId: convoLog.sessionId, conversationNumber: parsed.number });
  } catch (e) {
    console.log('[sms:log_write_error]', e.message);
  }
}

async function startSmsGateway(webhookBaseUrl) {
  if (!config.smsGatewayUrl || !config.smsGatewayUsername || !config.smsGatewayPassword) {
    console.log('\n  SMS Gateway not configured. Set smsGatewayUrl/Username/Password in config.');
    console.log('  Install Android SMS Gateway by capcom6 on your phone.');
    console.log('  See: https://github.com/capcom6/android-sms-gateway\n');
    return false;
  }

  const webhookUrl = `${webhookBaseUrl}/api/sms/webhook`;
  await registerWebhook(webhookUrl);

  emitLog('connected', { message: `SMS Gateway connected via ${config.smsGatewayUrl}` });
  console.log(`\n  SMS Gateway: ${config.smsGatewayUrl}`);
  console.log(`  Webhook: ${webhookUrl}\n`);

  return true;
}

function stopSmsGateway() {
  // Webhook is passive — nothing to stop
}

export { startSmsGateway, stopSmsGateway, setSmsSocketIO, handleIncomingSms };
