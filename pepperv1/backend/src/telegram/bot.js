import { config } from '../config.js';
import { executeClaudePrompt, killProcess, codeAgentOptions, clearClarificationState } from '../claude-bridge.js';
import { parseMessage, resolveSession, createOrUpdateConversation, closeConversation, getConversationMode } from '../conversation-manager.js';
import { addToLogIndex, nextLogNumber } from '../index.js';
import { createRuntimeAwareProgress } from '../runtime-health.js';
import { extractImages } from '../transport-utils.js';
import { writeFileSync, mkdirSync, readdirSync, unlinkSync, readFileSync, existsSync } from 'fs';
import { join, dirname, basename } from 'path';
import { fileURLToPath } from 'url';
import { randomBytes } from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOGS_DIR = join(__dirname, '..', '..', 'bot', 'logs');
const SHORT_TERM_DIR = join(__dirname, '..', '..', 'bot', 'memory', 'short-term');

const TELEGRAM_API = 'https://api.telegram.org/bot';
const SESSION_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes — after this, start a fresh conversation
const CHAT_SESSIONS_PATH = join(__dirname, '..', '..', 'bot', 'memory', 'chat-sessions.json');
let polling = false;
let offset = 0;
let io = null;

// Track active sessions per chat for conversation continuity
// Map<chatId, { sessionId, lastActivity }>
const chatSessions = new Map();

function loadChatSessions() {
  try {
    if (existsSync(CHAT_SESSIONS_PATH)) {
      const data = JSON.parse(readFileSync(CHAT_SESSIONS_PATH, 'utf-8'));
      for (const [key, value] of Object.entries(data)) {
        chatSessions.set(Number(key), value);
      }
    }
  } catch {}
}

function saveChatSessions() {
  try {
    const obj = {};
    for (const [key, value] of chatSessions) {
      obj[key] = value;
    }
    writeFileSync(CHAT_SESSIONS_PATH, JSON.stringify(obj, null, 2));
  } catch {}
}

loadChatSessions();

// Per-conversation message queue to prevent concurrent processing collisions
// Map<queueKey, Promise> — keyed by "conv:<number>" or "chat:<chatId>"
const chatQueues = new Map();

// Track how many conversations are actively running (for cleanup gating)
let activeCount = 0;

function setSocketIO(socketIO) {
  io = socketIO;
}

// Determine the queue key for an incoming update:
// - Numbered messages get "conv:<number>" so different numbers run in parallel
// - Unnumbered messages get "chat:<chatId>" preserving per-chat serialization
function getQueueKey(update) {
  const msg = update.message;
  const hasPhoto = !!(msg?.photo?.length);
  if (!msg?.text && !msg?.caption && !hasPhoto) return null;

  let prompt = msg.text || msg.caption || '';
  if (config.telegramPrefix && prompt.startsWith(config.telegramPrefix)) {
    prompt = prompt.slice(config.telegramPrefix.length).trim();
  } else if (config.telegramPrefix && !hasPhoto) {
    return null; // prefix configured but message doesn't match (allow photos through)
  }

  if (!prompt && !hasPhoto) return null;

  const parsed = parseMessage(prompt || 'What is this image?');
  if (parsed.number !== null) {
    return `conv:${parsed.number}`;
  }
  return `chat:${msg.chat.id}`;
}

/**
 * Downloads a photo from a Telegram message and saves to short-term memory.
 * Returns the file path or null.
 */
async function downloadTelegramPhoto(msg) {
  if (!msg.photo?.length) return null;

  try {
    // Telegram sends multiple sizes — grab the largest
    const photo = msg.photo[msg.photo.length - 1];
    const fileInfo = await apiCall('getFile', { file_id: photo.file_id });
    if (!fileInfo.ok) return null;

    const fileUrl = `https://api.telegram.org/file/bot${config.telegramToken}/${fileInfo.result.file_path}`;
    const resp = await fetch(fileUrl);
    if (!resp.ok) return null;

    const buffer = Buffer.from(await resp.arrayBuffer());
    mkdirSync(SHORT_TERM_DIR, { recursive: true });
    const filename = `tg_${randomBytes(4).toString('hex')}.jpg`;
    const filepath = join(SHORT_TERM_DIR, filename);
    writeFileSync(filepath, buffer);
    return filepath;
  } catch (err) {
    console.log('[telegram:image_download_error]', err.message);
    return null;
  }
}

function formatLogLine(type, data) {
  switch (type) {
    case 'assistant_text': {
      const text = String(data.text ?? '').trim();
      if (!text) return null; // skip empty
      return `[${data.model || '?'}] ${text.slice(0, 120).replace(/\n/g, ' ')}`;
    }
    case 'pipeline_phase':
      return `→ Phase ${data.phase}: ${data.description}`;
    case 'tool_use':
      return `  ▶ ${data.tool || data.name}[${data.model || '?'}] ${JSON.stringify(data.input || {}).slice(0, 100)}`;
    case 'tool_result': {
      const out = String(data.output ?? '').trim().replace(/\n/g, ' ');
      return `  ◀ ${data.tool || '?'}[${data.model || '?'}] ${out.slice(0, 120)}${out.length > 120 ? '…' : ''}`;
    }
    case 'cost':
      return `  cost[${data.model||'?'}]: $${(data.cost ?? 0).toFixed(4)} | ${data.duration}ms | in:${data.input_tokens ?? '?'} out:${data.output_tokens ?? '?'}${data.cache_read ? ` cache:${data.cache_read}` : ''}`;
    case 'tool_install':
      return `  [install] ${data.message}`;
    case 'warning':
      return `  ⚠ ${data.message}`;
    case 'stderr':
      return `  [stderr:${data.model||'?'}] ${String(data.text||'').slice(0,200)}`;
    case 'response':
      return `→ response (${data.responseLength} chars) to ${data.sender}`;
    case 'sent':
      return `✓ sent (${data.responseLength} chars) to ${data.to}`;
    default: {
      const compact = { ...data };
      delete compact.text;
      delete compact.fullEvents;
      return JSON.stringify(compact);
    }
  }
}

function emitLog(type, data) {
  const entry = { type, data, timestamp: new Date().toISOString() };
  io?.emit('log', entry);
  const line = formatLogLine(type, data);
  if (line !== null) console.log(`[telegram:${type}]`, line);
}

async function apiCall(method, body = {}) {
  const res = await fetch(`${TELEGRAM_API}${config.telegramToken}/${method}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function sendMessage(chatId, text) {
  // Telegram has a 4096 char limit per message
  const chunks = [];
  for (let i = 0; i < text.length; i += 4000) {
    chunks.push(text.slice(i, i + 4000));
  }
  for (const chunk of chunks) {
    await apiCall('sendMessage', { chat_id: chatId, text: chunk });
  }
}

async function sendPhoto(chatId, filePath, caption) {
  try {
    const fileData = readFileSync(filePath);
    const filename = basename(filePath);
    const formData = new FormData();
    formData.append('chat_id', String(chatId));
    formData.append('photo', new Blob([fileData]), filename);
    if (caption) formData.append('caption', caption.slice(0, 1024));

    const res = await fetch(`${TELEGRAM_API}${config.telegramToken}/sendPhoto`, {
      method: 'POST',
      body: formData,
    });
    const result = await res.json();
    if (!result.ok) {
      console.log('[telegram:sendPhoto_error]', result.description);
    }
    return result;
  } catch (err) {
    console.log('[telegram:sendPhoto_error]', err.message);
    return { ok: false };
  }
}

async function sendResponseWithImages(chatId, response) {
  const { images, cleanText } = extractImages(response);
  // Send each image first
  for (const imagePath of images) {
    await sendPhoto(chatId, imagePath);
  }
  // Then send the text (if any remains after stripping markers)
  if (cleanText) {
    await sendMessage(chatId, cleanText);
  }
}

function formatQuestionsMessage(questionsPayload) {
  const questions = Array.isArray(questionsPayload?.questions) ? questionsPayload.questions : [];
  if (!questions.length) {
    return 'I need one clarification before I continue. Please reply with the missing details.';
  }

  const lines = ['I need a few details before I continue:'];
  for (let i = 0; i < questions.length; i++) {
    const q = questions[i];
    lines.push('');
    lines.push(`${i + 1}. ${q.question || 'Please clarify:'}`);
    const opts = Array.isArray(q.options) ? q.options : [];
    for (let j = 0; j < opts.length; j++) {
      const opt = opts[j];
      lines.push(`   - ${opt.label || `Option ${j + 1}`}: ${opt.description || ''}`.trimEnd());
    }
  }
  lines.push('');
  lines.push('Reply with your answer(s), and I will continue.');
  return lines.join('\n');
}

async function handleStopCommand(chatId, number) {
  const processKey = number !== null ? `conv:${number}` : `chat:${chatId}`;
  const killed = killProcess(processKey);
  clearClarificationState(processKey);
  if (killed) {
    const label = number !== null ? `conversation #${number}` : 'current conversation';
    await sendMessage(chatId, `Stopped ${label}.`);
  } else {
    const label = number !== null ? `conversation #${number}` : 'this chat';
    await sendMessage(chatId, `Nothing running for ${label}.`);
  }
}

function handleUpdate(update) {
  // Intercept stop commands before queueing — they must bypass the queue
  const msg = update.message;
  if (msg?.text || msg?.caption) {
    let prompt = msg.text || msg.caption || '';
    if (config.telegramPrefix && prompt.startsWith(config.telegramPrefix)) {
      prompt = prompt.slice(config.telegramPrefix.length).trim();
    } else if (config.telegramPrefix) {
      // prefix configured but doesn't match — fall through to normal handling
      prompt = '';
    }
    if (prompt) {
      const parsed = parseMessage(prompt);
      if (parsed.command === 'stop') {
        handleStopCommand(msg.chat.id, parsed.number);
        return;
      }
    }
  }

  const key = getQueueKey(update);
  if (!key) {
    // Still need to handle non-text updates (e.g. /start, /id) that getQueueKey skips
    const chatId = update.message?.chat?.id;
    if (!chatId) return;
    const prev = chatQueues.get(`chat:${chatId}`) || Promise.resolve();
    const next = prev.then(() => processUpdate(update)).catch(() => {});
    chatQueues.set(`chat:${chatId}`, next);
    return;
  }

  // Chain onto the existing queue for this key
  const prev = chatQueues.get(key) || Promise.resolve();
  const next = prev.then(() => processUpdate(update)).catch(() => {});
  chatQueues.set(key, next);
}

async function processUpdate(update) {
  const msg = update.message;
  const hasPhoto = !!(msg?.photo?.length);
  if (!msg?.text && !msg?.caption && !hasPhoto) return;

  const chatId = msg.chat.id;
  const sender = msg.from.first_name || msg.from.username || String(chatId);
  const text = msg.text || msg.caption || '';

  // Check allowed users
  if (config.telegramAllowedIds.length > 0 && !config.telegramAllowedIds.includes(chatId)) {
    emitLog('blocked', { sender, chatId, reason: 'not in allowed list' });
    return;
  }

  // Handle /start command
  if (text === '/start') {
    await sendMessage(chatId, `Pepper Claude Bridge active.\n\nYour chat ID: ${chatId}\nSend any message and Claude Code will respond.\n\nPrefix: "${config.telegramPrefix || ''}"\nSend /id to get your chat ID for the allowed list.`);
    return;
  }

  if (text === '/id') {
    await sendMessage(chatId, `Your chat ID: ${chatId}`);
    return;
  }

  // Extract prompt
  let prompt = text;
  if (config.telegramPrefix && text.startsWith(config.telegramPrefix)) {
    prompt = text.slice(config.telegramPrefix.length).trim();
  } else if (config.telegramPrefix && !hasPhoto) {
    return; // Has prefix configured but message doesn't match (allow photos through)
  }

  if (!prompt && !hasPhoto) return;

  // Check for /new command to force a fresh session
  if (prompt && prompt.toLowerCase() === '/new') {
    chatSessions.delete(chatId);
    saveChatSessions();
    clearClarificationState(`chat:${chatId}`);
    await sendMessage(chatId, 'Session cleared. Next message starts a fresh conversation.');
    return;
  }

  // Parse for numbered conversation prefix
  const parsed = parseMessage(prompt || 'What is this image?');

  // Handle close command
  if (parsed.command === 'close') {
    const closed = closeConversation(parsed.number);
    clearClarificationState(`conv:${parsed.number}`);
    await sendMessage(chatId, closed
      ? `Conversation #${parsed.number} closed.`
      : `No active conversation #${parsed.number}.`);
    return;
  }

  // Determine session: numbered conversation takes priority, then implicit timeout-based
  let resumeSessionId = null;
  if (parsed.number !== null) {
    resumeSessionId = resolveSession(parsed.number);
    emitLog('incoming', { sender, prompt: parsed.body, chatId, conversation: parsed.number, resuming: resumeSessionId });
  } else {
    const existing = chatSessions.get(chatId);
    if (existing && (Date.now() - existing.lastActivity) < SESSION_TIMEOUT_MS) {
      resumeSessionId = existing.sessionId;
      emitLog('incoming', { sender, prompt: parsed.body, chatId, resuming: existing.sessionId });
    } else {
      emitLog('incoming', { sender, prompt: parsed.body, chatId });
    }
  }

  // Download image if present
  let finalPrompt = parsed.body;
  if (hasPhoto) {
    const imagePath = await downloadTelegramPhoto(msg);
    if (imagePath) {
      const caption = finalPrompt || 'What is this image?';
      finalPrompt = `[The user sent an image. Read it with your Read tool at: ${imagePath}]\n\n${caption}`;
      emitLog('image', { sender, path: imagePath });
    }
  }

  // Send "typing" indicator
  await apiCall('sendChatAction', { chat_id: chatId, action: 'typing' });

  // Collect conversation log for this request
  const convoLog = { sender, prompt: finalPrompt, chatId, conversationNumber: parsed.number, resumeSessionId, timestamp: new Date().toISOString(), events: [] };

  const processKey = parsed.number !== null ? `conv:${parsed.number}` : `chat:${chatId}`;

  const isKnownCode = parsed.number !== null && getConversationMode(parsed.number) === 'code';

  activeCount++;
  try {
    const progressWrapper = createRuntimeAwareProgress((type, data) => {
      emitLog(type, { sender, ...data });
      convoLog.events.push({ type, ...data });
    });
    const onProgress = progressWrapper.onProgress;
    convoLog.runtimeFingerprint = progressWrapper.health.bootFingerprint;
    convoLog.runtimeStaleDetected = progressWrapper.health.stale;
    convoLog.runtimeChangedFiles = progressWrapper.health.changedFiles;
    if (progressWrapper.health.stale) {
      emitLog('runtime_stale_code_detected', { sender, chatId, changedFiles: progressWrapper.health.changedFiles });
    }

    let execResult;
    let didDelegate = false;
    if (isKnownCode) {
      execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, resumeSessionId, processKey, clarificationKey: processKey }));
    } else {
      execResult = await executeClaudePrompt(finalPrompt, { onProgress, resumeSessionId, processKey, clarificationKey: processKey, detectDelegation: true });
      if (execResult.delegation) {
        didDelegate = true;
        emitLog('delegation', { sender, employee: 'coder', model: execResult.delegation.model });
        execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, processKey, clarificationKey: processKey }, execResult.delegation.model));
      }
    }
    if (execResult.status === 'needs_user_input') {
      convoLog.clarificationState = {
        status: 'needs_user_input',
        questions: execResult.questions,
      };
      const message = formatQuestionsMessage(execResult.questions);
      await sendMessage(chatId, message);
      emitLog('clarification_requested', { sender, chatId, conversation: parsed.number });
      return;
    }
    const { response, sessionId, fullEvents } = execResult;

    // Store session for future messages
    const mode = (isKnownCode || didDelegate) ? 'code' : 'assistant';
    if (sessionId) {
      if (parsed.number !== null) {
        createOrUpdateConversation(parsed.number, sessionId, parsed.body, 'telegram', mode);
      }
      chatSessions.set(chatId, { sessionId, lastActivity: Date.now() });
      saveChatSessions();
    }

    convoLog.response = response;
    convoLog.sessionId = sessionId;
    convoLog.fullEvents = fullEvents;
    emitLog('response', { sender, prompt: parsed.body, responseLength: response.length, sessionId, conversation: parsed.number });
    await sendResponseWithImages(chatId, response);
    emitLog('sent', { to: sender, responseLength: response.length });
  } catch (err) {
    // If deliberately stopped, don't retry or send error
    if (err.stopped) {
      // Stop handler already sent a message — just let finally clean up
    } else if (resumeSessionId) {
      // If resume failed, try once more without resuming
      emitLog('resume_failed', { sender, error: err.message, fallback: 'fresh session' });
      chatSessions.delete(chatId);
      saveChatSessions();
      try {
        const progressWrapper = createRuntimeAwareProgress((type, data) => {
          emitLog(type, { sender, ...data });
          convoLog.events.push({ type, ...data });
        });
        const onProgress = progressWrapper.onProgress;
        convoLog.runtimeFingerprint = progressWrapper.health.bootFingerprint;
        convoLog.runtimeStaleDetected = progressWrapper.health.stale;
        convoLog.runtimeChangedFiles = progressWrapper.health.changedFiles;
        if (progressWrapper.health.stale) {
          emitLog('runtime_stale_code_detected', { sender, chatId, changedFiles: progressWrapper.health.changedFiles });
        }

        let execResult;
        let didDelegate = false;
        if (isKnownCode) {
          execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, processKey, clarificationKey: processKey }));
        } else {
          execResult = await executeClaudePrompt(finalPrompt, { onProgress, processKey, clarificationKey: processKey, detectDelegation: true });
          if (execResult.delegation) {
            didDelegate = true;
            emitLog('delegation', { sender, employee: 'coder', model: execResult.delegation.model });
            execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, processKey, clarificationKey: processKey }, execResult.delegation.model));
          }
        }
        if (execResult.status === 'needs_user_input') {
          convoLog.clarificationState = {
            status: 'needs_user_input',
            questions: execResult.questions,
          };
          const message = formatQuestionsMessage(execResult.questions);
          await sendMessage(chatId, message);
          emitLog('clarification_requested', { sender, chatId, conversation: parsed.number });
          return;
        }
        const { response, sessionId, fullEvents } = execResult;

        const mode = (isKnownCode || didDelegate) ? 'code' : 'assistant';
        if (sessionId) {
          if (parsed.number !== null) {
            createOrUpdateConversation(parsed.number, sessionId, finalPrompt, 'telegram', mode);
          }
          chatSessions.set(chatId, { sessionId, lastActivity: Date.now() });
          saveChatSessions();
        }
        convoLog.response = response;
        convoLog.sessionId = sessionId;
        convoLog.fullEvents = fullEvents;
        emitLog('response', { sender, prompt: parsed.body, responseLength: response.length, sessionId, conversation: parsed.number });
        await sendResponseWithImages(chatId, response);
        emitLog('sent', { to: sender, responseLength: response.length });
      } catch (retryErr) {
        convoLog.error = retryErr.message;
        emitLog('error', { sender, prompt: parsed.body, error: retryErr.message });
        await sendMessage(chatId, `Error: ${retryErr.message}`);
      }
    } else {
      convoLog.error = err.message;
      emitLog('error', { sender, prompt: parsed.body, error: err.message });
      await sendMessage(chatId, `Error: ${err.message}`);
    }
  } finally {
    activeCount--;
  }

  // Persist conversation log to disk
  try {
    mkdirSync(LOGS_DIR, { recursive: true });
    const filename = `${nextLogNumber()}_${sender}.json`;
    writeFileSync(join(LOGS_DIR, filename), JSON.stringify(convoLog, null, 2));
    addToLogIndex(filename, convoLog);
    io?.emit('conversation_update', { sessionId: convoLog.sessionId, conversationNumber: parsed.number });
  } catch (e) {
    console.log('[telegram:log_write_error]', e.message);
  }

  // Clean up temp files in memory/short-term/ only when no other conversations are in-flight
  if (activeCount === 0) {
    try {
      const files = readdirSync(SHORT_TERM_DIR);
      for (const f of files) {
        unlinkSync(join(SHORT_TERM_DIR, f));
      }
    } catch (e) {
      // Non-critical — don't let cleanup failures break the bot
    }
  }
}

async function pollUpdates() {
  if (!polling) return;
  try {
    const data = await apiCall('getUpdates', {
      offset,
      timeout: 30,
      allowed_updates: ['message'],
    });

    if (data.ok && data.result?.length) {
      for (const update of data.result) {
        offset = update.update_id + 1;
        // Queue per chat — messages from the same chat process sequentially
        handleUpdate(update);
      }
    }
  } catch (err) {
    console.log('[telegram:poll_error]', err.message);
    await new Promise((r) => setTimeout(r, 5000));
  }

  if (polling) {
    pollUpdates();
  }
}

async function startTelegram() {
  if (!config.telegramToken) {
    console.log('\n  Telegram bot not configured. Set telegramToken in config.json');
    console.log('  1. Message @BotFather on Telegram');
    console.log('  2. Send /newbot and follow the steps');
    console.log('  3. Copy the token to config.json as "telegramToken"\n');
    return false;
  }

  // Verify token
  const me = await apiCall('getMe');
  if (!me.ok) {
    console.log('[telegram] Invalid token:', me.description);
    return false;
  }

  console.log(`\n  Telegram bot: @${me.result.username}`);
  console.log(`  Message it on Telegram to use Claude Code\n`);

  emitLog('connected', { bot: me.result.username, message: `Telegram bot @${me.result.username} is online` });

  polling = true;
  pollUpdates();
  return true;
}

function stopTelegram() {
  polling = false;
}

export { startTelegram, stopTelegram, setSocketIO as setTelegramSocketIO };
