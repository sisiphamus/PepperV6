import { executeClaudePrompt, killProcess, codeAgentOptions, clearClarificationState } from './claude-bridge.js';
import { config } from './config.js';
import { parseMessage, resolveSession, createOrUpdateConversation, closeConversation, getConversationMode } from './conversation-manager.js';
import { downloadContentFromMessage } from '@whiskeysockets/baileys';
import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { randomBytes } from 'crypto';
import { createRuntimeAwareProgress } from './runtime-health.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SHORT_TERM_DIR = join(__dirname, '..', 'bot', 'memory', 'short-term');

const rateLimitMap = new Map();

function isRateLimited(jid) {
  const now = Date.now();
  const timestamps = rateLimitMap.get(jid) || [];
  const recent = timestamps.filter((t) => now - t < 60000);
  rateLimitMap.set(jid, recent);
  return recent.length >= config.rateLimitPerMinute;
}

function recordMessage(jid) {
  const timestamps = rateLimitMap.get(jid) || [];
  timestamps.push(Date.now());
  rateLimitMap.set(jid, timestamps);
}

function isAllowed(jid) {
  if (config.allowAllNumbers) return true;
  const number = jid.replace(/@.*/, '');
  return config.allowedNumbers.some((n) => number.includes(n.replace(/\D/g, '')));
}

function extractPrompt(text) {
  if (!text) return null;
  if (config.prefix && text.startsWith(config.prefix)) {
    return text.slice(config.prefix.length).trim();
  }
  if (!config.prefix) return text.trim();
  return null;
}

function formatQuestionsForText(questionsPayload) {
  const questions = Array.isArray(questionsPayload?.questions) ? questionsPayload.questions : [];
  if (!questions.length) {
    return 'I need a bit more detail before I continue. Please reply with the missing details.';
  }
  const lines = ['I need a few details before I continue:'];
  for (let i = 0; i < questions.length; i++) {
    lines.push(`${i + 1}. ${questions[i].question || 'Please clarify'}`);
  }
  lines.push('Reply with your answer(s), and I will continue.');
  return lines.join('\n');
}

/**
 * Downloads an image from a WhatsApp message and saves it to short-term memory.
 * Returns the file path or null.
 */
async function downloadWhatsAppImage(message) {
  const imageMsg = message.message?.imageMessage || message.message?.extendedTextMessage?.contextInfo?.quotedMessage?.imageMessage;
  if (!imageMsg) return null;

  try {
    mkdirSync(SHORT_TERM_DIR, { recursive: true });
    const stream = await downloadContentFromMessage(imageMsg, 'image');
    const chunks = [];
    for await (const chunk of stream) chunks.push(chunk);
    const buffer = Buffer.concat(chunks);

    const ext = (imageMsg.mimetype || 'image/jpeg').includes('png') ? 'png' : 'jpg';
    const filename = `wa_${randomBytes(4).toString('hex')}.${ext}`;
    const filepath = join(SHORT_TERM_DIR, filename);
    writeFileSync(filepath, buffer);
    return filepath;
  } catch (err) {
    console.log('[whatsapp:image_download_error]', err.message);
    return null;
  }
}

/**
 * Processes an incoming WhatsApp message.
 * Returns { response, sender, prompt } or null if the message should be ignored.
 */
export async function handleMessage(message, emitLog) {
  const jid = message.key.remoteJid;
  if (!jid || jid === 'status@broadcast') return null;

  // Extract text from various message types
  const text =
    message.message?.conversation ||
    message.message?.extendedTextMessage?.text ||
    message.message?.imageMessage?.caption ||
    null;

  // Check if there's an image attached
  const hasImage = !!(message.message?.imageMessage);

  // Need either text or an image
  if (!text && !hasImage) return null;

  const prompt = extractPrompt(text || (hasImage ? 'What is this image?' : null));
  if (!prompt) return null;

  const sender = message.pushName || jid.replace(/@.*/, '');
  emitLog?.('incoming', { sender, prompt, jid });

  if (!isAllowed(jid)) {
    emitLog?.('blocked', { sender, jid, reason: 'not in allowed list' });
    return null;
  }

  if (isRateLimited(jid)) {
    emitLog?.('rate-limited', { sender, jid });
    return { response: 'Rate limited. Please wait a moment.', sender, prompt, jid };
  }

  recordMessage(jid);

  // Parse for numbered conversation prefix
  const parsed = parseMessage(prompt);

  // Handle close command
  if (parsed.command === 'close') {
    const closed = closeConversation(parsed.number);
    clearClarificationState(`wa:conv:${parsed.number}`);
    const response = closed
      ? `Conversation #${parsed.number} closed.`
      : `No active conversation #${parsed.number}.`;
    return { response, sender, prompt, jid };
  }

  // Handle stop command
  if (parsed.command === 'stop') {
    const processKey = parsed.number !== null ? `wa:conv:${parsed.number}` : `wa:chat:${jid}`;
    const killed = killProcess(processKey);
    clearClarificationState(processKey);
    if (killed) {
      const label = parsed.number !== null ? `conversation #${parsed.number}` : 'current conversation';
      return { response: `Stopped ${label}.`, sender, prompt, jid };
    } else {
      const label = parsed.number !== null ? `conversation #${parsed.number}` : 'this chat';
      return { response: `Nothing running for ${label}.`, sender, prompt, jid };
    }
  }

  const resumeSessionId = parsed.number !== null ? resolveSession(parsed.number) : null;
  emitLog?.('processing', { sender, prompt: parsed.body, conversation: parsed.number });

  // Download image if present and prepend path to prompt
  let finalPrompt = parsed.body;
  if (hasImage) {
    const imagePath = await downloadWhatsAppImage(message);
    if (imagePath) {
      finalPrompt = `[The user sent an image. Read it with your Read tool at: ${imagePath}]\n\n${finalPrompt}`;
      emitLog?.('image', { sender, path: imagePath });
    }
  }

  const processKey = parsed.number !== null ? `wa:conv:${parsed.number}` : `wa:chat:${jid}`;
  const isKnownCode = parsed.number !== null && getConversationMode(parsed.number) === 'code';
  const progressWrapper = createRuntimeAwareProgress((type, data) => emitLog?.(type, { sender, ...data }));
  const onProgress = progressWrapper.onProgress;
  if (progressWrapper.health.stale) {
    emitLog?.('runtime_stale_code_detected', { sender, jid, changedFiles: progressWrapper.health.changedFiles });
  }

  try {
    let execResult;
    let didDelegate = false;
    if (isKnownCode) {
      execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, resumeSessionId, processKey, clarificationKey: processKey }));
    } else {
      execResult = await executeClaudePrompt(finalPrompt, { onProgress, resumeSessionId, processKey, clarificationKey: processKey, detectDelegation: true });
      if (execResult.delegation) {
        didDelegate = true;
        emitLog?.('delegation', { sender, employee: 'coder', model: execResult.delegation.model });
        execResult = await executeClaudePrompt(finalPrompt, codeAgentOptions({ onProgress, processKey, clarificationKey: processKey }, execResult.delegation.model));
      }
    }
    if (execResult.status === 'needs_user_input') {
      const response = formatQuestionsForText(execResult.questions);
      return { response, sender, prompt: parsed.body, jid, sessionId: execResult.sessionId, fullEvents: execResult.fullEvents, conversationNumber: parsed.number };
    }
    const response = execResult.response;

    const mode = (isKnownCode || didDelegate) ? 'code' : 'assistant';
    if (execResult.sessionId && parsed.number !== null) {
      createOrUpdateConversation(parsed.number, execResult.sessionId, parsed.body, 'whatsapp', mode);
    }

    emitLog?.('response', { sender, prompt: parsed.body, responseLength: response.length });
    return {
      response,
      sender,
      prompt: parsed.body,
      jid,
      sessionId: execResult.sessionId,
      fullEvents: execResult.fullEvents,
      conversationNumber: parsed.number,
      runtimeFingerprint: progressWrapper.health.bootFingerprint,
      runtimeStaleDetected: progressWrapper.health.stale,
      runtimeChangedFiles: progressWrapper.health.changedFiles,
    };
  } catch (err) {
    if (err.stopped) {
      return null; // Stop handler already returned a response
    }
    emitLog?.('error', { sender, prompt: parsed.body, error: err.message });
    return {
      response: `Error: ${err.message}`,
      sender,
      prompt: parsed.body,
      jid,
      runtimeFingerprint: progressWrapper.health.bootFingerprint,
      runtimeStaleDetected: progressWrapper.health.stale,
      runtimeChangedFiles: progressWrapper.health.changedFiles,
    };
  }
}

export { isAllowed, isRateLimited, extractPrompt, recordMessage };
