import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  makeCacheableSignalKeyStore,
  fetchLatestWaWebVersion,
  Browsers,
} from '@whiskeysockets/baileys';
import { Boom } from '@hapi/boom';
import pino from 'pino';
import QRCode from 'qrcode';
import { writeFileSync, mkdirSync, readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config } from './config.js';
import { handleMessage } from './message-handler.js';
import { addToLogIndex, nextLogNumber } from './index.js';
import { extractImages } from './transport-utils.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const LOGS_DIR = join(__dirname, '..', 'bot', 'logs');

const logger = pino({ level: 'silent' });

let sock = null;
let io = null;
let connectionStatus = 'disconnected';
let lastQR = null;
let reconnectAttempt = 0;
const MAX_RECONNECT_ATTEMPTS = 10;

// Track message IDs sent by the bot to prevent infinite loops
const botSentIds = new Set();

function setSocketIO(socketIO) {
  io = socketIO;
}

function emitLog(type, data) {
  const entry = { type, data, timestamp: new Date().toISOString() };
  io?.emit('log', entry);
  if (type !== 'qr') {
    console.log(`[${type}]`, JSON.stringify(data));
  }
}

function getStatus() {
  return connectionStatus;
}

async function startWhatsApp() {
  const { state, saveCreds } = await useMultiFileAuthState(config.authDir);

  let version;
  try {
    const result = await fetchLatestWaWebVersion({});
    version = result.version;
    console.log(`Using WhatsApp Web version: ${version.join('.')}`);
  } catch {
    version = [2, 3000, 1033498124];
    console.log(`Using fallback WhatsApp Web version: ${version.join('.')}`);
  }

  sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    version,
    browser: Browsers.windows('Chrome'),
    logger,
    generateHighQualityLinkPreview: false,
    markOnlineOnConnect: true,
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      connectionStatus = 'waiting_for_qr';
      lastQR = qr;

      QRCode.toString(qr, { type: 'terminal', small: true }, (err, str) => {
        if (!err) console.log('\n' + str);
      });

      QRCode.toDataURL(qr, { width: 280, margin: 2 }, (err, dataUrl) => {
        if (!err) {
          io?.emit('qr', dataUrl);
        }
      });

      io?.emit('status', connectionStatus);
      emitLog('qr', { message: 'QR code generated - scan with WhatsApp' });
    }

    if (connection === 'close') {
      connectionStatus = 'disconnected';
      io?.emit('status', connectionStatus);

      const statusCode = new Boom(lastDisconnect?.error)?.output?.statusCode;
      const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

      emitLog('disconnected', { statusCode, willReconnect: shouldReconnect });

      if (shouldReconnect && reconnectAttempt < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempt++;
        const delay = Math.min(3000 * Math.pow(1.5, reconnectAttempt - 1), 30000);
        console.log(`Reconnecting in ${Math.round(delay / 1000)}s (attempt ${reconnectAttempt}/${MAX_RECONNECT_ATTEMPTS})...`);
        setTimeout(startWhatsApp, delay);
      } else if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
        console.log('Max reconnection attempts reached. Restart the server to try again.');
        emitLog('max_retries', { message: 'Max reconnection attempts reached' });
      } else {
        console.log('Logged out. Delete auth_state folder and restart to re-authenticate.');
        emitLog('logged_out', { message: 'Scan QR code again to reconnect' });
      }
    }

    if (connection === 'open') {
      connectionStatus = 'connected';
      reconnectAttempt = 0;
      io?.emit('status', connectionStatus);
      emitLog('connected', { message: 'WhatsApp connected successfully' });
      console.log('WhatsApp connected!');
    }
  });

  sock.ev.on('messages.upsert', (upsert) => {
    const messages = upsert.messages || [];
    for (const msg of messages) {
      const msgId = msg.key.id;
      if (botSentIds.has(msgId)) {
        botSentIds.delete(msgId);
        continue;
      }

      // Fire off each message concurrently — each spawns its own Claude instance
      (async () => {
        const result = await handleMessage(msg, emitLog);
        if (result && result.response) {
          try {
            const { images, cleanText } = extractImages(result.response);
            // Send each image first
            for (const imagePath of images) {
              try {
                const imageData = readFileSync(imagePath);
                const imgSent = await sock.sendMessage(result.jid, { image: imageData });
                if (imgSent?.key?.id) botSentIds.add(imgSent.key.id);
              } catch (imgErr) {
                emitLog('send_image_error', { to: result.sender, path: imagePath, error: imgErr.message });
              }
            }
            // Then send the text (if any remains)
            if (cleanText) {
              const sent = await sock.sendMessage(result.jid, { text: cleanText });
              if (sent?.key?.id) botSentIds.add(sent.key.id);
            }
            emitLog('sent', { to: result.sender, responseLength: result.response.length, imageCount: images.length });
          } catch (err) {
            emitLog('send_error', { to: result.sender, error: err.message });
          }

          // Persist conversation log
          try {
            mkdirSync(LOGS_DIR, { recursive: true });
            const filename = `${nextLogNumber()}_${result.sender}.json`;
            const convoLog = {
              sender: result.sender,
              prompt: result.prompt,
              jid: result.jid,
              conversationNumber: result.conversationNumber ?? null,
              sessionId: result.sessionId || null,
              timestamp: new Date().toISOString(),
              fullEvents: result.fullEvents || [],
              response: result.response,
              runtimeFingerprint: result.runtimeFingerprint || null,
              runtimeStaleDetected: !!result.runtimeStaleDetected,
              runtimeChangedFiles: result.runtimeChangedFiles || [],
            };
            writeFileSync(join(LOGS_DIR, filename), JSON.stringify(convoLog, null, 2));
            addToLogIndex(filename, convoLog);
            io?.emit('conversation_update', { sessionId: result.sessionId, conversationNumber: result.conversationNumber });
          } catch (e) {
            console.log('[whatsapp:log_write_error]', e.message);
          }
        }
      })();
    }
  });

  return sock;
}

function getLastQR() {
  return lastQR;
}

export { startWhatsApp, setSocketIO, getStatus, getLastQR };
