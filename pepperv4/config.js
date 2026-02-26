// Config loader — reads from pepperv1/backend/config.json (shared runtime config).
// Copied from pepperv1/backend/src/config.js so pepperv4 has no source imports from v1.

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, '..', 'pepperv1', 'backend', 'config.json');

const defaults = {
  port: 3457,
  allowedNumbers: [18032920205],
  allowAllNumbers: false,
  claudeCommand: 'claude',
  claudeArgs: ['--print'],
  maxResponseLength: 4000,
  messageTimeout: 900000,
  rateLimitPerMinute: 10,
  workingDirectory: process.cwd(),
  codeClaudeArgs: ['--print'],
  codeWorkingDirectory: process.cwd(),
  prefix: '!claude ',
  authDir: join(__dirname, '..', 'pepperv1', 'backend', 'auth_state'),
  telegramToken: '8588798112:AAGq_cTWMfd3HyQErQVUeMnVFitmcHWbhgM',
  telegramPrefix: '',
  telegramAllowedIds: [],
  smsGatewayUrl: '',
  smsGatewayUsername: '',
  smsGatewayPassword: '',
  smsAllowedNumbers: [],
  smsPrefix: '',
  outputDirectory: join(__dirname, '..', 'bot', 'outputs'),
};

function loadConfig() {
  if (existsSync(CONFIG_PATH)) {
    try {
      const file = JSON.parse(readFileSync(CONFIG_PATH, 'utf-8'));
      return { ...defaults, ...file };
    } catch {
      return { ...defaults };
    }
  }
  return { ...defaults };
}

function saveConfig(config) {
  const toSave = { ...config };
  delete toSave.authDir;
  writeFileSync(CONFIG_PATH, JSON.stringify(toSave, null, 2));
}

const config = loadConfig();

export { config, saveConfig, loadConfig };
