// Config loader — reads from pepperv1/backend/config.json (shared runtime config).
// Copied from pepperv1/backend/src/config.js so pepperv4 has no source imports from v1.

import { readFileSync, writeFileSync, renameSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { randomBytes } from 'crypto';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, '..', 'pepperv1', 'backend', 'config.json');

const defaults = {
  port: 3457,
  allowedNumbers: [],
  allowAllNumbers: false,
  claudeCommand: 'claude',
  claudeArgs: ['--print', '--max-turns', '25'],
  maxResponseLength: 4000,
  messageTimeout: 900000,
  rateLimitPerMinute: 10,
  workingDirectory: process.cwd(),
  codeClaudeArgs: ['--print'],
  codeWorkingDirectory: process.cwd(),
  prefix: '!claude ',
  authDir: join(__dirname, '..', 'pepperv1', 'backend', 'auth_state'),
  telegramToken: '',
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
  const tmpPath = CONFIG_PATH + `.tmp.${randomBytes(4).toString('hex')}`;
  writeFileSync(tmpPath, JSON.stringify(toSave, null, 2));
  renameSync(tmpPath, CONFIG_PATH);
}

const config = loadConfig();

export { config, saveConfig, loadConfig };
