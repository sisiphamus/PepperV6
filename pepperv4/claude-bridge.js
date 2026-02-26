// Drop-in replacement for pepperv1/backend/src/claude-bridge.js
// Adds pepperv4 support via PEPPER_V4_ENABLED env var.
// Copy this to pepperv1/backend/src/claude-bridge.js when ready to switch.
//
// Defaults:
//   - PEPPER_V4_ENABLED=true  → uses pepperv4
//   - PEPPER_V3_ENABLED=true  → uses pepperv3
//   - Neither                 → uses pepperv3 (safe default)

import * as pepperv2 from '../../../pepperv2/index.js';
import * as pepperv3 from '../../../pepperv3/index.js';
import * as pepperv4 from '../../../pepperv4/index.js';

function useV4() {
  const raw = String(process.env.PEPPER_V4_ENABLED || '').trim().toLowerCase();
  if (!raw) return false;
  return raw !== 'false' && raw !== '0' && raw !== 'no';
}

function useV3() {
  const raw = String(process.env.PEPPER_V3_ENABLED || '').trim().toLowerCase();
  if (!raw) return true;
  return raw !== 'false' && raw !== '0' && raw !== 'no';
}

function bridge() {
  if (useV4()) return pepperv4;
  return useV3() ? pepperv3 : pepperv2;
}

export function executeClaudePrompt(prompt, options) {
  return bridge().executeClaudePrompt(prompt, options);
}

export function killProcess(key) {
  return bridge().killProcess(key);
}

export function codeAgentOptions(baseOptions, modelOverride) {
  return bridge().codeAgentOptions(baseOptions, modelOverride);
}

export function employeeAgentOptions(employeeName, baseOptions, modelOverride) {
  return bridge().employeeAgentOptions(employeeName, baseOptions, modelOverride);
}

export function getEmployeeMode(employeeName) {
  return bridge().getEmployeeMode(employeeName);
}

export function setProcessChangeListener(fn) {
  return bridge().setProcessChangeListener(fn);
}

export function setProcessActivityListener(fn) {
  return bridge().setProcessActivityListener(fn);
}

export function getActiveProcessSummary() {
  return bridge().getActiveProcessSummary();
}

export function getClarificationState(key) {
  return bridge().getClarificationState(key);
}

export function clearClarificationState(key) {
  return bridge().clearClarificationState(key);
}
