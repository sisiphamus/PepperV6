import * as pepperv4 from '../../../pepperv4/index.js';

export function executeClaudePrompt(prompt, options) {
  return pepperv4.executeClaudePrompt(prompt, options);
}

export function killProcess(key) {
  return pepperv4.killProcess(key);
}

export function codeAgentOptions(baseOptions, modelOverride) {
  return pepperv4.codeAgentOptions(baseOptions, modelOverride);
}

export function employeeAgentOptions(employeeName, baseOptions, modelOverride) {
  return pepperv4.employeeAgentOptions(employeeName, baseOptions, modelOverride);
}

export function getEmployeeMode(employeeName) {
  return pepperv4.getEmployeeMode(employeeName);
}

export function setProcessChangeListener(fn) {
  return pepperv4.setProcessChangeListener(fn);
}

export function setProcessActivityListener(fn) {
  return pepperv4.setProcessActivityListener(fn);
}

export function getActiveProcessSummary() {
  return pepperv4.getActiveProcessSummary();
}

export function getClarificationState(key) {
  return pepperv4.getClarificationState(key);
}

export function clearClarificationState(key) {
  return pepperv4.clearClarificationState(key);
}
