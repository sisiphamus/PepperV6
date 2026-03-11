// Pipeline orchestrator — coordinates A → B → C? → D → learn with feedback loops.

import { runModel } from './model-runner.js';
import { ensureBrowserReady } from '../../pepperv1/backend/src/browser-health.js';
import { runPhaseA, runPhaseB } from './ml-runner.js';
import { buildGapPrompt as modelBGapPrompt } from './prompts/model-b.js';
import { buildPrompt as modelCPrompt } from './prompts/model-c.js';
import { buildPrompt as modelDPrompt } from './prompts/model-d.js';
import { buildPrompt as learnerPrompt } from './prompts/learner.js';
import { parseOutputSpec, parseAuditResult, parseTeacherResult, parseLearnerResult } from '../util/output-parser.js';
import { createAggregator } from '../util/progress-aggregator.js';
import { getFullInventory, getContents, writeMemory, updateMemory, detectSiteContext } from '../memory/memory-manager.js';
import { config } from '../config.js';
import { setClaudeSessionId } from '../session/session-manager.js';
import { execSync } from 'child_process';
import { mkdirSync } from 'fs';

const MAX_FEEDBACK_LOOPS = 3;

const FAILURE_PATTERNS = [
  /i (?:can'?t|cannot|am unable to|don'?t have (?:the ability|access)|am not able to) (?:do|perform|complete|accomplish|execute|help with)/i,
  /(?:unfortunately|sorry),? (?:i |this )?(?:can'?t|cannot|isn'?t possible|is not possible|won'?t work)/i,
  /i don'?t (?:know how|have (?:enough|the (?:tools|knowledge|capability)))/i,
  /(?:beyond|outside) (?:my|the) (?:capabilities|scope|ability)/i,
  /(?:not (?:currently )?(?:able|possible|supported)) (?:to|for|in|with)/i,
  /i'?m (?:afraid|sorry) (?:i |that )?(?:can'?t|cannot)/i,
];

// Phrases that indicate a non-failure even if they contain "can't"-like words
const FALSE_POSITIVE_PATTERNS = [
  /(?:can'?t|cannot|couldn'?t) find (?:any|the|unread|new|recent)/i,
  /no (?:new |unread )?(?:emails|messages|tasks|assignments|notifications)/i,
  /inbox is (?:empty|clean|clear)/i,
  /nothing (?:new|due|pending|found)/i,
];

const SUCCESS_PATTERNS = [
  /^(?:all )?done\.?$/i,
  /^(?:all )?complete[d.]?\.?$/i,
  /^(?:task )?(?:finished|succeeded)\.?$/i,
  /^(?:sent|delivered|created|updated|deleted|saved|installed)\.?$/i,
  /^(?:email|message) sent\.?$/i,
];

function detectFailure(response) {
  if (!response) return true;
  const trimmed = response.trim();
  if (!trimmed) return true;
  // Short responses that match known success patterns are NOT failures
  if (trimmed.length < 20 && SUCCESS_PATTERNS.some(p => p.test(trimmed))) return false;
  // Truly empty/meaningless responses are failures
  if (trimmed.length < 3) return true;
  // Check for false positives first — these look like failures but aren't
  if (FALSE_POSITIVE_PATTERNS.some(p => p.test(response))) return false;
  // Long responses (>500 chars) with detailed content are likely successful completions
  // that happen to contain hedging language — only flag short failures
  if (trimmed.length > 500) return false;
  return FAILURE_PATTERNS.some(p => p.test(response));
}

export async function runPipeline(prompt, { onProgress, processKey, timeout, resumeSessionId, sessionContext, genomeOverride, skipLearning }) {
  const outputDir = config.outputDirectory;
  mkdirSync(outputDir, { recursive: true });
  const agg = createAggregator(onProgress);

  // ── Fast path: resumed session → skip A/B/C, send raw message ──
  // When resuming a conversation, the Claude session already has full context
  // from the previous turn(s). Re-running the pipeline would wrap the user's
  // follow-up in a fresh "Model D Executor" system prompt, destroying continuity.
  if (resumeSessionId) {
    agg.phase('D', 'Continuing conversation (resumed session)');

    const phaseD = await runModel({
      userPrompt: prompt,
      systemPrompt: undefined,
      model: null,
      claudeArgs: config.claudeArgs,
      onProgress: (type, data) => agg.forward('D', type, data),
      processKey: processKey ? `${processKey}:D` : null,
      timeout,
      cwd: outputDir,
      resumeSessionId,
    });

    // If the resumed session returned empty (stale/invalid session ID),
    // fall through to the full pipeline instead of returning nothing.
    if (!phaseD.response || !phaseD.response.trim()) {
      agg.phase('D', 'Resumed session returned empty — starting fresh pipeline');
      resumeSessionId = null;
      // Fall through to full pipeline below
    } else {
      // Track Claude's session ID back to our internal session
      if (sessionContext && phaseD.sessionId) {
        setClaudeSessionId(sessionContext.id, phaseD.sessionId);
      }

      if (phaseD.questionRequest) {
        return {
          status: 'needs_user_input',
          questions: phaseD.questionRequest,
          sessionId: phaseD.sessionId,
          fullEvents: phaseD.fullEvents,
        };
      }

      // Fire-and-forget learning
      learnInBackground(prompt, { taskDescription: prompt }, phaseD.response, phaseD.fullEvents, onProgress, processKey, timeout);

      return {
        status: 'completed',
        response: phaseD.response,
        sessionId: phaseD.sessionId,
        fullEvents: phaseD.fullEvents,
      };
    }
  }

  // ── Phase A: Dual-axis classifier (local ML) ──
  agg.phase('A', 'Classifying request (local ML)');
  const phaseAResponse = await runPhaseA(prompt);
  const outputSpec = parseOutputSpec(phaseAResponse);
  const intent = outputSpec.intent || 'query';
  const activeLabels = outputSpec.outputLabels
    ? Object.entries(outputSpec.outputLabels).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none'
    : outputSpec.outputType || 'text';
  const scoreStr = outputSpec.outputScores
    ? ' | scores: ' + Object.entries(outputSpec.outputScores).map(([k, v]) => `${k}=${v}`).join(' ')
    : '';
  agg.phase('A', `Complete → intent=${intent} formats=[${activeLabels}]${scoreStr}`);

  // ── Feedback loop: A → B → C? → D, max 3 iterations ──
  let loopCount = 0;
  let lastDResponse = null;
  let lastDSessionId = null;
  let lastDFullEvents = null;
  let previousFailure = null;
  const seenToolRequests = new Set(); // Track NEEDS_MORE_TOOLS to prevent duplicate requests

  while (loopCount < MAX_FEEDBACK_LOOPS) {
    loopCount++;

    // ── Phase B: Memory retrieval (local ML) + gap detection (Haiku on failure) ──
    const taskDesc = outputSpec.taskDescription || prompt;
    agg.phase('B', `Selecting relevant memory files (ML, pass ${loopCount})`);

    const inventory = getFullInventory();
    const phaseBResponse = await runPhaseB(
      previousFailure ? `${prompt}\n\nPrevious failure context: ${previousFailure.slice(0, 500)}` : prompt,
      inventory,
      intent
    );
    const audit = parseAuditResult(phaseBResponse);
    const selectedSummary = (audit.selectedMemories || [])
      .map(m => `${m.name} (${m.reason || m.category})`)
      .join(', ') || 'none';
    onProgress?.('pipeline_phase', { phase: 'B', description: `Selected: ${selectedSummary}` });

    // Gap detection: only invoke Haiku when a previous execution failed
    if (previousFailure != null) {
      agg.phase('B', 'Detecting knowledge gaps (Haiku)');
      const gapModel = await runModel({
        userPrompt: `Output ONLY a raw JSON object. No prose. No explanation. Identify missing memories for the failed task.\n\nFailed task: ${taskDesc}\n\nFailure output: ${previousFailure.slice(0, 800)}`,
        systemPrompt: modelBGapPrompt(taskDesc, inventory, prompt, previousFailure),
        model: 'haiku',
        claudeArgs: ['--print', '--max-turns', '1'],
        onProgress: (type, data) => agg.forward('B', type, data),
        processKey: processKey ? `${processKey}:Bgap` : null,
        timeout,
      });
      const gapAudit = parseAuditResult(gapModel.response);
      audit.missingMemories = gapAudit.missingMemories || [];
      audit.toolsNeeded = gapAudit.toolsNeeded || [];
      if (gapAudit.notes) audit.notes = (audit.notes ? audit.notes + ' | ' : '') + gapAudit.notes;
    }

    // If B didn't return valid JSON and we have a previous failure, force-create
    // a missing memory so C (Teacher) actually runs and researches the topic
    if (previousFailure != null && (!audit.missingMemories || audit.missingMemories.length === 0)) {
      onProgress?.('warning', { message: `Model B didn't identify gaps — forcing knowledge acquisition for: ${outputSpec.taskDescription}` });
      audit.missingMemories = [{
        name: outputSpec.taskDescription.replace(/[^a-z0-9]+/gi, '-').toLowerCase().slice(0, 50),
        category: 'knowledge',
        description: `How to: ${outputSpec.taskDescription}. The executor previously failed with: ${(previousFailure || '').slice(0, 300)}`,
        reason: 'Executor failed and auditor did not identify gaps — forcing research',
      }];
    }

    // ── Phase C: Teacher (if gaps found) ──
    let newlyCreatedMemories = [];
    if (audit.missingMemories && audit.missingMemories.length > 0) {
      agg.phase('C', `Creating ${audit.missingMemories.length} new memory file(s)`);

      const phaseC = await runModel({
        userPrompt: `Create the following memories:\n${audit.missingMemories.map(m => `- ${m.name}: ${m.description}`).join('\n')}`,
        systemPrompt: modelCPrompt(audit.missingMemories, inventory),
        model: 'sonnet',
        claudeArgs: ['--print', '--allowedTools', 'WebSearch,WebFetch,Bash'],
        onProgress: (type, data) => agg.forward('C', type, data),
        processKey: processKey ? `${processKey}:C` : null,
        timeout,
      });

      const teacherResult = parseTeacherResult(phaseC.response);
      for (const mem of teacherResult.memories) {
        try {
          await writeMemory(mem.name, mem.category, mem.content);
          newlyCreatedMemories.push(mem); // Keep for immediate use by D
          await tryInstallFromMemory(mem, onProgress);
        } catch (err) {
          // Non-fatal — log and continue
          onProgress?.('warning', { message: `Failed to write memory ${mem.name}: ${err.message}` });
        }
      }
    }

    // ── Phase D: Executor ──
    agg.phase('D', 'Executing task');

    // Gather memory contents for selected memories + newly created ones from C
    const selectedContents = getContents(audit.selectedMemories || []);
    // Add C's memories directly (they have name, category, content already)
    const newContents = newlyCreatedMemories.map(m => ({ name: m.name, category: m.category, content: m.content }));

    // Add site context detected from the prompt (deduplicate against Phase B selections)
    const selectedNames = new Set((audit.selectedMemories || []).map(m => m.name));
    const siteContext = detectSiteContext(prompt).filter(s => !selectedNames.has(s.name));
    const allMemoryContents = [...selectedContents, ...newContents, ...siteContext];

    // Always ensure the correct Chrome (AutomationProfile + CDP) is running before Phase D.
    // Fast no-op if Chrome is already up. Prevents model D from ever seeing "CDP not available"
    // and attempting to launch Chrome with wrong flags (missing --user-data-dir=AutomationProfile).
    await ensureBrowserReady();
    if (genomeOverride) {
      allMemoryContents.unshift({ name: 'agent-genome', category: 'evolution', content: genomeOverride });
    }

    const phaseD = await runModel({
      userPrompt: prompt,
      systemPrompt: modelDPrompt(prompt, outputSpec, allMemoryContents),
      model: null, // use config default
      claudeArgs: config.claudeArgs,
      onProgress: (type, data) => agg.forward('D', type, data),
      processKey: processKey ? `${processKey}:D` : null,
      timeout,
      cwd: outputDir,
      resumeSessionId,
    });

    // Track Claude's session ID back to our internal session
    if (sessionContext && phaseD.sessionId) {
      setClaudeSessionId(sessionContext.id, phaseD.sessionId);
    }

    if (phaseD.questionRequest) {
      return {
        status: 'needs_user_input',
        questions: phaseD.questionRequest,
        sessionId: phaseD.sessionId,
        fullEvents: phaseD.fullEvents,
      };
    }

    lastDResponse = phaseD.response;
    lastDSessionId = phaseD.sessionId;
    lastDFullEvents = phaseD.fullEvents;

    // Check if Model D needs more tools/knowledge or failed entirely
    const needsMore = lastDResponse?.match(/\[NEEDS_MORE_TOOLS:\s*(.+?)\]/);
    if (needsMore && loopCount < MAX_FEEDBACK_LOOPS) {
      const toolsNeeded = needsMore[1].trim();
      // Prevent the same tool request from looping — if we already tried this, give up
      if (seenToolRequests.has(toolsNeeded.toLowerCase())) {
        agg.phase('feedback', `Already attempted to resolve: ${toolsNeeded}. Stopping retry loop.`);
        break;
      }
      seenToolRequests.add(toolsNeeded.toLowerCase());
      agg.phase('feedback', `Model D needs: ${toolsNeeded}. Bypassing B and injecting targeted memory request.`);
      // Bypass B entirely — inject a precise missingMemories entry so C researches exactly what's needed
      audit.missingMemories = [buildToolMemoryRequest(toolsNeeded)];
      previousFailure = lastDResponse;
      // Skip directly to C (loopCount was already incremented at top of while loop — don't double-increment)
      if (loopCount < MAX_FEEDBACK_LOOPS) {
        agg.phase('C', `Creating 1 new memory file(s) for: ${toolsNeeded}`);
        const phaseC2 = await runModel({
          userPrompt: `Create the following memories:\n- ${audit.missingMemories[0].name}: ${audit.missingMemories[0].description}`,
          systemPrompt: modelCPrompt(audit.missingMemories, getFullInventory()),
          model: 'sonnet',
          claudeArgs: ['--print', '--allowedTools', 'WebSearch,WebFetch,Bash'],
          onProgress: (type, data) => agg.forward('C', type, data),
          processKey: processKey ? `${processKey}:C2` : null,
          timeout,
        });
        const teacherResult2 = parseTeacherResult(phaseC2.response);
        for (const mem of teacherResult2.memories) {
          try {
            writeMemory(mem.name, mem.category, mem.content);
            newlyCreatedMemories.push(mem);
            // If this memory describes how to install a tool, run the install now
            await tryInstallFromMemory(mem, onProgress);
          } catch (err) {
            onProgress?.('warning', { message: `Failed to write memory ${mem.name}: ${err.message}` });
          }
        }
        // Re-run D with the new memory
        const updatedContents = [...getContents(audit.selectedMemories || []), ...newlyCreatedMemories.map(m => ({ name: m.name, category: m.category, content: m.content })), ...detectSiteContext(prompt)];
        await ensureBrowserReady();
        const phaseD2 = await runModel({
          userPrompt: prompt,
          systemPrompt: modelDPrompt(prompt, outputSpec, updatedContents),
          model: null,
          claudeArgs: config.claudeArgs,
          onProgress: (type, data) => agg.forward('D', type, data),
          processKey: processKey ? `${processKey}:D2` : null,
          timeout,
          cwd: outputDir,
          resumeSessionId,
        });
        if (phaseD2.questionRequest) {
          return { status: 'needs_user_input', questions: phaseD2.questionRequest, sessionId: phaseD2.sessionId, fullEvents: phaseD2.fullEvents };
        }
        lastDResponse = phaseD2.response;
        lastDSessionId = phaseD2.sessionId;
        lastDFullEvents = phaseD2.fullEvents;
      }
      break;
    }

    if (detectFailure(lastDResponse) && loopCount < MAX_FEEDBACK_LOOPS) {
      agg.phase('feedback', `Model D couldn't complete the task. Looping back to B for more knowledge.`);
      previousFailure = lastDResponse || '(executor returned empty response)';
      continue;
    }

    break;
  }

  // ── Post-task learning (fire-and-forget) ──
  if (!skipLearning) learnInBackground(prompt, outputSpec, lastDResponse, lastDFullEvents, onProgress, processKey, timeout);

  return {
    status: 'completed',
    response: lastDResponse,
    sessionId: lastDSessionId,
    fullEvents: lastDFullEvents,
  };
}

// Maps a [NEEDS_MORE_TOOLS] description to a targeted missingMemories entry for Model C.
function buildToolMemoryRequest(toolsNeeded) {
  const lower = toolsNeeded.toLowerCase();
  if (lower.includes('playwright')) {
    return {
      name: 'playwright-mcp-setup',
      category: 'knowledge',
      description: 'How to install and use the Playwright MCP server (claude mcp add playwright) on Windows so that Claude Code subprocesses have access to browser_navigate, browser_snapshot, browser_click and other browser automation tools. Include: exact install command, how to verify it is active, and how to use it in a claude --print subprocess.',
      reason: toolsNeeded,
    };
  }
  // Generic fallback — let C figure it out from the description
  const slug = toolsNeeded.replace(/[^a-z0-9]+/gi, '-').toLowerCase().slice(0, 50);
  return {
    name: `tool-setup-${slug}`,
    category: 'knowledge',
    description: `How to install and use: ${toolsNeeded}. Include exact install/setup commands for Windows and how to verify the tool is available.`,
    reason: toolsNeeded,
  };
}

// Allowlist of safe install command patterns — blocks arbitrary code execution from memory files
const SAFE_INSTALL_PATTERNS = [
  /^npm\s+(install|i)\s/,
  /^npx\s/,
  /^pip3?\s+install\s/,
  /^winget\s+install\s/,
  /^choco\s+install\s/,
  /^claude\s+mcp\s+add\s/,
];

function isSafeInstallCommand(cmd) {
  return SAFE_INSTALL_PATTERNS.some(p => p.test(cmd.trim()));
}

// If a freshly-created memory describes tool installs, run them immediately so D can use them.
// Supports any install_command: lines — npm packages, pip packages, winget, etc.
async function tryInstallFromMemory(mem, onProgress) {
  const content = mem.content || '';

  // Collect ALL install_command: lines in the file (there may be multiple steps)
  const lines = content.split('\n');
  const installLines = lines
    .map(l => l.match(/^\s*install_command:\s*(.+)/i))
    .filter(Boolean)
    .map(m => {
      let cmd = m[1].trim();
      // Strip trailing markdown artifacts (**, __, *, etc.)
      cmd = cmd.replace(/[*_`]+$/, '').trim();
      // Replace bare 'claude' with full path from config (node's PATH may not include it)
      cmd = cmd.replace(/^claude\b/, config.claudeCommand);
      return cmd;
    })
    .filter(cmd => cmd.length > 0);

  for (const cmd of installLines) {
    if (!isSafeInstallCommand(cmd)) {
      onProgress?.('warning', { message: `Blocked unsafe install command: ${cmd}` });
      continue;
    }
    onProgress?.('tool_install', { message: `Installing: ${cmd}` });
    try {
      execSync(cmd, { stdio: 'pipe', timeout: 60000, shell: true });
      onProgress?.('tool_install', { message: `Installed: ${cmd}` });
    } catch (err) {
      onProgress?.('warning', { message: `Install failed (${cmd}): ${err.message?.slice(0, 200)}` });
    }
  }
}

function learnInBackground(prompt, outputSpec, executionResponse, fullEvents, onProgress, processKey, timeout) {
  const agg = createAggregator(onProgress);

  // Don't await — fire and forget
  (async () => {
    try {
      agg.phase('learn', 'Reviewing execution for learnings');

      const inventory = getFullInventory();

      // Build a compact execution trace from fullEvents: tool calls + results + assistant text
      let executionTrace = '';
      if (Array.isArray(fullEvents)) {
        const traceLines = [];
        for (const ev of fullEvents) {
          if (ev.type === 'tool_use') {
            traceLines.push(`TOOL_USE: ${ev.tool} ${JSON.stringify(ev.input || {}).slice(0, 200)}`);
          } else if (ev.type === 'tool_result') {
            const out = typeof ev.output === 'string' ? ev.output : JSON.stringify(ev.output || '');
            traceLines.push(`TOOL_RESULT: ${out.slice(0, 300)}`);
          } else if (ev.type === 'assistant_text' && ev.text) {
            traceLines.push(`ASSISTANT: ${ev.text.slice(0, 300)}`);
          } else if (ev.type === 'stderr' && ev.text) {
            traceLines.push(`STDERR: ${ev.text.slice(0, 200)}`);
          }
        }
        executionTrace = traceLines.join('\n');
      }
      // Cap trace at 8000 chars, prefer tail (most recent events are most informative)
      if (executionTrace.length > 8000) executionTrace = '...(truncated)\n' + executionTrace.slice(-8000);

      const result = await runModel({
        userPrompt: `Review this execution and save any useful knowledge.\n\nPrompt: ${prompt}\n\nFinal response: ${(executionResponse || '').slice(0, 1000)}\n\nExecution trace:\n${executionTrace}`,
        systemPrompt: learnerPrompt(prompt, outputSpec, (executionResponse || '').slice(0, 2000), inventory),
        model: 'sonnet',
        onProgress: (type, data) => agg.forward('learner', type, data),
        processKey: processKey ? `${processKey}:learner` : null,
        timeout: timeout || 900000,
      });

      const learnerResult = parseLearnerResult(result.response);
      if (!learnerResult.updates || learnerResult.updates.length === 0) {
        process.stderr.write(`[learner] No updates extracted from learner response (length=${(result.response || '').length})\n`);
      }
      for (const update of learnerResult.updates) {
        try {
          if (update.path && update.action === 'append') {
            // Block path traversal attempts
            if (update.path.includes('..') || update.path.startsWith('/') || /^[a-zA-Z]:/.test(update.path)) {
              process.stderr.write(`[learner] Blocked path traversal: ${update.path}\n`);
              continue;
            }
            await updateMemory(update.path, 'append', update.content);
            process.stderr.write(`[learner] Appended to ${update.path}\n`);
          } else {
            await writeMemory(update.name, update.category, update.content);
            process.stderr.write(`[learner] Wrote memory: ${update.name} (${update.category})\n`);
          }
        } catch (err) {
          process.stderr.write(`[learner] Failed to write memory ${update.name || update.path}: ${err.message}\n`);
        }
      }
    } catch (err) {
      // Non-fatal — learning is best-effort, but log the error for diagnostics
      process.stderr.write(`[learner] Learning failed: ${err.message}\n`);
    }
  })();
}
