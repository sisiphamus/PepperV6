// Model B: Knowledge & Skill Auditor.
// buildGapPrompt: slim Haiku-optimised prompt for gap detection after executor failure.
// NOTE: Full buildPrompt was removed — Phase B selection is now handled by ML (infer.py).

/**
 * Slim gap-detection prompt for Haiku.
 * Called only when a previous execution failed.
 * Returns JSON with missingMemories (and optionally toolsNeeded, notes).
 * selectedMemories is intentionally omitted — ML already handled retrieval.
 */
export function buildGapPrompt(taskDescription, memoryInventory, prompt, previousFailure) {
  const inventoryList = memoryInventory
    .map(m => `- [${m.category}] ${m.name}: ${m.description}`)
    .join('\n');

  return `You are a knowledge gap detector. An AI executor just failed at a task. Your job is to identify what knowledge or skills are MISSING from the memory library that caused the failure.

## Task That Failed
${taskDescription}

## What The Executor Said (failure output)
---
${(previousFailure || '').slice(0, 1500)}
---

## User's Original Request
${prompt}

## Existing Memory Library (do NOT request these — they already exist)
${inventoryList || '(none)'}

## Instructions
Identify memory files that SHOULD EXIST but DON'T, which would have allowed the executor to succeed.
Focus on: missing API knowledge, missing tool setup guides, missing workflow patterns, missing site interaction patterns.
Do NOT list memories that already exist above.

Your response MUST be ONLY a raw JSON object. Output nothing except the JSON:
{
  "selectedMemories": [],
  "missingMemories": [
    { "name": "proposed_name", "category": "skill|knowledge|preference|site", "description": "exactly what this memory should contain so the executor can succeed", "reason": "why it's needed based on the failure" }
  ],
  "toolsNeeded": ["any MCP tools or CLIs the executor needs that may not be installed"],
  "notes": "brief explanation of what went wrong and what will fix it"
}`;
}
