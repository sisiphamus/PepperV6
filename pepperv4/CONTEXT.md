# pepperv4 — Multi-Model Orchestration Pipeline

## What is this?

pepperv4 replaces pepperv1's `claude-bridge.js` — the layer that takes a user message and turns it into a Claude response. Instead of a single Claude call, pepperv4 runs a 4-model pipeline that analyzes the problem, gathers the right knowledge, fills gaps, and executes with full context.

pepperv1's server, messaging channels (WhatsApp, Telegram, SMS, web dashboard), and conversation management are unchanged. pepperv4 only replaces the "brain" — how messages get processed.

Built fresh from pepperv1's foundation. No code from pepperv2 or pepperv3. pepperv4 has zero source imports from pepperv1 — it has its own copy of config.js and only shares the runtime config.json and memory directory as data.

## How it works

```
User Message
    |
    |— simple? → Fast-path: single Claude call (same as pepperv1)
    |
    |— complex? → Full pipeline:
    |
    v
Model A (Sonnet) — Delegator
    Analyzes the request, defines the exact output format/structure
    |
    v
Model B (Sonnet) — Knowledge & Skill Auditor
    Reviews all memories (skills, knowledge, preferences, sites)
    Selects what's relevant, identifies gaps
    |
    |— gaps? → Model C (Sonnet) — Teacher
    |              Creates new memory files, then Model B re-audits
    v
Model D (config default) — Executor
    Takes output spec + relevant memories, does the actual work
    |
    |— needs more? → loops back to Model B (max 3 times)
    |
    v
Post-task Learner (Sonnet)
    Reviews what happened, saves useful knowledge for next time
```

## File structure

```
pepperv4/
  index.js                        — Bridge adapter (same API as old claude-bridge.js)
  config.js                       — Config loader (reads pepperv1/backend/config.json)
  claude-bridge.js                — Drop-in replacement for pepperv1's claude-bridge.js
  runtime-health.js               — Drop-in replacement for pepperv1's runtime-health.js
  pipeline/
    orchestrator.js               — A → B → C? → D → learn, with feedback loops
    model-runner.js               — Claude CLI subprocess spawner
    prompts/
      model-a.js                  — Delegator prompt
      model-b.js                  — Skill Auditor prompt
      model-c.js                  — Teacher prompt
      model-d.js                  — Executor prompt
      learner.js                  — Post-task learner prompt
  memory/
    memory-manager.js             — Reads/writes all memory types
    clarification-manager.js      — Manages needs_user_input / resumed flows
  util/
    output-parser.js              — JSON extraction from model outputs
    progress-aggregator.js        — Merges progress events from sub-models
    process-registry.js           — Process tracking + kill support
    fast-path.js                  — Simple message detection
```

## Memory system

pepperv4 manages 4 categories of memory, all stored in `pepperv1/backend/bot/memory/`:

| Category | Location | What it is |
|----------|----------|------------|
| Skills | `skills/{name}/SKILL.md` | Domain expertise (coding, marketing, data analysis) |
| Knowledge | `knowledge/{topic}.md` | Reference facts, frameworks, research |
| Preferences | `preferences/{topic}.md` | User-specific context (who you are, your accounts) |
| Sites | `sites/{site}.md` | Website interaction patterns (how to use Gmail, Todoist, etc.) |

Model B audits all 4 categories for every complex request and passes relevant content to Model D. Model C creates new memories when gaps are found. The post-task learner updates memories after execution.

## How to enable pepperv4

pepperv4 does NOT modify pepperv1 files. Instead, it provides drop-in replacements:

1. Copy `pepperv4/claude-bridge.js` → `pepperv1/backend/src/claude-bridge.js`
2. Copy `pepperv4/runtime-health.js` → `pepperv1/backend/src/runtime-health.js`
3. Set env var: `PEPPER_V4_ENABLED=true`
4. Start normally: `npm run dev`

To revert, restore the original files or set `PEPPER_V4_ENABLED=false`.

## Fast-path vs pipeline

- **Fast-path**: Messages under 200 chars matching simple patterns (questions, greetings) skip the pipeline and get a single Claude call. Same speed as pepperv1.
- **Full pipeline**: Everything else goes through A→B→C?→D. More thorough but uses multiple Claude invocations.

## Models used

| Role | Model | Why |
|------|-------|-----|
| A, B, C, post-task learner | Sonnet | Fast, cheap — these do structured analysis, not heavy work |
| D (Executor) | Config default | The one that does real work — uses whatever model is configured |
