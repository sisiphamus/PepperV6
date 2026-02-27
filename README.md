# PepperV5

The fifth generation of Pepper, my AI personal assistant that lives on my phone and runs my digital life.

PepperV5 takes the multi-model orchestration pipeline from earlier versions and makes it faster and cheaper by replacing two Claude API calls with local machine learning models. Messages come in through WhatsApp, Telegram, SMS, or the web dashboard. A trained TF-IDF classifier instantly categorizes the task. A second local model retrieves relevant memories. Only then does Claude step in to actually execute.

## How It Works

```
You (WhatsApp/Telegram/SMS/Web)
        |
  Phase A - Local ML classifier (scikit-learn, not Claude)
        |
  Phase B - Local TF-IDF memory retrieval
        |
  Phase C - Teacher (Claude Sonnet, only if knowledge gaps exist)
        |
  Phase D - Executor (Claude, does the actual work)
        |
  Learner - Saves what it learned for next time
```

The local ML layer means Phase A and B respond in milliseconds instead of seconds, and cost nothing per request.

## The ML Stack

- **Task Classifier**: TF-IDF vectorizer + Logistic Regression, trained on labeled message history. Classifies into: text response, image generation, shell command, presentation, file operation, or other
- **Memory Retriever**: TF-IDF keyword matching against the knowledge base, pulling relevant context before any LLM call
- **Inference Server**: Persistent Python subprocess communicating via newline-delimited JSON over stdin/stdout. Stays warm between requests for sub-100ms classification

## Architecture

- `pepperv4/` - The orchestration pipeline with ML integration
- `pepperv4/ml/` - Training scripts, inference server, saved models
- `pepperv4/pipeline/` - Orchestrator, model runner, ML runner
- `pepperv1/` - The battle-tested backend handling all messaging platforms
- `bot/` - Claude's working directory and output files

## Tech

Node.js, Python (scikit-learn), Claude CLI, Express, Socket.IO, Baileys (WhatsApp), Telegram Bot API

## Context

This is part of a larger evolution: Pepper (v0) -> pepperv1 -> Pepper2 -> Pepper-3 -> PepperV5 -> Overthink. Each version adds a new layer of intelligence. V5's contribution is proving that local ML can handle the routine classification work, saving the expensive models for where they actually matter.
