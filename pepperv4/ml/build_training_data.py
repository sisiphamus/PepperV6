"""
Extract real prompts from session logs and auto-label them for the new dual-axis taxonomy.
Outputs pepperv4/ml/data/real_examples.json for training.

Intent labels:  query, action, create, converse, instruct
Format labels:  inline, file, image, slides, browser
"""

import json, re, glob, os
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent.parent / 'pepperv1' / 'backend' / 'bot' / 'logs'
OUTPUT_PATH = Path(__file__).parent / 'data' / 'real_examples.json'

GREETING_WORDS = {'hi', 'hey', 'hello', 'yo', 'yes', 'no', 'ok', 'cool', 'nice', 'thanks',
                  'good morning', 'good night', 'sup', 'whats up', 'ping', 'bye'}

ACTION_VERBS = {'send', 'open', 'push', 'delete', 'navigate', 'add', 'go', 'install',
                'email', 'run', 'close', 'launch', 'click', 'move', 'set', 'connect',
                'fix', 'check', 'continue', 'start', 'stop', 'restart', 'kill',
                'download', 'upload', 'deploy', 'remove', 'update', 'save'}

CREATE_VERBS = {'create', 'make', 'build', 'write', 'generate', 'draw', 'design',
                'produce', 'compose', 'draft', 'prepare'}

BROWSER_KEYWORDS = {'gmail', 'canvas', 'website', 'navigate', 'browser', 'chrome', 'tab',
                    'google', 'linkedin', 'notion', 'todoist', 'gradescope', 'gcal',
                    'calendar', 'drive', 'docs', 'sheets', 'email', 'inbox', 'mail'}


def auto_label(prompt):
    """Heuristic labeler for the dual-axis taxonomy."""
    p = prompt.lower().strip()
    words = p.split()

    # ── Intent ──
    # Check converse first (short casual messages)
    if len(words) <= 4:
        first_word = words[0] if words else ''
        if first_word in GREETING_WORDS or p in GREETING_WORDS:
            intent = 'converse'
        elif any(w in GREETING_WORDS for w in words):
            intent = 'converse'
        else:
            intent = 'query'  # short but not greeting
    elif re.search(r'\b(from now on|remember that|remember this|never do|always do|teach yourself|whenever you)\b', p):
        intent = 'instruct'
    elif re.match(r'^(what|who|when|where|how|why|is|are|can|does|do|did|check|whats|what\'s)\b', p):
        intent = 'query'
    elif any(w in p.split() for w in CREATE_VERBS):
        intent = 'create'
    elif any(w in p.split() for w in ACTION_VERBS):
        intent = 'action'
    else:
        intent = 'query'

    # Override: if it starts with action verb, it's action
    if words and words[0] in ACTION_VERBS:
        intent = 'action'

    # Override: "continue your work" is action
    if 'continue' in p:
        intent = 'action'

    # ── Format ──
    formats = []
    if re.search(r'\b(presentation|slides?|deck|pptx|keynote)\b', p):
        formats.append('slides')
    if re.search(r'\b(image|picture|draw|diagram|chart|infographic|screenshot|photo)\b', p):
        formats.append('image')
    if re.search(r'\b(file|save as|export|\.py|\.pdf|\.csv|\.json|\.md|script|document|\.txt|\.xlsx)\b', p):
        formats.append('file')
    if any(kw in p for kw in BROWSER_KEYWORDS):
        formats.append('browser')
    if not formats:
        formats.append('inline')

    return intent, formats


def extract_from_logs():
    examples = []
    seen_prompts = set()

    log_files = sorted(glob.glob(str(LOGS_DIR / '*.json')),
                       key=lambda f: int(os.path.basename(f).split('_')[0])
                       if os.path.basename(f).split('_')[0].isdigit() else 999)

    for log_file in log_files:
        try:
            with open(log_file, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        prompt = data.get('prompt', '')
        if not prompt or not prompt.strip():
            continue

        # Deduplicate (many "Learn how to send gmail" repeats)
        key = prompt.strip()[:100]
        if key in seen_prompts:
            continue
        seen_prompts.add(key)

        intent, formats = auto_label(prompt)
        examples.append({
            'prompt': prompt.strip(),
            'intent': intent,
            'formats': formats,
            'source': os.path.basename(log_file),
        })

    return examples


def main():
    examples = extract_from_logs()

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f'Extracted {len(examples)} unique prompts -> {OUTPUT_PATH}')
    print()

    # Summary table
    from collections import Counter
    intents = Counter(e['intent'] for e in examples)
    print('Intent distribution:')
    for k, v in intents.most_common():
        print(f'  {k}: {v}')

    format_counts = Counter()
    for e in examples:
        for fmt in e['formats']:
            format_counts[fmt] += 1
    print('\nFormat distribution:')
    for k, v in format_counts.most_common():
        print(f'  {k}: {v}')

    print('\n--- Examples ---')
    for e in examples[:15]:
        p = e['prompt'][:70].replace('\n', ' ')
        print(f'  [{e["intent"]:>10}] [{",".join(e["formats"]):>15}] {p}')
    if len(examples) > 15:
        print(f'  ... and {len(examples) - 15} more')


if __name__ == '__main__':
    main()
