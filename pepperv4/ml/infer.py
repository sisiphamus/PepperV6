"""
Persistent inference subprocess for Phase A (dual-axis classifier) and Phase B (tiered retrieval).
Reads newline-delimited JSON from stdin, writes newline-delimited JSON to stdout.

Protocol:
  Phase A:  { "task": "phase_a", "prompt": "..." }
  Phase B:  { "task": "phase_b", "prompt": "...", "inventory": [...], "intent": "query" }
"""

import sys
import json
import re
import pickle
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix

# ── Constants ────────────────────────────────────────────────────────────────

INTENT_LABELS = ['query', 'action', 'create', 'converse', 'instruct']
FORMAT_LABELS = ['inline', 'file', 'image', 'slides', 'browser']
FORMAT_THRESHOLD = 0.38

MODEL_PATH = Path(__file__).parent / 'models' / 'phase_a_v2.pkl'

# Hand-crafted feature sets (must match train.py exactly)
QUESTION_STARTERS = {'what', 'whats', "what's", 'who', 'when', 'where', 'how',
                     'why', 'is', 'are', 'can', 'does', 'do', 'did',
                     'which', 'could', 'should', 'would', 'have', 'has'}
ACTION_VERBS = {'send', 'open', 'push', 'delete', 'navigate', 'add', 'go',
                'install', 'run', 'close', 'launch', 'click', 'move', 'fix',
                'continue', 'start', 'stop', 'deploy', 'connect', 'forward',
                'download', 'upload', 'update', 'set', 'mark', 'tick', 'deal',
                'learn', 'email', 'save', 'remove', 'log', 'submit', 'rotate',
                'complete', 'schedule', 'reply', 'pull', 'implement', 'set up'}
CREATE_VERBS = {'create', 'make', 'build', 'write', 'generate', 'draw',
                'design', 'produce', 'compose', 'draft', 'prepare', 'implement'}
SERVICE_KEYWORDS = {'gmail', 'calendar', 'gcal', 'canvas', 'notion', 'todoist',
                    'linkedin', 'github', 'google', 'drive', 'docs', 'sheets',
                    'slides', 'gradescope', 'chrome', 'email', 'mail', 'inbox',
                    'tasks', 'browser', 'io.rice', 'slack', 'discord'}
GREETING_SET = {'hi', 'hey', 'hello', 'yo', 'yes', 'no', 'ok', 'cool', 'nice',
                'thanks', 'sure', 'alright', 'great', 'awesome', 'k', 'lol',
                'haha', 'ping', 'sup', 'bye', 'noted', 'understood', 'true',
                'indeed', 'right', 'exactly', 'perfect', 'interesting', 'hmm'}
INSTRUCT_PHRASES = ['from now on', 'going forward', 'in the future', 'make a note',
                    'save this', 'remember this', 'remember that', 'note that',
                    'always use', 'never use', 'make sure to', 'teach yourself',
                    'keep in mind', 'memorize', 'update your memory',
                    'from now on teach', 'delete any']
INSTRUCT_STARTERS = {'from', 'never', 'always', 'remember', 'whenever',
                     'specifically', 'going', 'note', 'memorize', 'keep',
                     'in', 'teach', 'figure', 'update', 'log'}
CASUAL_PHRASES = ['tell me a joke', 'make me laugh', 'tell me something funny',
                  'chat with me', 'talk to me', 'entertain me', 'fun fact',
                  'tell me a fun fact', 'give me a fun fact', 'surprise me',
                  'how are you', "what's up", 'what do you think', 'any thoughts']

# Domain keyword map for metadata
DOMAIN_KEYWORDS = {
    'browser-automation': ['gmail', 'canvas', 'website', 'navigate', 'browser',
                           'chrome', 'tab', 'google', 'linkedin', 'notion'],
    'coding': ['script', 'code', 'python', 'function', 'debug', 'github',
               'push', 'commit', 'deploy', 'npm', 'pip'],
    'writing': ['write', 'essay', 'blog', 'article', 'report', 'summary', 'draft'],
    'research': ['research', 'find out', 'look up', 'study', 'analyze'],
    'email': ['email', 'gmail', 'send', 'inbox', 'mail'],
    'calendar': ['calendar', 'gcal', 'event', 'meeting', 'schedule', 'call'],
}

# Phase B: tier definitions
ALWAYS_INCLUDE = {'browser-preferences'}
SITE_TRIGGERS = {
    'gmail': ['gmail', 'email', 'mail', 'inbox', 'send email'],
    'canvas': ['canvas', 'assignment', 'coursework', 'class', 'course', 'homework', 'hw'],
    'gradescope': ['gradescope', 'grading', 'submission'],
    'google-docs': ['google doc', 'google docs', 'gdoc', 'gdocs'],
    'io-rice': ['io.rice', 'timecard', 'timesheet', 'work hours', 'io rice'],
    'slack': ['slack', 'slack message'],
    'mcmurtry': ['mcmurtry', 'ivp', 'committee'],
    'todoist': ['todoist', 'to-do', 'todo list', 'tasks', 'task list', 'google tasks'],
    'notion': ['notion', 'workspace', 'wiki'],
}
# Known contacts — when a name is mentioned, inject the contacts preference file
CONTACT_NAMES = {'adam', 'julia', 'bunge', 'sophiya', 'sami', 'towner', 'cynthia'}
PHASE_B_MIN_THRESHOLD = 0.12
PHASE_B_MAX_SKILL_RESULTS = 3

# ── Load Phase A model at startup ────────────────────────────────────────────

try:
    with open(MODEL_PATH, 'rb') as f:
        _model = pickle.load(f)
    _word_vec = _model['word_vectorizer']
    _char_vec = _model['char_vectorizer']
    _intent_clf = _model['intent_classifier']
    _format_clf = _model['format_classifier']
except Exception as e:
    sys.stderr.write(f'[infer.py] Failed to load Phase A model: {e}\n')
    sys.stderr.flush()
    _word_vec = None
    _char_vec = None
    _intent_clf = None
    _format_clf = None


# ── Hand-crafted features (must match train.py) ─────────────────────────────

def _extract_hand_features(prompt):
    words = prompt.lower().split()
    first_word = words[0] if words else ''
    prompt_lower = prompt.lower()
    starts_question = 1.0 if first_word in QUESTION_STARTERS else 0.0
    has_possessive = 1.0 if 'my' in words else 0.0
    return [
        starts_question,
        1.0 if any(w in ACTION_VERBS for w in words) else 0.0,
        1.0 if any(w in CREATE_VERBS for w in words) else 0.0,
        1.0 if any(w in SERVICE_KEYWORDS for w in words) else 0.0,
        1.0 if len(words) <= 3 and first_word in GREETING_SET else 0.0,
        has_possessive,
        1.0 if any(phrase in prompt_lower for phrase in INSTRUCT_PHRASES) else 0.0,
        1.0 if any(phrase in prompt_lower for phrase in CASUAL_PHRASES) else 0.0,
        1.0 if len(words) > 30 else 0.0,
        1.0 if first_word == 'go' else 0.0,
        1.0 if 'check' in words and (starts_question or has_possessive) else 0.0,
        1.0 if first_word in INSTRUCT_STARTERS and any(phrase in prompt_lower for phrase in INSTRUCT_PHRASES) else 0.0,
        1.0 if 'voice message' in prompt_lower or 'transcription' in prompt_lower else 0.0,
    ]


def _featurize(prompt):
    """Transform prompt into the combined feature matrix."""
    X_word = _word_vec.transform([prompt])
    X_char = _char_vec.transform([prompt])
    X_hand = csr_matrix(np.array([_extract_hand_features(prompt)]))
    return hstack([X_word, X_char, X_hand])


# ── Heuristic metadata ──────────────────────────────────────────────────────

def _compute_metadata(prompt):
    words = prompt.lower().split()
    clauses = len(re.split(r'[,;.!?]|\b(?:then|and also|also|after that)\b', prompt))
    action_count = sum(1 for w in words if w in ACTION_VERBS | CREATE_VERBS)

    if len(words) < 5 and action_count == 0:
        complexity = 'simple'
        steps = 1
    elif clauses >= 3 or action_count >= 2 or len(words) > 50:
        complexity = 'complex'
        steps = max(3, min(clauses + action_count, 10))
    else:
        complexity = 'moderate'
        steps = max(1, action_count + 1)

    domains = []
    prompt_lower = prompt.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(kw in prompt_lower for kw in kws):
            domains.append(domain)

    return complexity, steps, domains


def _map_to_legacy_type(intent, format_labels):
    """Map new taxonomy back to old 6-label system for backward compat."""
    if format_labels.get('slides'):
        return 'presentation'
    if format_labels.get('image'):
        return 'picture'
    if format_labels.get('file'):
        return 'specificFile'
    if intent in ('action', 'instruct'):
        return 'command'
    if intent == 'converse':
        return 'other'
    return 'text'


# ── Phase A: Dual-axis classifier ────────────────────────────────────────────

def run_phase_a(prompt):
    if _word_vec is None or _intent_clf is None:
        return _fallback_spec(prompt)

    X = _featurize(prompt)

    # Intent (single-label)
    intent_probas = _intent_clf.predict_proba(X)[0]
    intent_scores = {INTENT_LABELS[i]: float(p) for i, p in enumerate(intent_probas)}
    intent = max(intent_scores, key=intent_scores.get)

    # Format (multi-label)
    format_probas = _format_clf.predict_proba(X)
    format_scores = {}
    format_on = {}
    for i, label in enumerate(FORMAT_LABELS):
        score = float(format_probas[i][0][1])
        format_scores[label] = score
        format_on[label] = score >= FORMAT_THRESHOLD

    if not any(format_on.values()):
        best = max(format_scores, key=format_scores.get)
        format_on[best] = True

    # Metadata
    complexity, steps, domains = _compute_metadata(prompt)

    # Output format
    active_formats = [k for k, v in format_on.items() if v]
    if any(f in active_formats for f in ('file', 'image', 'slides')):
        delivery = 'file_link'
        fmt_type = 'file'
    else:
        delivery = 'inline'
        fmt_type = 'inline_text'

    return {
        'taskDescription': prompt[:500],
        'intent': intent,
        'intentScores': {k: round(v, 3) for k, v in intent_scores.items()},
        'outputType': _map_to_legacy_type(intent, format_on),
        'outputLabels': format_on,
        'outputScores': {k: round(v, 3) for k, v in format_scores.items()},
        'outputFormat': {
            'type': fmt_type,
            'structure': 'direct answer',
            'deliveryMethod': delivery,
        },
        'requiredDomains': domains,
        'complexity': complexity,
        'estimatedSteps': steps,
    }


def _fallback_spec(prompt):
    return {
        'taskDescription': prompt[:500],
        'intent': 'query',
        'intentScores': {l: 0.0 for l in INTENT_LABELS},
        'outputType': 'text',
        'outputLabels': {l: (l == 'inline') for l in FORMAT_LABELS},
        'outputScores': {l: 0.0 for l in FORMAT_LABELS},
        'outputFormat': {'type': 'inline_text', 'structure': 'direct answer', 'deliveryMethod': 'inline'},
        'requiredDomains': [],
        'complexity': 'simple',
        'estimatedSteps': 1,
    }


# ── Phase B: Tiered memory retrieval ────────────────────────────────────────

_phase_b_cache = {
    'inventory_hash': None,
    'vectorizer': None,
    'doc_vectors': None,
    'doc_meta': None,
}


def _compute_inventory_hash(inventory):
    content = json.dumps([(i.get('name', ''), i.get('path', '')) for i in inventory], sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def _build_search_doc(item, content):
    """Build a richer search document for TF-IDF indexing."""
    name = item.get('name', '')
    category = item.get('category', '')
    description = item.get('description', '')

    # Extract "When to use" section if present
    when_to_use = ''
    match = re.search(r'##\s*When\s+to\s+[Uu]se\s*\n([\s\S]*?)(?=\n##|\Z)', content)
    if match:
        when_to_use = match.group(1).strip()[:300]

    # Build composite: name weighted heavily
    parts = [
        f'{name} {name} {name}',
        category,
        description,
        when_to_use,
        content[:500],
    ]
    return ' '.join(parts)


def _ensure_index(inventory):
    """Build or refresh the TF-IDF index (cached)."""
    inv_hash = _compute_inventory_hash(inventory)
    if _phase_b_cache['inventory_hash'] == inv_hash:
        return

    from sklearn.feature_extraction.text import TfidfVectorizer

    doc_meta = []
    search_docs = []

    for item in inventory:
        path = item.get('path', '')
        try:
            with open(path, encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception:
            content = item.get('description', '')

        name = item.get('name', '')

        # Determine tier
        if name in ALWAYS_INCLUDE:
            tier = 'always'
        elif name in SITE_TRIGGERS:
            tier = 'site'
        else:
            tier = 'skill'

        doc_meta.append({
            'name': name,
            'category': item.get('category', 'knowledge'),
            'tier': tier,
        })
        search_docs.append(_build_search_doc(item, content))

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
    )
    doc_vectors = vec.fit_transform(search_docs)

    _phase_b_cache['inventory_hash'] = inv_hash
    _phase_b_cache['vectorizer'] = vec
    _phase_b_cache['doc_vectors'] = doc_vectors
    _phase_b_cache['doc_meta'] = doc_meta


def run_phase_b(prompt, inventory, intent='query'):
    if not inventory:
        return {
            'selectedMemories': [],
            'missingMemories': [],
            'toolsNeeded': [],
            'notes': 'No memory files in inventory',
        }

    _ensure_index(inventory)

    vec = _phase_b_cache['vectorizer']
    doc_vectors = _phase_b_cache['doc_vectors']
    doc_meta = _phase_b_cache['doc_meta']

    selected = []
    prompt_lower = prompt.lower()
    already_names = set()

    # ── Tier 1: Always-include (skip for converse) ──
    if intent != 'converse':
        for i, meta in enumerate(doc_meta):
            if meta['tier'] == 'always':
                selected.append({
                    'name': meta['name'],
                    'category': meta['category'],
                    'reason': 'always-include',
                })
                already_names.add(meta['name'])

    # ── Tier 2: Site-triggered (keyword match) ──
    if intent != 'converse':
        for site_name, triggers in SITE_TRIGGERS.items():
            if any(t in prompt_lower for t in triggers):
                for i, meta in enumerate(doc_meta):
                    if meta['name'] == site_name and meta['name'] not in already_names:
                        selected.append({
                            'name': meta['name'],
                            'category': meta['category'],
                            'reason': 'site-triggered',
                        })
                        already_names.add(meta['name'])

        # Contact name detection — inject contacts preference when a known name is mentioned
        prompt_words = set(prompt_lower.split())
        if prompt_words & CONTACT_NAMES:
            for i, meta in enumerate(doc_meta):
                if meta['name'] == 'contacts' and meta['name'] not in already_names:
                    selected.append({
                        'name': meta['name'],
                        'category': meta['category'],
                        'reason': 'contact-triggered',
                    })
                    already_names.add(meta['name'])

    # ── Tier 3: TF-IDF similarity for skill-matched ──
    if intent not in ('converse', 'instruct'):
        query_vec = vec.transform([prompt])
        sims = (doc_vectors * query_vec.T).toarray().flatten()

        # Collect candidates (skill tier only, not already selected)
        candidates = []
        for i, (score, meta) in enumerate(zip(sims, doc_meta)):
            if meta['name'] not in already_names and meta['tier'] == 'skill':
                candidates.append((score, meta))

        candidates.sort(key=lambda x: x[0], reverse=True)

        prev_score = None
        count = 0
        for score, meta in candidates:
            if score < PHASE_B_MIN_THRESHOLD:
                break
            if count >= PHASE_B_MAX_SKILL_RESULTS:
                break
            # Gap detection: stop if score drops by >50% from previous
            if prev_score is not None and score < prev_score * 0.5:
                break
            selected.append({
                'name': meta['name'],
                'category': meta['category'],
                'reason': f'similarity: {score:.2f}',
            })
            already_names.add(meta['name'])
            prev_score = score
            count += 1

    always_count = sum(1 for s in selected if s['reason'] == 'always-include')
    site_count = sum(1 for s in selected if 'site-triggered' in s['reason'])
    skill_count = sum(1 for s in selected if 'similarity' in s['reason'])

    return {
        'selectedMemories': selected,
        'missingMemories': [],
        'toolsNeeded': [],
        'notes': f'Tiered selection from {len(doc_meta)} files: {always_count} always, {site_count} site-triggered, {skill_count} skill-matched',
    }


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    sys.stderr.write('[infer.py] Ready (v2 dual-axis)\n')
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            task = req.get('task')
            if task == 'phase_a':
                result = run_phase_a(req.get('prompt', ''))
            elif task == 'phase_b':
                result = run_phase_b(
                    req.get('prompt', ''),
                    req.get('inventory', []),
                    req.get('intent', 'query'),
                )
            else:
                result = {'error': f'unknown task: {task}'}
        except Exception as e:
            result = {'error': str(e)}

        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
