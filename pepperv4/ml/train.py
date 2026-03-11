"""
Phase A dual-axis classifier.
Trains two models:
  1) intent_classifier  — single-label (query, action, create, converse, instruct)
  2) format_classifier  — multi-label  (inline, file, image, slides, browser)

Saves both plus vectorizers to models/phase_a_v2.pkl.
"""

import pickle, random, re, json
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

INTENT_LABELS = ['query', 'action', 'create', 'converse', 'instruct']
FORMAT_LABELS = ['inline', 'file', 'image', 'slides', 'browser']

REAL_DATA_PATH = Path(__file__).parent / 'data' / 'real_examples.json'
MODEL_OUT = Path(__file__).parent / 'models' / 'phase_a_v2.pkl'

# ── Filler pools for synthetic data ──────────────────────────────────────────

TOPICS = [
    "machine learning", "Python", "climate change", "the stock market",
    "quantum computing", "blockchain", "React hooks", "Docker",
    "neural networks", "supply chain", "cloud computing", "TypeScript",
    "the Cold War", "antibiotics", "solar energy", "space exploration",
    "gene therapy", "agile methodology", "inflation", "the Renaissance",
    "deep learning", "electric vehicles", "photosynthesis", "DNA replication",
    "Kubernetes", "the Roman Empire", "microservices", "CSS grid",
]
APPS = [
    "Chrome", "Spotify", "VS Code", "Slack", "Notepad", "Discord",
    "Excel", "Terminal", "Word", "Telegram", "Outlook", "Firefox",
    "Calculator", "Paint", "Task Manager", "Zoom",
]
FILENAMES = [
    "report.pdf", "notes.txt", "script.py", "config.json", "data.csv",
    "presentation.pptx", "diagram.png", "README.md", "output.xlsx",
]
NAMES = ["Adam", "John", "Sarah", "the team", "my manager", "the client", "Mia"]
SERVICES = ["Gmail", "Google Calendar", "Canvas", "Notion", "Todoist",
            "Google Tasks", "LinkedIn", "GitHub", "Google Docs", "Google Drive"]

# ── Seed templates ───────────────────────────────────────────────────────────

INTENT_SEEDS = {
    "query": [
        "what is {topic}",
        "whats on my {service}",
        "what's on my {service}",
        "check my {service}",
        "what do I have on {service}",
        "explain {topic}",
        "how does {topic} work",
        "summarize {topic} for me",
        "what are the key concepts in {topic}",
        "tell me about {topic}",
        "whats my schedule today",
        "what's due today",
        "what tasks do I have",
        "is there anything on my calendar",
        "who sent me an email",
        "what did {name} say",
        "did I get a response from {name}",
        "how is {topic} different from machine learning",
        "can you check if {name} replied",
        "what are my unread emails",
        "what assignments are due this week",
        "what's the status of my {topic} project",
        "describe {topic}",
        "compare {topic} and blockchain",
        "give me an overview of {topic}",
        "list the pros and cons of {topic}",
        "what should I know about {topic}",
        "break down {topic} into simple terms",
        "how many tasks do I have left",
        "what's my grade in comp",
        "whats the reading for today",
        "did {name} accept the invite",
    ],
    "action": [
        "send an email to {name}",
        "email {name} hi",
        "open {app}",
        "launch {app}",
        "push to github",
        "push yourself to github",
        "navigate to {service}",
        "go to my {service}",
        "add a meeting with {name} tomorrow",
        "add a call for {name} at 3pm to my calendar",
        "delete the file",
        "close {app}",
        "send {name} a message saying hi",
        "forward the email to {name}",
        "connect my bluetooth speaker",
        "fix the bug in the code",
        "run the tests",
        "deploy the app",
        "install {app}",
        "start the server",
        "click on the submit button",
        "move the mouse to the top right",
        "scroll down on the page",
        "continue your work",
        "keep going",
        "do it",
        "go ahead",
        "learn how to send an email via {service}",
        "learn how to use {app}",
        "fix my timecard hours",
        "check out the email from {name}",
        "deal with the warning for future pushes",
        "mark the task as done",
        "tick off the completed items",
        "find {name}'s email address",
        "look through my {service} for work related to {topic}",
        "download the attachment from {name}",
        "upload the file to {service}",
        "update the spreadsheet",
        "set up the environment",
        # Voice message patterns (long, rambling transcriptions)
        "Voice message transcription: I want you to go to {service} and check on {topic} for me also make sure to look at the latest updates",
        "Voice message transcription: alright so I need you to send {name} an email about {topic} and then also check my {service}",
        "Voice message transcription: hey can you go ahead and fix the {topic} issue we talked about also push the changes to github",
        "Voice message transcription: I need you to do the quiz that's in front of you on {service} go ahead and complete it",
        "Voice message transcription: go to {service} and also continue the task from earlier about {topic}",
        "check on my {service} and also handle the {topic} thing",
        "pull the latest changes and look at what needs to be fixed",
        "look at the logs and see what went wrong then fix it",
    ],
    "create": [
        "create a presentation about {topic}",
        "make slides on {topic}",
        "build a slide deck for {topic}",
        "write a Python script for {topic}",
        "generate a PDF report on {topic}",
        "create a file called {filename}",
        "write a report on {topic}",
        "draw me a picture of {topic}",
        "generate an image of {topic}",
        "create a diagram of {topic}",
        "make a chart showing {topic}",
        "build a website for {topic}",
        "design a logo for {topic}",
        "create a questionnaire for {topic}",
        "write a blog post about {topic}",
        "generate me a practice sheet for {topic}",
        "make an infographic about {topic}",
        "produce a document about {topic}",
        "build a tool that does {topic}",
        "create a config file for {topic}",
        "write a shell script to automate {topic}",
        "make a spreadsheet for {topic}",
        "design a form for {topic}",
        "prepare a briefing deck on {topic}",
        "compose a letter to {name} about {topic}",
        "draft an email to {name}",
    ],
    "converse": [
        "yo",
        "hi",
        "hey",
        "hello",
        "yes",
        "no",
        "ok",
        "cool",
        "nice",
        "thanks",
        "good morning",
        "good night",
        "how are you",
        "what's up",
        "just checking in",
        "ping",
        "nothing just testing",
        "tell me a joke",
        "I'm bored",
        "interesting",
        "got it",
        "sure",
        "alright",
        "sounds good",
        "perfect",
        "great",
        "awesome",
        "I see",
        "Ive logged in",
        "just testing",
        "testing",
        "haha",
        "lol",
        "sup",
        "k",
    ],
    "instruct": [
        "from now on use {service} for all tasks",
        "from now on only use {service} through chrome",
        "remember that my main email is user@example.com",
        "never use outlook for anything",
        "always prioritize the faster options",
        "teach yourself to use {app}",
        "when I say todo list I mean {service}",
        "from now on {topic} should be handled differently",
        "make sure to use {name}'s account",
        "specifically use the {name} account",
        "whenever you perform a task tick it off of {service}",
        "remember this format for future use",
        "save this to your memory",
        "don't ever do that again",
        "always check {service} before starting",
        "from now on prioritize {service} over {app}",
        "remember that {name}'s email is test@example.com",
        "when I say X I mean Y",
        "use this approach going forward",
        "stop using {app} and switch to {service}",
    ],
}

FORMAT_SEEDS = {
    # These get combined with intent seeds — format is determined by keywords
    "inline": [],  # default — no extra seeds needed
    "file": [
        "save the output to {filename}",
        "create a file called {filename}",
        "write a Python script",
        "export as {filename}",
        "generate a PDF",
        "write a .py script",
        "save it as {filename}",
        "create a markdown file",
        "produce a document I can download",
        "write a shell script",
        "save the results to {filename}",
    ],
    "image": [
        "draw me a picture of {topic}",
        "generate an image of {topic}",
        "create a diagram of {topic}",
        "make a chart showing {topic}",
        "take a screenshot",
        "visualize {topic}",
        "sketch a concept for {topic}",
        "design a logo",
        "make an infographic",
        "draw a flowchart",
    ],
    "slides": [
        "create a presentation about {topic}",
        "make slides on {topic}",
        "build a slide deck",
        "prepare a PowerPoint about {topic}",
        "make a slideshow",
        "create a pitch deck",
        "design slides for a talk on {topic}",
        "prepare a briefing deck on {topic}",
    ],
    "browser": [
        "go to my {service}",
        "navigate to {service}",
        "check my {service}",
        "open {service} in chrome",
        "whats on my {service}",
        "send an email to {name}",
        "email {name}",
        "look at my {service}",
        "add to my google calendar",
        "open a tab for {service}",
        "check gradescope",
        "go to canvas",
    ],
}

# Combined intent+format seeds for multi-label training
COMBINED_SEEDS = [
    # action + browser
    ("send an email to {name} via {service}", "action", ["browser"]),
    ("go to {service} and check my inbox", "action", ["browser"]),
    ("navigate to {service} and fix the issue", "action", ["browser"]),
    ("push to github and notify {name}", "action", ["inline"]),
    # create + file
    ("write a script and save as {filename}", "create", ["file"]),
    ("generate a report on {topic} and save it", "create", ["file"]),
    ("create a {filename} for the project", "create", ["file"]),
    # create + slides
    ("make a presentation about {topic}", "create", ["slides"]),
    ("build a pitch deck for {name}", "create", ["slides"]),
    # create + image
    ("draw a diagram of {topic}", "create", ["image"]),
    ("generate an infographic about {topic}", "create", ["image"]),
    # create + browser + file
    ("create a Google Doc about {topic} in my Drive", "create", ["file", "browser"]),
    ("save the practice sheet to {service}", "create", ["file", "browser"]),
    # action + browser + file
    ("download the attachment from {service}", "action", ["file", "browser"]),
    ("go to {service} and complete the assignment doc", "action", ["file", "browser"]),
    # query + browser
    ("whats on my {service}", "query", ["browser"]),
    ("check my {service} for updates", "query", ["browser"]),
    ("what assignments are on canvas", "query", ["browser"]),
    ("what emails did {name} send", "query", ["browser"]),
]

# ── Hand-crafted feature extraction ──────────────────────────────────────────

QUESTION_STARTERS = {'what', 'whats', "what's", 'who', 'when', 'where', 'how',
                     'why', 'is', 'are', 'can', 'does', 'do', 'did',
                     'which', 'could', 'should', 'would'}

ACTION_VERB_SET = {'send', 'open', 'push', 'delete', 'navigate', 'add', 'go',
                   'install', 'run', 'close', 'launch', 'click', 'move', 'fix',
                   'continue', 'start', 'stop', 'deploy', 'connect', 'forward',
                   'download', 'upload', 'update', 'set', 'mark', 'tick', 'deal',
                   'learn', 'email', 'save', 'remove', 'check'}

CREATE_VERB_SET = {'create', 'make', 'build', 'write', 'generate', 'draw',
                   'design', 'produce', 'compose', 'draft', 'prepare'}

SERVICE_KEYWORDS = {'gmail', 'calendar', 'gcal', 'canvas', 'notion', 'todoist',
                    'linkedin', 'github', 'google', 'drive', 'docs', 'sheets',
                    'slides', 'gradescope', 'chrome', 'email', 'mail', 'inbox',
                    'tasks', 'browser'}

GREETING_SET = {'hi', 'hey', 'hello', 'yo', 'yes', 'no', 'ok', 'cool', 'nice',
                'thanks', 'sure', 'alright', 'great', 'awesome', 'k', 'lol',
                'haha', 'ping', 'sup', 'bye'}


def extract_hand_features(prompt):
    """Returns a list of 6 numeric features."""
    words = prompt.lower().split()
    first_word = words[0] if words else ''

    starts_question = 1.0 if first_word in QUESTION_STARTERS else 0.0
    has_action = 1.0 if any(w in ACTION_VERB_SET for w in words) else 0.0
    has_create = 1.0 if any(w in CREATE_VERB_SET for w in words) else 0.0
    has_service = 1.0 if any(w in SERVICE_KEYWORDS for w in words) else 0.0
    is_short_greeting = 1.0 if len(words) <= 3 and first_word in GREETING_SET else 0.0
    has_possessive = 1.0 if 'my' in words else 0.0

    return [starts_question, has_action, has_create, has_service,
            is_short_greeting, has_possessive]

HAND_FEATURE_NAMES = ['starts_question', 'has_action', 'has_create',
                      'has_service', 'is_short_greeting', 'has_possessive']


# ── Data generation ──────────────────────────────────────────────────────────

def fill(template):
    t = template
    if '{topic}' in t:
        t = t.replace('{topic}', random.choice(TOPICS))
    if '{app}' in t:
        t = t.replace('{app}', random.choice(APPS))
    if '{filename}' in t:
        t = t.replace('{filename}', random.choice(FILENAMES))
    if '{name}' in t:
        t = t.replace('{name}', random.choice(NAMES))
    if '{service}' in t:
        t = t.replace('{service}', random.choice(SERVICES))
    return t


def generate_intent_data():
    """Generate synthetic intent-labeled data."""
    texts, labels = [], []
    for intent in INTENT_LABELS:
        seeds = INTENT_SEEDS[intent]
        for seed in seeds:
            n_fills = 8 if intent == 'converse' else 12
            seen = set()
            for _ in range(n_fills):
                text = fill(seed).strip()
                if text not in seen:
                    seen.add(text)
                    texts.append(text)
                    labels.append(intent)
    return texts, labels


def generate_format_data():
    """Generate synthetic format-labeled data (multi-label)."""
    texts, labels = [], []

    # Pure format seeds (each maps to one format)
    for fmt in FORMAT_LABELS:
        seeds = FORMAT_SEEDS.get(fmt, [])
        fmt_vec = [1 if f == fmt else 0 for f in FORMAT_LABELS]
        for seed in seeds:
            for _ in range(10):
                text = fill(seed).strip()
                texts.append(text)
                labels.append(list(fmt_vec))

    # Combined intent+format seeds
    for template, _, fmts in COMBINED_SEEDS:
        fmt_vec = [1 if f in fmts else 0 for f in FORMAT_LABELS]
        for _ in range(10):
            text = fill(template).strip()
            texts.append(text)
            labels.append(list(fmt_vec))

    return texts, labels


def load_real_examples():
    """Load manually-labeled real examples from data/real_examples.json."""
    if not REAL_DATA_PATH.exists():
        print(f'[warn] No real data at {REAL_DATA_PATH}')
        return [], [], [], []

    with open(REAL_DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    intent_texts, intent_labels = [], []
    format_texts, format_labels = [], []

    for ex in data:
        prompt = ex.get('prompt', '').strip()
        if not prompt:
            continue
        intent = ex.get('intent', 'query')
        formats = ex.get('formats', ['inline'])

        intent_texts.append(prompt)
        intent_labels.append(intent)

        fmt_vec = [1 if f in formats else 0 for f in FORMAT_LABELS]
        format_texts.append(prompt)
        format_labels.append(fmt_vec)

    return intent_texts, intent_labels, format_texts, format_labels


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    # ── Generate synthetic data ──
    print('Generating synthetic data...')
    syn_intent_texts, syn_intent_labels = generate_intent_data()
    syn_format_texts, syn_format_labels = generate_format_data()

    # ── Load real data (weighted 5x) ──
    real_int_t, real_int_l, real_fmt_t, real_fmt_l = load_real_examples()
    REAL_WEIGHT = 5
    print(f'Real examples: {len(real_int_t)} (weighted {REAL_WEIGHT}x)')

    # ── Combine intent data ──
    all_intent_texts = syn_intent_texts + real_int_t * REAL_WEIGHT
    all_intent_labels = syn_intent_labels + real_int_l * REAL_WEIGHT
    print(f'Total intent examples: {len(all_intent_texts)}')

    from collections import Counter
    print('Intent distribution:', dict(Counter(all_intent_labels)))

    # ── Combine format data ──
    all_format_texts = syn_format_texts + real_fmt_t * REAL_WEIGHT
    all_format_labels = syn_format_labels + real_fmt_l * REAL_WEIGHT
    print(f'Total format examples: {len(all_format_texts)}')

    # ── Build shared vectorizers ──
    print('\nFitting vectorizers...')
    # Union of all texts for fitting
    all_texts = list(set(all_intent_texts + all_format_texts))

    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        analyzer='word',
    )
    word_vec.fit(all_texts)

    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=3000,
        sublinear_tf=True,
        analyzer='char_wb',
    )
    char_vec.fit(all_texts)

    # ── Train intent classifier ──
    print('\nTraining intent classifier...')
    X_int_word = word_vec.transform(all_intent_texts)
    X_int_char = char_vec.transform(all_intent_texts)
    X_int_hand = csr_matrix(np.array([extract_hand_features(t) for t in all_intent_texts]))
    X_intent = hstack([X_int_word, X_int_char, X_int_hand])

    # Encode intent labels as integers
    intent_label_map = {l: i for i, l in enumerate(INTENT_LABELS)}
    y_intent = np.array([intent_label_map[l] for l in all_intent_labels])

    intent_clf = LogisticRegression(
        max_iter=1000, C=1.0, random_state=42,
        solver='lbfgs',
    )
    intent_clf.fit(X_intent, y_intent)

    print('Intent cross-val accuracy (5-fold):')
    scores = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                           solver='lbfgs'),
        X_intent, y_intent, cv=5, scoring='accuracy',
    )
    print(f'  {scores.mean():.3f} +/- {scores.std():.3f}')

    # ── Train format classifier ──
    print('\nTraining format classifier...')
    X_fmt_word = word_vec.transform(all_format_texts)
    X_fmt_char = char_vec.transform(all_format_texts)
    X_fmt_hand = csr_matrix(np.array([extract_hand_features(t) for t in all_format_texts]))
    X_format = hstack([X_fmt_word, X_fmt_char, X_fmt_hand])
    y_format = np.array(all_format_labels)

    format_clf = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    )
    format_clf.fit(X_format, y_format)

    print('Format cross-val F1 per label (5-fold):')
    for i, label in enumerate(FORMAT_LABELS):
        try:
            s = cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0, random_state=42),
                X_format, y_format[:, i], cv=5, scoring='f1',
            )
            print(f'  {label}: {s.mean():.3f} +/- {s.std():.3f}')
        except Exception as e:
            print(f'  {label}: skip ({e})')

    # ── Save model ──
    MODEL_OUT.parent.mkdir(exist_ok=True)
    model = {
        'word_vectorizer': word_vec,
        'char_vectorizer': char_vec,
        'intent_classifier': intent_clf,
        'format_classifier': format_clf,
        'intent_labels': INTENT_LABELS,
        'format_labels': FORMAT_LABELS,
        'hand_feature_names': HAND_FEATURE_NAMES,
    }
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nModel saved to {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.0f} KB)')

    # ── Smoke tests ──
    print('\n=== Smoke Tests ===')
    test_cases = [
        ("Yo", "converse", ["inline"]),
        ("Whats on my gmail", "query", ["browser"]),
        ("Send an email to Adam", "action", ["browser"]),
        ("Create a presentation about React", "create", ["slides"]),
        ("From now on only use gmail", "instruct", ["inline"]),
        ("Push to github", "action", ["inline"]),
        ("Draw me a picture of a cat", "create", ["image"]),
        ("Write a Python script for sorting", "create", ["file"]),
        ("Tell me a joke", "converse", ["inline"]),
        ("Continue your work", "action", ["inline"]),
        ("Yes", "converse", ["inline"]),
        ("Whats on my Google tasks today", "query", ["browser"]),
        ("Add a call for mia tomorrow to my gcal", "action", ["browser"]),
        ("From now on teach yourself to prioritize the faster options", "instruct", ["inline"]),
        ("Check my canvas for assignments", "query", ["browser"]),
    ]

    passed = 0
    for prompt, exp_intent, exp_formats in test_cases:
        X_w = word_vec.transform([prompt])
        X_c = char_vec.transform([prompt])
        X_h = csr_matrix(np.array([extract_hand_features(prompt)]))
        X = hstack([X_w, X_c, X_h])

        pred_intent_idx = intent_clf.predict(X)[0]
        pred_intent = INTENT_LABELS[pred_intent_idx]

        pred_format_probs = format_clf.predict_proba(X)
        pred_formats = [FORMAT_LABELS[i] for i in range(len(FORMAT_LABELS))
                        if pred_format_probs[i][0][1] >= 0.38]
        if not pred_formats:
            best_i = max(range(len(FORMAT_LABELS)),
                         key=lambda i: pred_format_probs[i][0][1])
            pred_formats = [FORMAT_LABELS[best_i]]

        intent_ok = pred_intent == exp_intent
        format_ok = all(f in pred_formats for f in exp_formats)
        ok = intent_ok and format_ok
        if ok:
            passed += 1

        status = 'OK' if ok else 'FAIL'
        print(f'  [{status}] "{prompt[:50]}"')
        if not ok:
            if not intent_ok:
                print(f'         intent: got={pred_intent}, expected={exp_intent}')
            if not format_ok:
                print(f'         format: got={pred_formats}, expected={exp_formats}')

    print(f'\n{passed}/{len(test_cases)} smoke tests passed')


if __name__ == '__main__':
    main()
