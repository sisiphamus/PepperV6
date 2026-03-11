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
    "recursion", "graph theory", "linear algebra", "sorting algorithms",
    "database indexing", "REST APIs", "OAuth", "websockets",
    "data structures", "big data", "serverless", "containerization",
    "cryptography", "networking", "operating systems", "compilers",
]
APPS = [
    "Chrome", "Spotify", "VS Code", "Slack", "Notepad", "Discord",
    "Excel", "Terminal", "Word", "Telegram", "Outlook", "Firefox",
    "Calculator", "Paint", "Task Manager", "Zoom", "Figma", "Postman",
    "Docker Desktop", "GitHub Desktop", "Obsidian", "Notion",
]
FILENAMES = [
    "report.pdf", "notes.txt", "script.py", "config.json", "data.csv",
    "presentation.pptx", "diagram.png", "README.md", "output.xlsx",
    "summary.docx", "analysis.ipynb", "template.html", "export.zip",
]
NAMES = ["Adam", "John", "Sarah", "the team", "my manager", "the client", "Mia",
         "Julia", "Dr. Bunge", "Sophiya", "the professor", "my advisor"]
SERVICES = ["Gmail", "Google Calendar", "Canvas", "Notion", "Todoist",
            "Google Tasks", "LinkedIn", "GitHub", "Google Docs", "Google Drive",
            "Gradescope", "Slack", "Google Sheets", "io.rice.edu"]

# ── Seed templates ───────────────────────────────────────────────────────────

INTENT_SEEDS = {
    "query": [
        # What/check info (not requiring an action)
        "what is {topic}",
        "whats on my {service}",
        "what's on my {service}",
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
        "what's on the todo list",
        "what assignments are on canvas",
        "what emails did {name} send",
        "check my {service} for updates",
        "what's due this week on {service}",
        "how many unread messages do I have",
        "what's the latest on {topic}",
        "have I gotten any emails about {topic}",
        "what does {name} want",
        "is there a meeting today",
        "when is the next deadline",
        "look at my {service} and tell me what's there",
        "check my {service} and tell me",
        "what's on my {service} today",
        "read my {service} and summarize",
        "look at my emails",
        "show me what's on {service}",
        "pull up my {service} and tell me",
        "what grade did I get on the last assignment",
        "check if {name} responded",
        "what's my current grade",
        "how did I do on the last exam",
        "also look at my logs",
        "look at the recent outputs",
        "check the logs and tell me what happened",
        "also check the email from {name}",
        "look through my {service} for anything about {topic}",
        "see if there are any new messages",
        "what's in my inbox",
        "find out what happened with {topic}",
        "look at the {service} page and tell me",
        "{name} should have granted me access, did they",
        "did {name} grant me access",
        "check if that worked",
        "what does the error say",
        "look at the results",
    ],
    "action": [
        # Do something, take effect
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
        "deal with the warning for future pushes",
        "mark the task as done",
        "tick off the completed items",
        "find {name}'s email address",
        "look through my {service} for work related to {topic}",
        "download the attachment from {name}",
        "upload the file to {service}",
        "update the spreadsheet",
        "set up the environment",
        "go to {service} and complete the task",
        "go to {service} and do {topic}",
        "go to canvas and submit the assignment",
        "go to my gmail and reply to {name}",
        "go to {service} and fix the issue",
        "go to {service} and do what needs to be done",
        "pull the latest changes and look at what needs to be fixed",
        "look at the logs and see what went wrong then fix it",
        "check out the email from {name} and deal with it",
        "also look at the email and handle it",
        "go over the document and make changes",
        "continue the task from earlier",
        "continue this task",
        "go back and fix",
        "log my hours on {service}",
        "log my hours for this week",
        "add my hours to {service}",
        "rotate the pictures and fix the layout",
        "do the quiz that's in front of you",
        "there is a quiz in front of you do it",
        "complete the assignment in front of you",
        "complete the form",
        "go to {service} and complete the quiz",
        "go to {service} and study up on it",
        "learn how the course works on {service}",
        "schedule an email to {name} for tomorrow",
        "schedule a message to {name}",
        "send {name} an email about {topic} tomorrow morning",
        "go to {service} and also check my inbox",
        "push the changes to github",
        "push and notify {name}",
        # Voice message patterns
        "Voice message transcription: I want you to go to {service} and check on {topic} for me also make sure to look at the latest updates",
        "Voice message transcription: alright so I need you to send {name} an email about {topic} and then also check my {service}",
        "Voice message transcription: hey can you go ahead and fix the {topic} issue we talked about also push the changes to github",
        "Voice message transcription: I need you to do the quiz that's in front of you on {service} go ahead and complete it",
        "Voice message transcription: go to {service} and also continue the task from earlier about {topic}",
        "Voice message transcription: I want you to work on improving model A and model B look at recent examples of your logs",
        "check on my {service} and also handle the {topic} thing",
        "install and set up the following tools",
        "set up {topic} for me",
        "go over the draft and make changes",
        "go over my {service} and do what needs to be done",
        "look at my gmail and reply to any important emails",
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
        "write me a summary of {topic}",
        "generate a study guide for {topic}",
        "make a cheat sheet on {topic}",
        "create a template for {topic}",
        "write a script that does {topic}",
        "generate practice problems for {topic}",
        "build me a {topic} tracker",
        "create a {topic} planner",
        "write a memo about {topic}",
        "draft a proposal for {topic}",
        "produce a slide deck explaining {topic}",
        "create a Google doc about {topic}",
        "make a form for {topic}",
        "build an app for {topic}",
        "write the code for {topic}",
        "implement {topic}",
        "set up a project for {topic}",
        "create a section in the form for {topic}",
        "make yourself auth run on startup",
        "make a memory file for {topic}",
        "write a skill file for {topic}",
    ],
    "converse": [
        # Simple greetings / reactions / casual chat
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
        "make me laugh",
        "tell me something funny",
        "say something",
        "chat with me",
        "I'm listening",
        "talk to me",
        "entertain me",
        "what do you think",
        "what's your opinion",
        "any thoughts",
        "thoughts",
        "hmm",
        "hm",
        "interesting thought",
        "fair enough",
        "makes sense",
        "okay okay",
        "not bad",
        "good point",
        "right",
        "exactly",
        "of course",
        "obviously",
        "true",
        "indeed",
        "noted",
        "understood",
        "copy that",
        "roger that",
        "on it",
        "brb",
        "afk",
        "back",
        "I'm back",
        "hey there",
        "testing testing",
        "one two three",
        "can you hear me",
        "Voice message transcription: testing the voice message can you hear me",
        "tell me a fun fact",
        "give me a fun fact",
        "surprise me",
        "say hi",
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
        "note that my main account is {service}",
        "make a note of this",
        "remember this",
        "memorize this",
        "save this preference",
        "going forward always use {service}",
        "in the future use {service} not {app}",
        "from now on when I say {topic} I mean this",
        "teach yourself how to do {topic}",
        "learn how to handle {topic} for future tasks",
        "make sure to always {topic} first",
        "never {topic} without asking me first",
        "can you decode a voice message if I send it",
        "figure out how to handle {topic} in the future",
        "remember that {name} prefers {service}",
        "add {name}'s email to your memory",
        "save {name}'s contact info",
        "update your memory with this",
        "keep this in mind for next time",
        "from now on teach yourself to prioritize the faster options",
        "delete any apps script after you use them",
        "always save outputs to {service}",
        "log any important findings to your memory",
        "when you do a task also tick it off {service}",
        "specifically use the {service} account not {app}",
        "make note that I use {service} for tasks",
    ],
}

FORMAT_SEEDS = {
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
        "write a script and save it",
        "output to a file",
        "save this to a text file",
        "create a downloadable report",
        "export the data as {filename}",
        "write code and save it as {filename}",
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
        "generate a photo of {topic}",
        "create a visual for {topic}",
        "draw a diagram",
        "create an illustration of {topic}",
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
        "make a Google Slides presentation",
        "build me a deck on {topic}",
        "create slides explaining {topic}",
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
        "go to {service} and do",
        "open gmail",
        "navigate to gmail",
        "go to google drive",
        "log into {service}",
        "submit on {service}",
        "reply to {name} on {service}",
        "go to {service} and complete",
        "go to canvas and submit",
        "schedule an email to {name}",
        "send a message via {service}",
        "forward the email via gmail",
        "add an event to my calendar",
    ],
}

# Combined intent+format seeds for multi-label training
COMBINED_SEEDS = [
    # action + browser
    ("send an email to {name} via {service}", "action", ["browser"]),
    ("go to {service} and check my inbox", "action", ["browser"]),
    ("navigate to {service} and fix the issue", "action", ["browser"]),
    ("push to github and notify {name}", "action", ["inline"]),
    ("go to canvas and submit the assignment", "action", ["browser"]),
    ("go to gmail and reply to {name}", "action", ["browser"]),
    ("schedule an email to {name} for tomorrow morning", "action", ["browser"]),
    ("add an event to my gcal for tomorrow", "action", ["browser"]),
    ("log my hours on io.rice.edu", "action", ["browser"]),
    ("go to gradescope and check my submission", "action", ["browser"]),
    ("go to {service} and do the quiz", "action", ["browser"]),
    ("complete the quiz on {service}", "action", ["browser"]),
    ("go to {service} and complete the form", "action", ["browser"]),
    # create + file
    ("write a script and save as {filename}", "create", ["file"]),
    ("generate a report on {topic} and save it", "create", ["file"]),
    ("create a {filename} for the project", "create", ["file"]),
    ("write code and export as {filename}", "create", ["file"]),
    ("generate a PDF of {topic}", "create", ["file"]),
    # create + slides
    ("make a presentation about {topic}", "create", ["slides"]),
    ("build a pitch deck for {name}", "create", ["slides"]),
    ("create google slides on {topic}", "create", ["slides"]),
    # create + image
    ("draw a diagram of {topic}", "create", ["image"]),
    ("generate an infographic about {topic}", "create", ["image"]),
    ("make a chart for {topic}", "create", ["image"]),
    # create + browser + file
    ("create a Google Doc about {topic} in my Drive", "create", ["file", "browser"]),
    ("save the practice sheet to {service}", "create", ["file", "browser"]),
    ("make a Google sheet for {topic}", "create", ["file", "browser"]),
    # action + browser + file
    ("download the attachment from {service}", "action", ["file", "browser"]),
    ("go to {service} and complete the assignment doc", "action", ["file", "browser"]),
    # query + browser
    ("whats on my {service}", "query", ["browser"]),
    ("check my {service} for updates", "query", ["browser"]),
    ("what assignments are on canvas", "query", ["browser"]),
    ("what emails did {name} send", "query", ["browser"]),
    ("what's my grade on {service}", "query", ["browser"]),
    ("look at my {service} and tell me what's there", "query", ["browser"]),
    ("what's in my gmail inbox", "query", ["browser"]),
    ("check {service} and tell me what's due", "query", ["browser"]),
    ("what's on my google tasks", "query", ["browser"]),
    ("check gradescope for my latest submission", "query", ["browser"]),
    ("what's my grade on the latest comp homework", "query", ["browser"]),
    # converse + inline
    ("make me laugh", "converse", ["inline"]),
    ("tell me a joke", "converse", ["inline"]),
    ("tell me something funny", "converse", ["inline"]),
    ("make me smile", "converse", ["inline"]),
    ("give me a fun fact", "converse", ["inline"]),
    ("tell me a fun fact", "converse", ["inline"]),
    ("surprise me with something", "converse", ["inline"]),
    ("entertain me", "converse", ["inline"]),
    # instruct + inline (no browser even if service mentioned)
    ("from now on only use gmail through chrome", "instruct", ["inline"]),
    ("from now on use {service} for everything", "instruct", ["inline"]),
    ("remember that my main email is user@example.com", "instruct", ["inline"]),
    ("never use outlook for anything", "instruct", ["inline"]),
    ("when I say todo list I mean google tasks", "instruct", ["inline"]),
    ("always check {service} before starting a task", "instruct", ["inline"]),
    ("from now on teach yourself to prioritize the faster options", "instruct", ["inline"]),
    ("delete any apps script after you use them", "instruct", ["inline"]),
    ("save this to your memory", "instruct", ["inline"]),
    ("remember this preference", "instruct", ["inline"]),
    ("note that I use {service} for {topic}", "instruct", ["inline"]),
    ("going forward use {service} not {app}", "instruct", ["inline"]),
]

# ── Hand-crafted feature extraction ──────────────────────────────────────────

QUESTION_STARTERS = {'what', 'whats', "what's", 'who', 'when', 'where', 'how',
                     'why', 'is', 'are', 'can', 'does', 'do', 'did',
                     'which', 'could', 'should', 'would', 'have', 'has'}

ACTION_VERB_SET = {'send', 'open', 'push', 'delete', 'navigate', 'add', 'go',
                   'install', 'run', 'close', 'launch', 'click', 'move', 'fix',
                   'continue', 'start', 'stop', 'deploy', 'connect', 'forward',
                   'download', 'upload', 'update', 'set', 'mark', 'tick', 'deal',
                   'learn', 'email', 'save', 'remove', 'log', 'submit', 'rotate',
                   'complete', 'schedule', 'reply', 'pull', 'implement', 'set up'}

# Note: 'check' removed from ACTION_VERB_SET — it's ambiguous (query "check my email" vs action)
# The TF-IDF + context will handle check correctly

CREATE_VERB_SET = {'create', 'make', 'build', 'write', 'generate', 'draw',
                   'design', 'produce', 'compose', 'draft', 'prepare', 'implement'}

SERVICE_KEYWORDS = {'gmail', 'calendar', 'gcal', 'canvas', 'notion', 'todoist',
                    'linkedin', 'github', 'google', 'drive', 'docs', 'sheets',
                    'slides', 'gradescope', 'chrome', 'email', 'mail', 'inbox',
                    'tasks', 'browser', 'io.rice', 'slack', 'discord'}

GREETING_SET = {'hi', 'hey', 'hello', 'yo', 'yes', 'no', 'ok', 'cool', 'nice',
                'thanks', 'sure', 'alright', 'great', 'awesome', 'k', 'lol',
                'haha', 'ping', 'sup', 'bye', 'noted', 'understood', 'true',
                'indeed', 'right', 'exactly', 'perfect', 'interesting', 'hmm'}

INSTRUCT_STARTERS = {'from', 'never', 'always', 'remember', 'when', 'whenever',
                     'specifically', 'going', 'note', 'save', 'memorize', 'keep',
                     'in', 'make', 'teach', 'figure', 'add', 'update', 'log'}

INSTRUCT_PHRASES = ['from now on', 'going forward', 'in the future', 'make a note',
                    'save this', 'remember this', 'remember that', 'note that',
                    'always use', 'never use', 'make sure to', 'teach yourself',
                    'keep in mind', 'memorize', 'update your memory',
                    'from now on teach', 'delete any']

CASUAL_PHRASES = ['tell me a joke', 'make me laugh', 'tell me something funny',
                  'chat with me', 'talk to me', 'entertain me', 'fun fact',
                  'tell me a fun fact', 'give me a fun fact', 'surprise me',
                  'how are you', "what's up", 'what do you think', 'any thoughts']


def extract_hand_features(prompt):
    """Returns a list of numeric features."""
    words = prompt.lower().split()
    first_word = words[0] if words else ''
    prompt_lower = prompt.lower()

    starts_question = 1.0 if first_word in QUESTION_STARTERS else 0.0
    has_action = 1.0 if any(w in ACTION_VERB_SET for w in words) else 0.0
    has_create = 1.0 if any(w in CREATE_VERB_SET for w in words) else 0.0
    has_service = 1.0 if any(w in SERVICE_KEYWORDS for w in words) else 0.0
    is_short_greeting = 1.0 if len(words) <= 3 and first_word in GREETING_SET else 0.0
    has_possessive = 1.0 if 'my' in words else 0.0
    is_instruct = 1.0 if any(phrase in prompt_lower for phrase in INSTRUCT_PHRASES) else 0.0
    is_casual = 1.0 if any(phrase in prompt_lower for phrase in CASUAL_PHRASES) else 0.0
    is_long = 1.0 if len(words) > 30 else 0.0
    starts_go = 1.0 if first_word == 'go' else 0.0
    has_check_query = 1.0 if 'check' in words and (starts_question or has_possessive) else 0.0
    # "from now on", "going forward" etc — strong instruct signal
    instruct_starter = 1.0 if first_word in INSTRUCT_STARTERS and is_instruct else 0.0
    # Voice message
    is_voice = 1.0 if 'voice message' in prompt_lower or 'transcription' in prompt_lower else 0.0

    return [starts_question, has_action, has_create, has_service,
            is_short_greeting, has_possessive, is_instruct, is_casual,
            is_long, starts_go, has_check_query, instruct_starter, is_voice]

HAND_FEATURE_NAMES = ['starts_question', 'has_action', 'has_create',
                      'has_service', 'is_short_greeting', 'has_possessive',
                      'is_instruct', 'is_casual', 'is_long', 'starts_go',
                      'has_check_query', 'instruct_starter', 'is_voice']


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
            n_fills = 6 if intent == 'converse' else 10
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

    # ── Load real data (weighted 8x) ──
    real_int_t, real_int_l, real_fmt_t, real_fmt_l = load_real_examples()
    REAL_WEIGHT = 8
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
    all_texts = list(set(all_intent_texts + all_format_texts))

    word_vec = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=8000,
        sublinear_tf=True,
        analyzer='word',
        min_df=1,
    )
    word_vec.fit(all_texts)

    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=4000,
        sublinear_tf=True,
        analyzer='char_wb',
        min_df=1,
    )
    char_vec.fit(all_texts)

    # ── Train intent classifier ──
    print('\nTraining intent classifier...')
    X_int_word = word_vec.transform(all_intent_texts)
    X_int_char = char_vec.transform(all_intent_texts)
    X_int_hand = csr_matrix(np.array([extract_hand_features(t) for t in all_intent_texts]))
    X_intent = hstack([X_int_word, X_int_char, X_int_hand])

    intent_label_map = {l: i for i, l in enumerate(INTENT_LABELS)}
    y_intent = np.array([intent_label_map[l] for l in all_intent_labels])

    intent_clf = LogisticRegression(
        max_iter=2000, C=2.0, random_state=42,
        solver='lbfgs',
    )
    intent_clf.fit(X_intent, y_intent)

    print('Intent cross-val accuracy (5-fold):')
    scores = cross_val_score(
        LogisticRegression(max_iter=2000, C=2.0, random_state=42,
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
        LogisticRegression(max_iter=2000, C=2.0, random_state=42),
    )
    format_clf.fit(X_format, y_format)

    print('Format cross-val F1 per label (5-fold):')
    for i, label in enumerate(FORMAT_LABELS):
        try:
            s = cross_val_score(
                LogisticRegression(max_iter=2000, C=2.0, random_state=42),
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
        # Original failing cases
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
        # Extra coverage
        ("Make me laugh", "converse", ["inline"]),
        ("Tell me something funny", "converse", ["inline"]),
        ("What's my grade on the latest comp homework", "query", ["browser"]),
        ("Check gradescope for my latest submission", "query", ["browser"]),
        ("Schedule an email to Dr. Bunge for tomorrow morning", "action", ["browser"]),
        ("Log my hours on io.rice.edu for this week", "action", ["browser"]),
        ("Generate a PDF report on climate change", "create", ["file"]),
        ("Make slides on machine learning", "create", ["slides"]),
        ("From now on use Gmail for all tasks", "instruct", ["inline"]),
        ("Remember that my main email is user@example.com", "instruct", ["inline"]),
        ("Save this to your memory", "instruct", ["inline"]),
        ("Delete any apps script after you use them", "instruct", ["inline"]),
        ("Go to canvas and submit the assignment", "action", ["browser"]),
        ("There is a quiz in front of you, do it", "action", ["browser"]),
        ("What's on my Google tasks", "query", ["browser"]),
        ("What's in my inbox", "query", ["browser"]),
        ("Also look at my logs", "query", ["inline"]),
        ("Look at the recent outputs", "query", ["inline"]),
        ("Voice message transcription: testing can you hear me", "converse", ["inline"]),
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
        print(f'  [{status}] "{prompt[:60]}"')
        if not ok:
            if not intent_ok:
                print(f'         intent: got={pred_intent}, expected={exp_intent}')
            if not format_ok:
                print(f'         format: got={pred_formats}, expected={exp_formats}')

    print(f'\n{passed}/{len(test_cases)} smoke tests passed')


if __name__ == '__main__':
    main()
