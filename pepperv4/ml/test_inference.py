"""
Test Phase A and Phase B inference end-to-end.
Run: python pepperv4/ml/test_inference.py
"""

import sys
import json
import os

# Add parent to path so we can import infer
sys.path.insert(0, os.path.dirname(__file__))
from infer import run_phase_a, run_phase_b

# ── Phase A Test Cases ────────────────────────────────────────────────────────

PHASE_A_TESTS = [
    # (prompt, expected_intent, expected_formats, description)
    ("Yo", "converse", ["inline"], "Short greeting"),
    ("Yes", "converse", ["inline"], "Single word affirmative"),
    ("Whats on my gmail", "query", ["browser"], "Gmail query"),
    ("Send email to Adam", "action", ["browser"], "Email action"),
    ("Create a presentation about React", "create", ["slides"], "Create slides"),
    ("From now on only use gmail", "instruct", None, "Standing instruction"),
    ("Push yourself to github", "action", None, "Git action"),
    ("Add a call for mia to my gcal", "action", ["browser"], "Calendar action"),
    ("Whats on the todo list", "query", None, "Todo query"),
    ("Continue your work", "action", ["inline"], "Continue action"),
    ("Write me a summary", "create", None, "Writing task"),
    ("Hello", "converse", ["inline"], "Greeting"),
    ("Install and set up the following", "action", None, "Install action"),
    ("Check out the email from git guardian", "action", ["browser"], "Check email"),
    ("Whats my comp reading in today?", "query", None, "Course query"),
]

# ── Phase B Test Cases (mock inventory) ───────────────────────────────────────

# Create a minimal mock inventory for Phase B testing
MOCK_INVENTORY = [
    {"name": "browser-preferences", "category": "preference", "path": "", "description": "Browser config"},
    {"name": "gmail", "category": "site", "path": "", "description": "Gmail navigation patterns"},
    {"name": "canvas", "category": "site", "path": "", "description": "Canvas LMS patterns"},
    {"name": "todoist", "category": "site", "path": "", "description": "Todoist task manager"},
    {"name": "notion", "category": "site", "path": "", "description": "Notion workspace"},
    {"name": "coder", "category": "skill", "path": "", "description": "Coding skill"},
    {"name": "writing", "category": "skill", "path": "", "description": "Writing skill"},
    {"name": "ui-designer", "category": "skill", "path": "", "description": "UI design skill"},
]

PHASE_B_TESTS = [
    # (prompt, intent, must_include, must_not_include, description)
    ("Yo", "converse", [], ["browser-preferences", "gmail", "coder"], "Converse → 0 memories"),
    ("Yes", "converse", [], ["browser-preferences"], "Short converse → 0 memories"),
    ("Whats on my gmail", "query", ["browser-preferences", "gmail"], ["coder", "writing"], "Gmail → browser-prefs + gmail"),
    ("Send email to Adam", "action", ["browser-preferences", "gmail"], ["coder", "ui-designer"], "Email → gmail + browser-prefs"),
    ("From now on only use gmail", "instruct", ["browser-preferences"], [], "Instruct → skip skill tier"),
]


def test_phase_a():
    print("=" * 70)
    print("PHASE A TESTS")
    print("=" * 70)
    passed = 0
    failed = 0

    for prompt, expected_intent, expected_formats, desc in PHASE_A_TESTS:
        result = run_phase_a(prompt)
        actual_intent = result.get("intent", "?")
        actual_formats = [k for k, v in result.get("outputLabels", {}).items() if v]

        intent_ok = actual_intent == expected_intent
        format_ok = True
        if expected_formats is not None:
            # Check that all expected formats are present
            format_ok = all(f in actual_formats for f in expected_formats)

        if intent_ok and format_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{status}] {desc}")
        print(f"         Prompt:  \"{prompt}\"")
        print(f"         Intent:  {actual_intent} (expected {expected_intent}) {'✓' if intent_ok else '✗'}")
        if expected_formats is not None:
            print(f"         Formats: {actual_formats} (expected {expected_formats}) {'✓' if format_ok else '✗'}")
        else:
            print(f"         Formats: {actual_formats}")
        print()

    print(f"Phase A: {passed}/{passed + failed} passed")
    return passed, failed


def test_phase_b():
    print()
    print("=" * 70)
    print("PHASE B TESTS (mock inventory — no file reads)")
    print("=" * 70)
    passed = 0
    failed = 0

    for prompt, intent, must_include, must_not_include, desc in PHASE_B_TESTS:
        result = run_phase_b(prompt, MOCK_INVENTORY, intent)
        selected_names = [m["name"] for m in result.get("selectedMemories", [])]

        include_ok = all(n in selected_names for n in must_include)
        exclude_ok = all(n not in selected_names for n in must_not_include)

        if include_ok and exclude_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{status}] {desc}")
        print(f"         Prompt:  \"{prompt}\" (intent={intent})")
        print(f"         Selected: {selected_names}")
        if not include_ok:
            missing = [n for n in must_include if n not in selected_names]
            print(f"         Missing required: {missing}")
        if not exclude_ok:
            unwanted = [n for n in must_not_include if n in selected_names]
            print(f"         Unwanted present: {unwanted}")
        print(f"         Notes: {result.get('notes', '')}")
        print()

    print(f"Phase B: {passed}/{passed + failed} passed")
    return passed, failed


if __name__ == "__main__":
    a_passed, a_failed = test_phase_a()
    b_passed, b_failed = test_phase_b()

    total_passed = a_passed + b_passed
    total_failed = a_failed + b_failed
    print()
    print("=" * 70)
    print(f"TOTAL: {total_passed}/{total_passed + total_failed} passed")
    if total_failed > 0:
        print(f"  {total_failed} FAILURES")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
        sys.exit(0)
