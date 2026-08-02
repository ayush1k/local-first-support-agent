import json
import sys
from pathlib import Path

# Ensure root directory is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import config
from src.state import AgentState
from src.nodes.triage import triage_node

def main():
    print("=" * 70)
    print("TESTING TRIAGE NODE CLASSIFICATION ON SAMPLE QUESTIONS")
    print("=" * 70)

    with open(config.SAMPLE_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    questions = sample_data.get("questions", [])
    expected = {
        "Q-001": "answerable",
        "Q-002": "answerable",
        "Q-003": "requires_clarification",
        "Q-004": "requires_escalation",
        "Q-005": "out_of_scope",
    }

    correct_count = 0

    for q in questions:
        q_id = q["question_id"]
        q_text = q["question"]
        expected_class = expected.get(q_id, "unknown")

        state = AgentState(question=q_text)
        update = triage_node(state)
        classified = update.get("classification")

        matched = (classified == expected_class)
        if matched:
            correct_count += 1

        status_str = "[PASS]" if matched else "[CHECK]"

        print(f"\n{status_str} Question ID: {q_id}")
        print(f"Question:       {q_text[:90]}...")
        print(f"Classified As:  {classified}")
        print(f"Expected:       {expected_class}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {correct_count}/{len(questions)} Questions Classified as Expected.")
    print("=" * 70)

if __name__ == "__main__":
    main()
