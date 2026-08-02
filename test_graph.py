import json
import sys
from pathlib import Path

# Ensure root folder is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.state import AgentState, ClassificationType
from src.graph import graph


def main():
    print("=" * 70)
    print("TESTING LANGGRAPH END-TO-END WORKFLOW (Q-001)")
    print("=" * 70)

    # 1. Load sample_questions.json and locate Q-001
    sample_file = Path(__file__).parent / "sample_questions.json"
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_item = next((q for q in data.get("questions", []) if q.get("question_id") == "Q-001"), None)
    if not q_item:
        print("[FAIL] Q-001 not found in sample_questions.json")
        sys.exit(1)

    question_text = q_item["question"]
    print(f"Question ID: {q_item['question_id']}")
    print(f"Question:    {question_text}\n")

    # 2. Initialize AgentState
    initial_state = AgentState(question=question_text)

    # 3. Execute graph and print step-by-step routing logs
    print("--- Executing Graph Workflow ---")
    current_state_dict = {}

    for step_output in graph.stream(initial_state):
        for node_name, state_update in step_output.items():
            print(f"\n[NODE ENTERED]: '{node_name}'")
            current_state_dict.update(state_update)

            if node_name == "triage":
                classification = state_update.get("classification")
                print(f" -> Triage Classification: {classification}")

            elif node_name == "retrieval":
                docs = state_update.get("retrieved_docs", [])
                print(f" -> Retrieved Passages: {len(docs)} documents/cases")
                for idx, doc in enumerate(docs[:2], 1):
                    print(f"    Passage {idx}: [{doc.get('source_id')}] {doc.get('title')}")

            elif node_name == "generation":
                draft = state_update.get("draft_answer", "")
                print(f" -> Draft Answer Generated ({len(draft)} chars):")
                print(f"    {draft[:150]}...")

            elif node_name == "verification":
                attempts = state_update.get("verification_attempts")
                final_out = state_update.get("final_output")
                print(f" -> Verification Attempts: {attempts}")
                if final_out:
                    print(f" -> Confidence: {final_out.confidence}")
                    print(f" -> Requires Human: {final_out.requires_human}")
                    print(f" -> Cited Sources Count: {len(final_out.sources)}")

            elif node_name == "safe_failure":
                print(" -> [SAFE FAILURE FALLBACK ROUTED]")

    # 4. Final state validation
    final_output = current_state_dict.get("final_output")

    print("\n" + "=" * 70)
    print("FINAL WORKFLOW EXECUTION SUMMARY")
    print("=" * 70)

    if final_output is not None:
        print(f"Classification:     {final_output.classification}")
        print(f"Confidence Score:   {final_output.confidence}")
        print(f"Requires Human:     {final_output.requires_human}")
        print(f"Reason:             {final_output.reason}")
        print(f"Cited Sources:      {[s.source_id for s in final_output.sources]}")
        print("\nFinal Answer Payload:")
        print(final_output.answer)

        # Verification check
        if final_output.classification == ClassificationType.ANSWERABLE and len(final_output.answer) > 0:
            print("\n[SUCCESS] End-to-end graph orchestration verified successfully for Q-001!")
            sys.exit(0)
        else:
            print("\n[FAIL] Unexpected final output classification or empty answer.")
            sys.exit(1)
    else:
        print("\n[FAIL] Workflow completed without setting final_output state.")
        sys.exit(1)


if __name__ == "__main__":
    main()
