import sys
from pathlib import Path
from pydantic import ValidationError

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.state import AgentState, FinalOutput, SourceItem, ClassificationType


def main():
    print("=" * 60)
    print("TESTING PYDANTIC AGENTSTATE & FINALOUTPUT VALIDATION")
    print("=" * 60)

    # 1. Instantiate state with dummy data
    dummy_source = SourceItem(
        source_id="doc_dashboards.md",
        passage="Dashboards can be created via the + New Dashboard button."
    )

    dummy_output = FinalOutput(
        classification=ClassificationType.ANSWERABLE,
        answer="To create a dashboard in OrbitDesk, click the '+ New Dashboard' button on the navigation bar.",
        sources=[dummy_source],
        confidence=0.95,
        requires_human=False,
        reason="Answer found directly in doc_dashboards.md",
        warnings=[]
    )

    state = AgentState(
        question="How do I create a new dashboard?",
        classification=ClassificationType.ANSWERABLE,
        retrieved_docs=[
            {"id": "doc_dashboards.md", "content": "Dashboards guide...", "score": 0.92}
        ],
        draft_answer="To create a dashboard in OrbitDesk, click the '+ New Dashboard' button.",
        verification_attempts=1,
        final_output=dummy_output
    )

    print("\n[SUCCESS] AgentState successfully instantiated:")
    print(state.model_dump_json(indent=2))

    # 2. Test validation failure case (invalid confidence score > 1.0)
    print("\nTesting Pydantic validation error handling (confidence > 1.0)...")
    try:
        FinalOutput(
            classification=ClassificationType.ANSWERABLE,
            answer="Test answer",
            sources=[],
            confidence=1.5,  # Out of range [0, 1]
            requires_human=False,
            reason="Testing validation"
        )
        print("[FAIL] Expected ValidationError but passed!")
        sys.exit(1)
    except ValidationError as e:
        print(f"[SUCCESS] Caught expected ValidationError:\n{e}")

    print("\n" + "=" * 60)
    print("PYDANTIC STATE VALIDATION TEST PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
