import sys
from pathlib import Path

# Ensure root folder is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.state import AgentState, ClassificationType
from src.store import retrieve_context
from src.nodes.generation import generation_node
from src.nodes.verification import verification_node

def main():
    print("=" * 70)
    print("TESTING GENERATION & VERIFICATION NODES")
    print("=" * 70)

    # 1. Prepare simulated initial state
    question = "What happens if I change my workspace timezone?"
    retrieved = retrieve_context(question, top_k=2)

    initial_state = AgentState(
        question=question,
        classification=ClassificationType.ANSWERABLE,
        retrieved_docs=retrieved,
        verification_attempts=0
    )

    print(f"Question:           {initial_state.question}", flush=True)
    print(f"Retrieved Docs:     {len(initial_state.retrieved_docs)} passages", flush=True)
    print(f"Initial Attempts:   {initial_state.verification_attempts}", flush=True)

    # 2. Run Generation Node
    print("\n--- Running Generation Node ---", flush=True)
    gen_update = generation_node(initial_state)
    draft_answer = gen_update.get("draft_answer", "")

    # Update state with draft_answer
    state_after_gen = initial_state.model_copy(update=gen_update)
    print(f"Draft Answer Generated:\n{draft_answer}\n", flush=True)

    # 3. Run Verification Node
    print("--- Running Verification Node ---", flush=True)
    verif_update = verification_node(state_after_gen)
    final_state = state_after_gen.model_copy(update=verif_update)

    print(f"Verification Attempts: {final_state.verification_attempts}", flush=True)
    print(f"Final Output Object:\n{final_state.final_output.model_dump_json(indent=2)}", flush=True)

    print("\n" + "=" * 70, flush=True)
    if final_state.verification_attempts == 1 and final_state.final_output is not None:
        print("[SUCCESS] Generation and Verification nodes executed successfully!", flush=True)
    else:
        print("[FAIL] Verification node failed to produce valid output or increment attempts.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
