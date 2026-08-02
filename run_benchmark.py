import json
import time
import sys
from pathlib import Path
import jsonschema

# Ensure root folder is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.state import AgentState, ClassificationType
from src.graph import graph


def main():
    print("=" * 80)
    print("ORBITDESK LOCAL SUPPORT AGENT - PRE-SUBMISSION INTEGRATION BENCHMARK")
    print("=" * 80)

    # Load output_schema.json
    schema_path = Path(__file__).parent / "output_schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        output_schema = json.load(f)

    # Load sample_questions.json
    sample_path = Path(__file__).parent / "sample_questions.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    questions = sample_data.get("questions", [])
    print(f"Loaded {len(questions)} test cases from sample_questions.json\n")

    benchmark_results = []

    for q_item in questions:
        q_id = q_item.get("question_id", "UNKNOWN")
        q_text = q_item.get("question", "")

        print("-" * 80)
        print(f"RUNNING BENCHMARK CASE: [{q_id}]")
        print(f"Question: {q_text}")
        print("-" * 80)

        initial_state = AgentState(question=q_text)
        routing_path = []
        state_updates = {}

        start_time = time.time()

        # Execute stream to log node transitions
        for step in graph.stream(initial_state):
            for node_name, update_dict in step.items():
                routing_path.append(node_name)
                state_updates.update(update_dict)
                print(f" -> Transitioned to node: '{node_name}'")

        end_time = time.time()
        latency = round(end_time - start_time, 2)
        routing_str = " -> ".join(routing_path) + " -> END"

        final_output = state_updates.get("final_output")

        schema_valid = False
        classification_val = "N/A"
        requires_human = "N/A"

        if final_output is not None:
            final_dict = final_output.model_dump(mode="json")
            classification_val = final_dict.get("classification")
            requires_human = final_dict.get("requires_human")

            # Validate against JSON Schema
            try:
                jsonschema.validate(instance=final_dict, schema=output_schema)
                schema_valid = True
                print(f" [Schema Validation]: PASSED against output_schema.json")
            except jsonschema.ValidationError as ve:
                schema_valid = False
                print(f" [Schema Validation]: FAILED - {ve.message}")

            print(f" [Classification]:   {classification_val}")
            print(f" [Confidence Score]: {final_dict.get('confidence')}")
            print(f" [Requires Human]:   {requires_human}")
            print(f" [Reason]:           {final_dict.get('reason')}")
            print(f" [Sources Count]:    {len(final_dict.get('sources', []))}")
            for src in final_dict.get("sources", []):
                print(f"   - Source ID: {src.get('source_id')}")

        print(f" [Routing Path]:     {routing_str}")
        print(f" [Execution Latency]: {latency} seconds")

        status = "PASS" if schema_valid and final_output is not None else "FAIL"

        benchmark_results.append({
            "case_id": q_id,
            "question": q_text[:40] + "...",
            "classification": classification_val,
            "routing_path": " -> ".join(routing_path),
            "latency": latency,
            "schema_valid": schema_valid,
            "status": status,
        })
        print("\n")

    # Print Final Benchmark Summary Report Table
    print("=" * 95)
    print("FINAL BENCHMARK SUMMARY REPORT")
    print("=" * 95)
    print(f"| {'Case ID':<8} | {'Classification':<22} | {'Routing Path':<35} | {'Latency (s)':<11} | {'Schema Valid':<12} | {'Status':<6} |")
    print("-" * 95)

    all_passed = True
    for res in benchmark_results:
        c_id = res["case_id"]
        cls = res["classification"]
        path = res["routing_path"]
        lat = f"{res['latency']:.2f}"
        s_val = "YES" if res["schema_valid"] else "NO"
        st = res["status"]
        if st != "PASS":
            all_passed = False
        print(f"| {c_id:<8} | {cls:<22} | {path:<35} | {lat:<11} | {s_val:<12} | {st:<6} |")

    print("=" * 95)

    if all_passed:
        print("\n[SUCCESS] All 5 benchmark test cases executed cleanly with 100% schema validation compliance!")
        sys.exit(0)
    else:
        print("\n[FAIL] One or more benchmark test cases failed validation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
