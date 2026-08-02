import sys
from pathlib import Path

# Ensure workspace root is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.store import retrieve_context

def main():
    query = "What happens if I change my timezone?"
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    results = retrieve_context(query, top_k=5)

    print(f"\nRetrieved {len(results)} relevant passages:\n")
    found_timezone_doc = False

    for i, res in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Source ID:   {res['source_id']}")
        print(f"File Name:   {res['file_name']}")
        print(f"Title:       {res['title']}")
        print(f"Source Type: {res['source_type']}")
        print(f"Score:       {res['score']}")
        print(f"Passage snippet:\n{res['passage'][:250]}...\n")

        if "KB-003" in res["source_id"] or "03_workspace_settings" in res["file_name"]:
            found_timezone_doc = True

    print("=" * 70)
    if found_timezone_doc:
        print("[SUCCESS] Successfully retrieved the correct timezone documentation (KB-003 / 03_workspace_settings_and_timezones.md)!")
    else:
        print("[FAIL] Timezone document was not retrieved in top results.")
        sys.exit(1)

if __name__ == "__main__":
    main()
