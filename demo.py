import json
from pydantic import BaseModel

print("==================================================")
print("Booting up OrbitDesk Local Support Agent...")
print("Initializing and downloading local models...")
print("==================================================\n")

# Importing the graph will trigger any module-level initializations
from src.graph import graph 

# Force the LLM and Embedding models to load into memory by running a warm-up query.
print("\n[System] Performing initial model warm-up. Please wait...\n")
_ = graph.invoke({'question': 'warm-up query'})
print("\n[System] Models loaded successfully! Ready for live testing.\n")
print("==================================================")

def run_query(query):
    print(f"\n[USER QUERY]: {query}")
    print("-" * 50)
    result = graph.invoke({'question': query})
    print("\n--- Final JSON Output ---")
    final_output = result.get('final_output')
    if isinstance(final_output, BaseModel):
        print(final_output.model_dump_json(indent=2))
    elif hasattr(final_output, 'model_dump_json'):
        print(final_output.model_dump_json(indent=2))
    elif hasattr(final_output, 'model_dump'):
        print(json.dumps(final_output.model_dump(mode='json'), indent=2))
    else:
        print(json.dumps(final_output, indent=2))
    print("=" * 50)

if __name__ == "__main__":
    while True:
        q = input("\nEnter a support question (or 'exit'): ")
        if q.lower() == 'exit':
            break
        run_query(q)