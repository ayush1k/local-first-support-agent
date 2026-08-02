import json
from pydantic import BaseModel
from src.graph import graph # or 'workflow' / 'compiled_graph' depending on your variable name

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