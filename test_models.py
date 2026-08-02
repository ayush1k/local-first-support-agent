import sys
import time
from pathlib import Path

# Ensure root folder is in python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.models import get_llm

def main():
    print("=" * 60)
    print("TESTING LOCAL HUGGING FACE MODEL LOADING & GENERATION")
    print("=" * 60)

    start_init = time.time()
    llm = get_llm()
    init_duration = time.time() - start_init

    prompt = "Say hello world"
    print(f"\nPrompt: '{prompt}'")
    print("Generating response...")

    start_gen = time.time()
    response = llm.generate(prompt=prompt, system_prompt="You are a helpful assistant.")
    gen_duration = time.time() - start_gen

    print("\n" + "=" * 60)
    print(f"MODEL RESPONSE:\n{response}")
    print("=" * 60)
    print(f"Device Used:            {llm.device}")
    print(f"Model Load Time:        {llm.load_time:.2f} seconds")
    print(f"Generation Time:        {gen_duration:.2f} seconds")
    print("=" * 60)

    if response:
        print("[SUCCESS] Local LLM model loaded and generated response successfully!")
    else:
        print("[FAIL] Empty response received from model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
