import sys
import torch
import config

def main():
    print("=" * 50)
    print("LOCAL-FIRST SUPPORT AGENT - ENVIRONMENT TEST")
    print("=" * 50)
    print(f"Python Version:         {sys.version.split()[0]}")
    print(f"PyTorch Version:        {torch.__version__}")
    print(f"Detected Device:        {config.DEVICE}")
    print(f"Embedding Model:        {config.EMBEDDING_MODEL_NAME}")
    print(f"LLM Model:              {config.LLM_MODEL_NAME}")
    print(f"Base Directory:         {config.BASE_DIR}")
    print(f"Knowledge Base Dir:     {config.KNOWLEDGE_BASE_DIR} (Exists: {config.KNOWLEDGE_BASE_DIR.exists()})")
    print(f"Resolved Cases Path:    {config.RESOLVED_CASES_PATH} (Exists: {config.RESOLVED_CASES_PATH.exists()})")
    print(f"Sample Questions Path:  {config.SAMPLE_QUESTIONS_PATH} (Exists: {config.SAMPLE_QUESTIONS_PATH.exists()})")
    print(f"Output Schema Path:     {config.OUTPUT_SCHEMA_PATH} (Exists: {config.OUTPUT_SCHEMA_PATH.exists()})")
    print("=" * 50)
    print("ENVIRONMENT TEST COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
