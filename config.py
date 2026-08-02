import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Hugging Face Authentication Token
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
RESOLVED_CASES_PATH = BASE_DIR / "resolved_cases.json"
SAMPLE_QUESTIONS_PATH = BASE_DIR / "sample_questions.json"
OUTPUT_SCHEMA_PATH = BASE_DIR / "output_schema.json"

# Model Configurations
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# Device Selection Logic
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()

# System Settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
