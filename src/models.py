import os
import time
from typing import Optional, List, Dict, Any

# Suppress HF, tqdm, and tokenizers verbosity/progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import config

transformers.logging.set_verbosity_error()





class LocalLLM:
    def __init__(
        self,
        model_name: str = config.LLM_MODEL_NAME,
        device: str = config.DEVICE,
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_time: float = 0.0

        print(f"[LocalLLM] Initializing local HF model '{self.model_name}' on device '{self.device}'...")
        start_time = time.time()

        kwargs = {}
        if config.HF_TOKEN:
            kwargs["token"] = config.HF_TOKEN

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs,
        )

        # Set pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )


        if self.device != "cpu":
            self.model.to(self.device)

        self.load_time = time.time() - start_time
        print(f"[LocalLLM] Model loaded successfully in {self.load_time:.2f} seconds.")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = "You are a helpful and precise technical support agent for OrbitDesk.",
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """Generates text response using chat template formatting if supported."""
        start_gen = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        do_sample = temperature > 0.0

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Slice generated tokens past input prompt
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        gen_duration = time.time() - start_gen
        print(f"[LocalLLM] Generation completed in {gen_duration:.2f} seconds.")
        return response_text


# Module-level singleton instance
_global_llm: Optional[LocalLLM] = None


def get_llm() -> LocalLLM:
    global _global_llm
    if _global_llm is None:
        _global_llm = LocalLLM()
    return _global_llm


def generate_response(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Helper function to generate a response from the local LLM."""
    llm = get_llm()
    return llm.generate(prompt=prompt, system_prompt=system_prompt)
