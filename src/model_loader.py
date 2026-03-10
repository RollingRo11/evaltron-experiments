"""Load base model and merge LoRA adapter."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
LORA_ADAPTER = "timhua/wood_v2_sftr4_filt"


def load_model_and_tokenizer():
    """Load base model, merge LoRA, return (model, tokenizer)."""
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter: {LORA_ADAPTER}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)

    print("Merging LoRA weights into base model")
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Model loaded and ready")
    return model, tokenizer
