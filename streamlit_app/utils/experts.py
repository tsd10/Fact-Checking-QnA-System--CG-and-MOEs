import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL_PATH = "/home/tushard/ondemand/NLP Final Project/qwen2.5-3B"
EXPERT_PATHS = {
    "history": "/home/tushard/ondemand/NLP Final Project/qwen2.5-3b-history_finetuned",
    "music": "/home/tushard/ondemand/NLP Final Project/qwen-2.5-3b-final-music",
    "politics": "/home/tushard/ondemand/NLP Final Project/qwen-2.5-3b-final-politics"
}

cache = {}

def unload_all_models():
    for tokenizer_obj, model_obj in cache.values():
        try:
            del model_obj
            del tokenizer_obj
        except Exception:
            pass
    cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def generate_answer(question, domain):
    if domain not in EXPERT_PATHS:
        return f"❌ No expert model available for domain: {domain}"

    if domain not in cache:
        unload_all_models()

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        model = PeftModel.from_pretrained(base_model, EXPERT_PATHS[domain])
        model.eval()
        model.to("cuda")

        cache[domain] = (tokenizer, model)

    tokenizer, model = cache[domain]
    device = next(model.parameters()).device

    question=question+" Answer precisely, in one word if needed "
    inputs = tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            top_p=0.95,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 🔍 Step 1: Remove original question if echoed
    if question.lower() in response.lower():
        response = response.lower().replace(question.lower(), "").strip()

    # 🔍 Step 2: Strip trailing instruction if model echoes full prompt
    if "Answer precisely" in response:
        response = response.split("Answer precisely")[-1].strip()

    # 🔍 Step 3: Keep just the last sentence/phrase
    if "\n" in response:
        response = response.split("\n")[-1].strip()
    elif "." in response:
        response = response.split(".")[-1].strip()

    return response