from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

DOMAIN_MAP = {0: "history", 1: "music", 2: "politics"}

base_model_path = "/home/tushard/ondemand/NLP Final Project/qwen2.5-7B"
adapter_path = "/home/tushard/ondemand/NLP Final Project/qwen2.5-7b-router-final"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# === Load router model on CPU ===
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_path,
    num_labels=3,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# === Apply LoRA adapter and move fully to CPU ===
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to("cpu")  # ✅ force full model and adapter to CPU
model.eval()

def route_question(question):
    #question=question+" Answer precisely in one sentence"
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # ✅ match CPU model

    print({k: v.device for k, v in inputs.items()})  # sanity check

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    return DOMAIN_MAP[pred]
