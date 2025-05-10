from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

FACT_VERIFIER_PATH = "/home/tushard/ondemand/NLP Final Project/qwen2.5-3b-lora-fact-verifier-final"
BASE_MODEL_PATH = "/home/tushard/ondemand/NLP Final Project/qwen2.5-3B"

# Load tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_PATH,
    num_labels=2,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

model = PeftModel.from_pretrained(base_model, FACT_VERIFIER_PATH)
model.eval()

def fact_check(question, answer):
    prompt = f"Q: {question}\nA: {answer}\nIs this answer factually correct?"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    return pred == 1  # 1 = factually correct, 0 = incorrect
