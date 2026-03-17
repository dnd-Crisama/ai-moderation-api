# ai-moderation-api/inference.py
import torch
import torch.nn.functional as F
from model_loader import get_tokenizer, get_model, get_thresholds, get_max_len

def predict(text: str) -> dict:
    tokenizer  = get_tokenizer()
    model      = get_model()
    thresholds = get_thresholds()
    max_len    = get_max_len()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

    outputs = model(**inputs)
    score   = float(F.softmax(outputs.logits, dim=-1)[0][1])

    if score >= thresholds["delete"]:
        action, label = "DELETE", "TOXIC"
    elif score >= thresholds["flag"]:
        action, label = "FLAG", "SUSPICIOUS"
    else:
        action, label = "ALLOW", "CLEAN"

    return {"score": round(score, 4), "action": action, "label": label}
