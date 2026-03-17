# ai-moderation-api/inference.py
import torch.nn.functional as F
import torch
from model_loader import get_tokenizer, get_model, get_thresholds, get_max_len
# ↑ bỏ get_device — ONNX Runtime tự handle CPU, không cần

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
    # Không cần .to(device) — ONNX Runtime chạy trên CPU tự động

    outputs = model(**inputs)
    logits  = outputs.logits  # shape: [1, 2]

    score = float(F.softmax(logits, dim=-1)[0][1])

    if score >= thresholds["delete"]:
        action = "DELETE"
        label  = "TOXIC"
    elif score >= thresholds["flag"]:
        action = "FLAG"
        label  = "SUSPICIOUS"
    else:
        action = "ALLOW"
        label  = "CLEAN"

    return {"score": round(score, 4), "action": action, "label": label}
