import numpy as np
# 1. Đổi get_session thành get_model
from model_loader import get_tokenizer, get_session, get_model, get_thresholds, get_max_len

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predict(text: str) -> dict:
    tokenizer  = get_tokenizer()
    model      = get_model() # 2. Gọi get_model() thay vì get_session()
    thresholds = get_thresholds()
    max_len    = get_max_len()

    encoded = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_len,
    )

    inputs = {
        "input_ids":      encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64),
    }

    # 3. Lấy raw ONNX session từ object ORTModel thông qua thuộc tính .model
    outputs = model.model.run(None, inputs) 
    logits  = outputs[0][0]  # shape: [2]
    probs   = softmax(logits)
    score   = float(probs[1])

    if score >= thresholds["delete"]:
        action, label = "DELETE", "TOXIC"
    elif score >= thresholds["flag"]:
        action, label = "FLAG", "SUSPICIOUS"
    else:
        action, label = "ALLOW", "CLEAN"

    return {"score": round(score, 4), "action": action, "label": label}
