import torch, torch.nn.functional as F
from model_loader import get_tokenizer, get_model, get_thresholds, get_device, get_max_len

def predict(text: str) -> dict:
    tokenizer  = get_tokenizer()
    model      = get_model()
    thresholds = get_thresholds()
    device     = get_device()
    max_len    = get_max_len()

    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    score = float(F.softmax(logits, dim=-1)[0][1])

    if score >= thresholds['delete']:
        action = 'DELETE'
        label  = 'TOXIC'
    elif score >= thresholds['flag']:
        action = 'FLAG'
        label  = 'SUSPICIOUS'
    else:
        action = 'ALLOW'
        label  = 'CLEAN'

    return { 'score': round(score, 4), 'action': action, 'label': label }
