# ai-moderation-api/model_loader.py
import os, json
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "crisama/toxic-comment-vi-en")
HF_TOKEN     = os.getenv("HF_TOKEN", None)

_tokenizer  = None
_model      = None
_thresholds = {"delete": 0.95, "flag": 0.75}
_max_len    = 128

def load_model():
    global _tokenizer, _model, _thresholds, _max_len

    print(f"[model_loader] Downloading: {HF_MODEL_ID}")
    local_dir = snapshot_download(
        repo_id=HF_MODEL_ID,
        token=HF_TOKEN,
        local_dir="/tmp/toxic_model"
    )
    print(f"[model_loader] Downloaded to {local_dir}")

    _tokenizer = AutoTokenizer.from_pretrained(local_dir)

    # Tìm ONNX file — thử quantized trước, fallback sang model.onnx
    for fname in ["model_quantized.onnx", "model.onnx"]:
        if os.path.exists(f"{local_dir}/{fname}"):
            print(f"[model_loader] Loading ONNX: {fname}")
            _model = ORTModelForSequenceClassification.from_pretrained(
                local_dir, file_name=fname
            )
            break
    else:
        raise FileNotFoundError(
            f"No ONNX file found in {local_dir}. "
            "Run Cell 17 in Colab to export model_quantized.onnx first."
        )

    # Load thresholds từ meta.json
    meta_path = f"{local_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _thresholds = meta.get("thresholds", _thresholds)
        _max_len    = meta.get("max_length", _max_len)

    print(f"[model_loader] Ready. thresholds={_thresholds} max_len={_max_len}")

def get_tokenizer():  return _tokenizer
def get_model():      return _model
def get_thresholds(): return _thresholds
def get_max_len():    return _max_len
