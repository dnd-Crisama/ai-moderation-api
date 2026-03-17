from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os, json, torch

HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_TOKEN    = os.getenv("HF_TOKEN", None)

_tokenizer = None
_model     = None
_thresholds = {"delete": 0.95, "flag": 0.75}

def load_model():
    global _tokenizer, _model, _thresholds
    local_dir = snapshot_download(
        repo_id=HF_MODEL_ID, token=HF_TOKEN, local_dir="/tmp/toxic_model"
    )
    _tokenizer = AutoTokenizer.from_pretrained(local_dir)
    _model = ORTModelForSequenceClassification.from_pretrained(
        local_dir, file_name="model_quantized.onnx"
    )
    meta_path = f"{local_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _thresholds = json.load(f).get("thresholds", _thresholds)
    print(f"✅ ONNX model loaded. Peak RAM: ~150MB")

def get_tokenizer(): return _tokenizer
def get_model():     return _model
def get_thresholds(): return _thresholds
