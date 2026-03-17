# ai-moderation-api/model_loader.py
import os, json, gc
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer          # ← import ở TOP, không trong hàm
from onnxruntime import SessionOptions, GraphOptimizationLevel

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "crisama/toxic-comment-vi-en")
HF_TOKEN     = os.getenv("HF_TOKEN", None)

_tokenizer  = None
_model      = None
_thresholds = {"delete": 0.95, "flag": 0.75}
_max_len    = 128

def load_model():
    global _tokenizer, _model, _thresholds, _max_len

    print(f"[loader] Downloading: {HF_MODEL_ID}")
    local_dir = snapshot_download(
        repo_id=HF_MODEL_ID,
        token=HF_TOKEN,
        local_dir="/tmp/toxic_model",
        ignore_patterns=["*.safetensors", "*.bin", "model.onnx"],
    )

    _tokenizer = AutoTokenizer.from_pretrained(local_dir)

    onnx_path = f"{local_dir}/model_quantized.onnx"
    size_mb   = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"[loader] Loading ONNX ({size_mb:.0f} MB)...")

    sess_opts = SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    _model = ORTModelForSequenceClassification.from_pretrained(
        local_dir,
        file_name="model_quantized.onnx",
        session_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    gc.collect()

    meta_path = f"{local_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _thresholds = meta.get("thresholds", _thresholds)
        _max_len    = meta.get("max_length", _max_len)

    print(f"[loader] Ready. thresholds={_thresholds} max_len={_max_len}")

def get_tokenizer():  return _tokenizer
def get_model():      return _model
def get_thresholds(): return _thresholds
def get_max_len():    return _max_len
