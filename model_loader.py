import os, json, gc
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import onnxruntime as ort

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "crisama/toxic-comment-vi-en")
HF_TOKEN     = os.getenv("HF_TOKEN", None)

_tokenizer  = None
_session    = None
_thresholds = {"delete": 0.95, "flag": 0.75}
_max_len    = 128

def load_model():
    global _tokenizer, _session, _thresholds, _max_len

    print(f"[loader] Downloading/Loading from cache: {HF_MODEL_ID}")
    local_dir = snapshot_download(
        repo_id=HF_MODEL_ID,
        token=HF_TOKEN,
        local_dir="/app/toxic_model",
        ignore_patterns=["*.safetensors", "*.bin", "model.onnx"],
    )

    _tokenizer = AutoTokenizer.from_pretrained(local_dir)

    onnx_path = f"{local_dir}/model_quantized.onnx"
    
    # Cấu hình ONNX tiết kiệm RAM nhất có thể
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Load session trực tiếp (không dùng optimum)
    _session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    gc.collect() # Ép dọn dẹp RAM rác ngay lập tức

    meta_path = f"{local_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _thresholds = meta.get("thresholds", _thresholds)
        _max_len    = meta.get("max_length", _max_len)

    print("[loader] Ready.")

def get_tokenizer():  return _tokenizer
def get_session():    return _session
def get_thresholds(): return _thresholds
def get_max_len():    return _max_len
