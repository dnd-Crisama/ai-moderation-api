import os, json, gc
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

HF_MODEL_ID  = os.getenv("HF_MODEL_ID", "crisama/toxic-comment-vi-en")
HF_TOKEN     = os.getenv("HF_TOKEN", None)

_tokenizer  = None
_session    = None
_thresholds = {"delete": 0.95, "flag": 0.75}
_max_len    = 128

def load_model():
    global _tokenizer, _session, _thresholds, _max_len

    os.makedirs("/tmp/toxic_model", exist_ok=True)

    # Download từng file cần thiết — KHÔNG download safetensors
    files_needed = [
        "model_quantized.onnx",
        "tokenizer_config.json",
        "vocab.txt",
        "bpe.codes",
        "special_tokens_map.json",
        "added_tokens.json",
        "meta.json",
    ]

    print("[loader] Downloading required files only...")
    local_paths = {}
    for fname in files_needed:
        try:
            path = hf_hub_download(
                repo_id=HF_MODEL_ID,
                filename=fname,
                token=HF_TOKEN,
                local_dir="/tmp/toxic_model",
            )
            local_paths[fname] = path
            size = os.path.getsize(path) / 1024 / 1024
            print(f"[loader]   {fname}: {size:.1f} MB")
        except Exception as e:
            print(f"[loader]   {fname}: skip ({e})")

    # Load tokenizer — PhoBERT dùng BPE, load qua tokenizers library
    tokenizer_cfg_path = "/tmp/toxic_model/tokenizer_config.json"
    with open(tokenizer_cfg_path) as f:
        tcfg = json.load(f)

    # PhoBERT tokenizer: dùng vocab.txt + bpe.codes trực tiếp
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        "/tmp/toxic_model",
        local_files_only=True,
    )

    # Load ONNX session — chỉ tốn ~140MB RAM
    onnx_path = "/tmp/toxic_model/model_quantized.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    print(f"[loader] Loading ONNX ({os.path.getsize(onnx_path)/1024/1024:.0f} MB)...")
    _session = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    gc.collect()

    # Load thresholds
    meta_path = "/tmp/toxic_model/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _thresholds = meta.get("thresholds", _thresholds)
        _max_len    = meta.get("max_length", _max_len)

    print(f"[loader] Ready. thresholds={_thresholds}")

def get_tokenizer():  return _tokenizer
def get_session():    return _session
def get_thresholds(): return _thresholds
def get_max_len():    return _max_len
