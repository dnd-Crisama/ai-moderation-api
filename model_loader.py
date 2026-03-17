import os, json, torch, threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'crisama/toxic-comment-vi-en')
HF_TOKEN    = os.getenv('HF_TOKEN', None)
MAX_LEN     = 128
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR  = os.getenv("MODEL_DIR", "/tmp/toxic_model")
CACHE_DIR = os.getenv("HF_CACHE", "/tmp/hf_cache")

_tokenizer = None
_model     = None
_thresholds = None
_lock = threading.Lock()

def load_model():
    global _tokenizer, _model, _thresholds

    if _model is not None:
        return

    with _lock:
        if _model is not None:
            return

        print(f'Loading model from HF: {HF_MODEL_ID} on {DEVICE}')

        local_dir = snapshot_download(
            repo_id=HF_MODEL_ID,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
            local_dir=BASE_DIR,
            local_dir_use_symlinks=False
        )

        _tokenizer = AutoTokenizer.from_pretrained(local_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(local_dir)

        _model.to(DEVICE)
        _model.eval()

        # load thresholds
        meta_path = f'{local_dir}/meta.json'
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            _thresholds = meta.get('thresholds', {'delete': 0.95, 'flag': 0.75})
        else:
            _thresholds = {'delete': 0.95, 'flag': 0.75}

        print('Model loaded. Thresholds:', _thresholds)

def get_tokenizer(): return _tokenizer
def get_model(): return _model
def get_thresholds(): return _thresholds
def get_device(): return DEVICE
def get_max_len(): return MAX_LEN
