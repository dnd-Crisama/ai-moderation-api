FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải model sẵn trong lúc build image thay vì tải lúc runtime
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='crisama/toxic-comment-vi-en', local_dir='/app/toxic_model', ignore_patterns=['*.safetensors', '*.bin', 'model.onnx'])"

COPY . .

# Set biến môi trường để tối ưu RAM luôn trong Docker
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
