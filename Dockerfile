FROM python:3.11-slim

WORKDIR /app

# DÒNG NÀY RẤT QUAN TRỌNG: Cài đặt thư viện C++ lõi để onnxruntime có thể chạy
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tải model sẵn vào trong image để tránh quá tải RAM lúc khởi động
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='crisama/toxic-comment-vi-en', local_dir='/app/toxic_model', ignore_patterns=['*.safetensors', '*.bin', 'model.onnx'])"

COPY . .

# Các biến môi trường ép tiết kiệm RAM
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV WEB_CONCURRENCY=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
