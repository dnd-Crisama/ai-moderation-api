FROM python:3.11-slim

WORKDIR /app

# Cài đặt thư viện C++ OpenMP bắt buộc cho onnxruntime
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Các biến môi trường ép tiết kiệm RAM
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV WEB_CONCURRENCY=1

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
