FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# CPU-only torch wheel first (huge size difference vs default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.12.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Everything else comes straight from requirements.txt - single source of truth
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
