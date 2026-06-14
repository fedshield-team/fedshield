FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy pandas scikit-learn flwr \
    shap streamlit plotly

COPY . .

EXPOSE 8080 8501

CMD ["python", "server/flower_server.py"]