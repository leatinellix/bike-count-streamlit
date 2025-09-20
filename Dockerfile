FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY src ./src
EXPOSE 8501
CMD ["streamlit","run","app/app.py","--server.address=0.0.0.0","--server.port=8501"]
