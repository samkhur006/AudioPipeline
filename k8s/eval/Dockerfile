FROM python:3.11-slim

# system deps 
RUN apt-get update && apt-get install -y git && apt-get install -y build-essential libsndfile1 && rm -rf /var/lib/apt/lists/*

# python deps
RUN pip install --no-cache-dir \
    mlflow==2.14.1 \
    tensorflow==2.15.0 \
    tensorflow_hub==0.16.1 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    boto3==1.34.162

WORKDIR /app
COPY yamnet_eval.py . 
ENTRYPOINT ["python", "yamnet_eval.py"]
