
# YAMNet Audio Classification MLOps Pipeline

A complete MLOps pipeline for audio classification using Google's YAMNet model, featuring Kubernetes-based training, evaluation, and inference with MLflow experiment tracking.

## 🏗️ Architecture Overview

The pipeline implements a comprehensive MLOps workflow with the following components:

```
                         ┌──────────────────────────────────────────┐
                         │              Data Ingestion              │
                         │  • Raw audio (WAV) from bucket/DB       │
  External Sources       │  • Labels / manifests (CSV/Parquet)     │
  ─────────────────▶     │                                         │
                         └───────────────┬──────────────────────────┘
                                         │ standardized datasets
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         Experimentation & Tracking                          │
│                                                                            │
│  ┌───────────────┐     params/metrics/artifacts     ┌───────────────────┐  │
│  │  Trainers /   │ ───────────────────────────────▶ │ MLflow Tracking   │  │
│  │  Evaluators   │                                 │ Server (UI + API) │  │
│  │  (K8s Jobs)   │ ◀─────────────────────────────── │ Postgres (runs)   │  │
│  └──────┬────────┘         model URIs / versions    └─────────┬─────────┘  │
│         │                                               links  │            │
│         │ artifacts (models, plots, logs)                    ┌─▼──────────┐ │
│         └──────────────────────────────────────────────────▶ │ MinIO/S3   │ │
│                                                              │ (Artifacts │ │
│                                                              │  & Models) │ │
│                                                              └────┬───────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                         │ chosen model (URI / version / stage)
                                         ▼
                           ┌──────────────────────────────────────┐
                           │          Serving Endpoints           │
                           │  • Custom YAMNet Inference Server    │
                           │                                      │
                           │  • Exposes /predict (+ /metrics)     │
                           └───────────────┬──────────────────────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │  Clients / Integrations│
                               │  • Batch jobs          │
                               │  • Apps / APIs         │
                               └───────────────────────┘
```

## 🚨 Known Issues

### ARM Architecture Compatibility

This project encountered two major hardware-related issues on ARM64 (M1/M2 Mac) systems:

1. **TensorFlow Compatibility Issue**: TensorFlow has compatibility issues with Mac  
   [Reference: TensorFlow crashes on macOS](https://stackoverflow.com/questions/79744362/import-tensorflow-statement-crashes-or-hangs-on-macos)
2. **TensorFlow Serving ARM Incompatibility**: TensorFlow Docker Runtime images are not compatible with ARM64
   [Reference: TensorFlow Serving incompatible with CPUs without AVX](https://discuss.pytorch.org/t/illegal-instruction-core-dumped-when-importing-pytorch/196950)

**Quick test demonstrating the issue:**
```bash
docker pull tensorflow/tensorflow:latest
docker run -it --rm tensorflow/tensorflow:latest python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

**Error:**
```
The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no specific platform was requested




Steps for repproducing are following:



Step 0: Create the datasets using create_dataset.py

----------------------------------------------------
#### Step 1: Start Minikube Cluster

minikube start --cpus=4 --memory=8g \
  --kubernetes-version=v1.30.6 \
  --wait=apiserver,system_pods \
  --wait-timeout=8m
kubectl create ns mlops-train
```

#### Step 2: Deploy Secrets
```bash
kubectl apply -f k8s/secrets/minio.yaml   
kubectl apply -f k8s/secrets/mlflow_secrets.yaml   
```

**Verify secrets:**
```bash
kubectl -n mlops-train get secrets
```
```
NAME           TYPE     DATA   AGE
minio-secret   Opaque   2      6s
pg-secret      Opaque   3      78s
```

#### Step 3: Deploy PostgreSQL
```bash
kubectl apply -f k8s/postgres/postgres_service.yaml
kubectl apply -f k8s/postgres/postgres_db.yaml
```

**Verify PostgreSQL deployment:**
```bash
kubectl get services -n mlops-train
kubectl get pvc -n mlops-train
kubectl get pods -n mlops-train
```

#### Step 4: Deploy MinIO
```bash
kubectl apply -f k8s/minio/minio_service.yaml 
kubectl apply -f k8s/minio/minio.yaml
```

**Verify MinIO deployment:**
```bash
kubectl get sts -n mlops-train
```

#### Step 5: Deploy MLflow
```bash
kubectl apply -f k8s/mlflow/mlflow_service.yaml
kubectl apply -f k8s/mlflow/mlflow.yaml
```

**Verify MLflow deployment:**
```bash
kubectl get deployments -n mlops-train
```

#### Step 6: Build Evaluation Image
```bash
eval $(minikube -p minikube docker-env)
docker build -t yamnet-eval:dev ./k8s/eval
```

#### Step 7: Mount Dataset
```bash
DATASET_DIR="$(pwd)/real_audio_dataset"  
minikube mount "${DATASET_DIR}":/mnt/datasets/real_audio_dataset
```

#### Step 8: Create MinIO Bucket
```bash
kubectl port-forward -n mlops-train svc/minio 9001:9001 &
```
Create bucket "mlflow-artifacts" from http://localhost:9001

#### Step 9: Run Evaluation Job
```bash
kubectl -n mlops-train apply -f k8s/eval/eval_job.yaml
kubectl get jobs -n mlops-train
```

#### Step 10: Access MLflow UI
```bash
kubectl port-forward -n mlops-train svc/mlflow 5000:5000 &
```
Access metrics at: http://localhost:5000/

All the metrics were correctly recorded here! (Accuracy, f1, recall etc)


----------------------------------------------------
#### Step 11: Get models  from minio
> kubectl -n mlops-train port-forward svc/minio 9000:9000 9001:9001

The model is indeed found in the path:
mlflow-artifacts/<experiment_id>/<run_id>/artifacts/...

where experiment_id and run_id are found in the mlflow UI


----------------------------------------------------
#### Step 12: Inferene . Set up serving runtime

> kubectl -n mlops-train apply -f inference/tf_runtime.yaml

> kubectl -n mlops-train get servingruntime
NAME            DISABLED   MODELTYPE    CONTAINERS   AGE
tfserving-cpu              tensorflow   tfserving    37s


----------------------------------------------------
#### Step 13:  Inference

> kubectl -n mlops-train apply -f inference/inference_service.yaml




There is an issue with the current setup:-
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     CrashLoopBackOff   1 (4s ago)     31s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Error              2 (18s ago)    45s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     CrashLoopBackOff   2 (7s ago)     51s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Error              3 (30s ago)    74s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     CrashLoopBackOff   3 (1s ago)     75s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Terminating        3 (20s ago)    94s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Terminating        3              94s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Terminating        3              94s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Terminating        3              95s
yamnet-tf-predictor-649dc6bcb4-t88jc   0/1     Terminating        3              95s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Pending            0              0s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Pending            0              0s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Init:0/1           0              0s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Init:0/1           0              2s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     PodInitializing    0              6s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              0              14s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              1 (2s ago)     15s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     CrashLoopBackOff   1 (7s ago)     21s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              2 (20s ago)    34s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     CrashLoopBackOff   2 (8s ago)     41s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              3 (21s ago)    54s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     CrashLoopBackOff   3 (2s ago)     56s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              4 (50s ago)    104s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     CrashLoopBackOff   4 (8s ago)     111s
yamnet-tf-predictor-6c5768869b-78sbn   0/1     Error              5 (84s ago)    3m7s


On more debigging found the following reason:

I could not get the inference to work properly as the current tensorflow image is not compatible with the arm64.




 > docker run -it --rm tensorflow/tensorflow:latest python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

 The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no specific platform was requested