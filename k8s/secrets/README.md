
DAY 1: 

Mostly the day went well. Got the data conversion and eval scripts to work. THen started setting up the cluster. Came up with the following design:
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
│  │  Evaluators   │                               │ Server (UI + API) │  │
│  │  (K8s Jobs)   │ ◀─────────────────────────────── │ Postgres (runs)   │  │
│  └──────┬────────┘         model URIs / versions    └─────────┬─────────┘  │
│         │                                               links  │            │
│         │ artifacts (models, plots, logs)                    ┌─▼──────────┐ │
│         └──────────────────────────────────────────────────▶ │ MinIO/S3   │ │
│                                                              │ (Artifacts │ │
│                                                              │  & Models) │ │
│                                                              └────┬───────┘ │
│                                                                   │          │
│                                                 model versioning  │          │
│                                                              ┌────▼───────┐ │
│                                                              │ MLflow     │ │
│                                                              │ Model Reg. │ │
│                                                              └────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                         │ chosen model (URI / version / stage)
                                         ▼
                           ┌──────────────────────────────────────┐
                           │          Serving Endpoints           │
                           │  • KServe InferenceService           │
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


Everything went well but ran into the following 2 hardware related issues:

1. Lateset tensorflow compatibility issue with mac. 
More details: 

Wasted an hour on this.

2. Tensorflow imnage incompatible with my arm64 mac.
Quick test for it:
> docker pull tensorflow/tensorflow:latest

latest: Pulling from tensorflow/tensorflow
a3be5d4ce401: Pull complete 
0856fa628903: Pull complete 
e51521d48ce2: Pull complete 
a5299857df41: Pull complete 
382d78987601: Pull complete 
20e7013a1bbe: Pull complete 
ef599a2c102b: Pull complete 
8c7d24d8652c: Pull complete 
847d743bea26: Pull complete 
32fde7d864fe: Pull complete 
3525200428d9: Pull complete 
ff44b2a62492: Pull complete 
Digest: sha256:6b0a7db409e62b7cf188ce72074c520af3e8c21f6d4d5b206f700520594dfb5a
Status: Downloaded newer image for tensorflow/tensorflow:latest
> docker.io/tensorflow/tensorflow:latest
> docker run -it --rm tensorflow/tensorflow:latest python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

 The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no specific platform was requested




Steps for repproducing are following:



Step 0: Create the datasets using create_dataset.py

----------------------------------------------------
Step 1:

minikube start --cpus=4 --memory=8g \
  --kubernetes-version=v1.30.6 \
  --wait=apiserver,system_pods \
  --wait-timeout=8m
kubectl create ns mlops-train

----------------------------------------------------
Step 2:

Created the secrets

kubectl apply -f minio.yaml   
kubectl apply -f mlflow_secrets.yaml   

> kubectl -n mlops-train get secrets  

NAME           TYPE     DATA   AGE
minio-secret   Opaque   2      6s
pg-secret      Opaque   3      78s



----------------------------------------------------
Step 3:
kubectl apply -f postgress_service.yaml
> kubectl get services -n mlops-train                                                       
NAME          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
postgres      ClusterIP   10.110.131.157   <none>        5432/TCP   17m
postgres-hl   ClusterIP   None             <none>        5432/TCP   17m



kubectl apply -f postgress_db.yaml

> kubectl get pvc -n mlops-train                                               
NAME                STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   VOLUMEATTRIBUTESCLASS   AGE
pgdata-postgres-0   Bound    pvc-1970064a-ce5f-4068-8ecd-ef3925dda754   2Gi        RWO            standard       <unset>                 3m24s

> kubectl get pods -n mlops-train                                                              
NAME         READY   STATUS    RESTARTS   AGE
postgres-0   1/1     Running   0          3m50s





----------------------------------------------------
Step 4:  Minio
kubectl apply -f minio_service.yaml 

> kubectl get services -n mlops-train
NAME          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
minio         ClusterIP   10.110.187.26    <none>        9000/TCP,9001/TCP   13s
postgres      ClusterIP   10.110.131.157   <none>        5432/TCP            60m
postgres-hl   ClusterIP   None             <none>        5432/TCP            60m

kubectl apply -f minio.yaml

> kubectl get sts -n mlops-train     
NAME       READY   AGE
minio      1/1     7s
postgres   1/1     47m


----------------------------------------------------
Step 5:  MLflow

kubectl apply -f mlflow_service.yaml

> kubectl get services -n mlops-train
NAME          TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
mlflow        ClusterIP   10.110.131.157   <none>        5432/TCP   17m
postgres      ClusterIP   10.110.131.157   <none>        5432/TCP   60m
postgres-hl   ClusterIP   None             <none>        5432/TCP   60m




kubectl apply -f mlflow.yaml

> kubectl get deployments -n mlops-train
NAME     READY   UP-TO-DATE   AVAILABLE   AGE
mlflow   1/1     1            1           16s


----------------------------------------------------
Step 6: Build the eval image with docker file

eval $(minikube -p minikube docker-env)
docker build -t yamnet-eval:dev ./eval


> docker images
REPOSITORY                                TAG                            IMAGE ID       CREATED          SIZE
yamnet-eval                               dev                            9b8c585c79a0   13 seconds ago   2.64GB


----------------------------------------------------
Step 7: 

DATASET_DIR="$(pwd)/real_audio_dataset"  
minikube mount "${DATASET_DIR}":/mnt/datasets/real_audio_dataset


> minikube ssh -- ls -lah /mnt/datasets/real_audio_dataset | head -n 20

total 5.5K
drwxr-xr-x 2 docker docker  544 Sep 27 23:16 dog
-rw-r--r-- 1 docker docker 4.1K Sep 27 23:08 metadata.json


----------------------------------------------------
Step 8:  Create minio bucket
kubectl port-forward -n mlops-train svc/minio 9001:9001 &
Created bucket - "mlflow-artifacts" from http://localhost:9001


----------------------------------------------------
Step 9: 
> kubectl -n mlops-train apply -f eval_job.yaml
> kubectl get jobs -n mlops-train

NAME                      STATUS     COMPLETIONS   DURATION   AGE
yamnet-eval-localfs-001   Complete   1/1           9s         35s



----------------------------------------------------
Step 10: Get MLflow UI
> kubectl port-forward -n mlops-train svc/mlflow 5000:5000 &

Get the metrics from:
http://localhost:5000/

All the metrics were correctly recorded here! (Accuracy, f1, recall etc)


----------------------------------------------------
Step 11: Get models  from minio
> kubectl -n mlops-train port-forward svc/minio 9000:9000 9001:9001

The model is indeed found in the path:
mlflow-artifacts/<experiment_id>/<run_id>/artifacts/...

where experiment_id and run_id are found in the mlflow UI


----------------------------------------------------
Step 12: Inferene . Set up serving runtime

> kubectl -n mlops-train apply -f inference/tf_runtime.yaml

> kubectl -n mlops-train get servingruntime
NAME            DISABLED   MODELTYPE    CONTAINERS   AGE
tfserving-cpu              tensorflow   tfserving    37s


----------------------------------------------------
Step 13: 
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