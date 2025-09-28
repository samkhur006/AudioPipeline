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