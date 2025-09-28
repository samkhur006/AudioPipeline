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