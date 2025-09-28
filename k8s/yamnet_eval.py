#!/usr/bin/env python3
import os, io, json, csv, time, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa, soundfile as sf
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import boto3

# ---------- Env / config ----------
EXP_NAME       = os.getenv("EXP_NAME", "yamnet-eval")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MANIFEST_JSON  = os.getenv("MANIFEST_JSON", "metadata.json")  # path or s3://...
CLASS_FILTER   = os.getenv("CLASS_FILTER", "dog")             # simple keyword filter
TOP_N          = int(os.getenv("TOP_N", "5"))
SAVE_WEIGHTS   = os.getenv("SAVE_WEIGHTS", "false").lower() == "true"
DATA_HASH      = os.getenv("DATA_HASH", "unknown")
GIT_SHA        = os.getenv("GIT_SHA", "unknown")
IMAGE_DIGEST   = os.getenv("IMAGE_DIGEST", "unknown")
DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "")  # e.g., "/data"

# S3/MinIO
AWS_ENDPOINT   = os.getenv("AWS_ENDPOINT_URL")  # set for MinIO; omit for AWS S3


def resolve_path(p: str) -> str:
    if p.startswith("s3://") or p.startswith("http"):
        return p
    if os.path.isabs(p):
        return p
    return str(Path(DATA_BASE_DIR) / p) if DATA_BASE_DIR else p

    
# ---------- Helpers ----------
def s3_client():
    if AWS_ENDPOINT:
        return boto3.client("s3", endpoint_url=AWS_ENDPOINT)
    return boto3.client("s3")

def read_text(uri):
    if uri.startswith("s3://"):
        s3 = s3_client()
        bucket, key = uri[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    return Path(uri).read_text()

def load_wav(uri):
    """Return mono 16k float32 waveform"""
    if uri.startswith("s3://"):
        s3 = s3_client()
        bucket, key = uri[5:].split("/", 1)
        bio = io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        # soundfile can read file-like
        y, sr = sf.read(bio, always_2d=False)
    else:
        y, sr = sf.read(uri, always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y.astype(np.float32)

def load_yamnet():
    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    return hub.load(yamnet_model_handle)

def class_names_from_csv(csv_path):
    names = []
    with tf.io.gfile.GFile(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"])
    return names

def predict_topn(model, waveform, class_names, top_n=5):
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_idx = np.argsort(mean_scores)[::-1][:top_n]
    preds = [{"class_name": class_names[i], "score": float(mean_scores[i]), "class_index": int(i)} for i in top_idx]
    return preds, mean_scores

def wrap_savedmodel_for_logging(model):
    # Export a simple SavedModel with waveform -> scores for logging
    class YamnetModule(tf.Module):
        def __init__(self, m): super().__init__(); self.m = m
        @tf.function(input_signature=[tf.TensorSpec([None], tf.float32, name="waveform")])
        def __call__(self, waveform):
            scores, embeddings, spectrogram = self.m(waveform)
            return {"scores": scores}
    return YamnetModule(model)

# ---------- Main ----------
def main():
    # MLflow init
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXP_NAME)

    # Load manifest
    meta = json.loads(read_text(MANIFEST_JSON))
    samples = meta["samples"]
    target_class = meta["dataset_info"].get("target_class", CLASS_FILTER)

    # Load YAMNet + classes
    yamnet = load_yamnet()
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    class_names = class_names_from_csv(class_map_path)

    # Eval loop
    total, correct = 0, 0
    y_true, y_pred = [], []
    results = []

    t_start = time.time()
    
    # Generate sequential run name
    run_name = f"run-{int(time.time())}-{os.getenv('HOSTNAME', 'local')[:8]}"
    print(f"Starting MLflow run: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        # Provenance
        mlflow.set_tags({
            "git_sha": GIT_SHA,
            "image_digest": IMAGE_DIGEST,
            "data_hash": DATA_HASH,
            "model": "YAMNet",
            "class_filter": target_class
        })
        mlflow.log_param("top_n", TOP_N)

        print(f"Processing {len(samples)} audio samples...")
        
        for i, s in enumerate(samples, 1):
            uri = s["filepath"]
            label = 1  # since your dataset builder created one-class "dog" dataset
            try:
                uri = resolve_path(s["filepath"])
                print(f"[{i}/{len(samples)}] Processing: {Path(uri).name}")
                print(f"  Full path: {uri}")
                print(f"  File exists: {os.path.exists(uri)}")
                
                wf = load_wav(uri)
                preds, _ = predict_topn(yamnet, tf.convert_to_tensor(wf, tf.float32), class_names, top_n=TOP_N)
                
                # Print top predictions
                print(f"  Top {TOP_N} predictions:")
                for j, pred in enumerate(preds[:3], 1):  # Show top 3
                    print(f"    {j}. {pred['class_name']}: {pred['score']:.4f}")
                
                is_dog_top1 = any(k in preds[0]["class_name"].lower() for k in ["dog", "bark", "yip", "howl", "canidae"])
                pred = 1 if is_dog_top1 else 0
                
                print(f"  Prediction: {'✅ DOG' if pred == 1 else '❌ NOT DOG'} (Expected: DOG)")
                print(f"  Correct: {'✅' if pred == label else '❌'}")
                print()
                
                y_true.append(label); y_pred.append(pred)
                correct += int(pred == label)
                total += 1
                results.append({"filepath": uri, "top1": preds[0], "top_predictions": preds, "correct": pred==label})
                
            except Exception as e:
                print(f"  ❌ ERROR: {str(e)}")
                results.append({"filepath": uri, "error": str(e)})

        # Metrics
        acc = correct / total if total else 0.0
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        
        # Print summary
        print("=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {pr:.4f}")
        print(f"Recall: {rc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Confusion Matrix: {cm.tolist()}")
        print("=" * 50)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", float(pr))
        mlflow.log_metric("recall", float(rc))
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("count", int(total))
        mlflow.log_param("target_class", target_class)

        # Artifacts
        out_json = "yamnet_evaluation_results.json"
        with open(out_json, "w") as f:
            json.dump({
                "total": total,
                "correct": correct,
                "accuracy": acc,
                "results": results,
                "confusion_matrix": cm.tolist()
            }, f, indent=2)
        mlflow.log_artifact(out_json)

        # Optional: log model snapshot
        if SAVE_WEIGHTS:
            export_dir = "yamnet_savedmodel"
            tf.saved_model.save(wrap_savedmodel_for_logging(yamnet), export_dir)
            # simplest: ship as artifacts (MLflow Model optional)
            mlflow.log_artifacts(export_dir, artifact_path="model")

        elapsed = time.time() - t_start
        mlflow.log_metric("elapsed_sec", elapsed)

if __name__ == "__main__":
    main()
