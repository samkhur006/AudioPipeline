#!/usr/bin/env python3
"""
YAMNet Dog Classifier Training with MLflow and MinIO Integration
Trains a classification layer on top of YAMNet embeddings with experiment tracking
"""

import os, io, json, csv, time, hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa, soundfile as sf
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import boto3
import matplotlib.pyplot as plt
import tempfile

# ---------- Env / config ----------
EXP_NAME       = os.getenv("EXP_NAME", "yamnet-training")
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MANIFEST_JSON  = os.getenv("MANIFEST_JSON", "metadata.json")  # path or s3://...
CLASS_FILTER   = os.getenv("CLASS_FILTER", "dog")             # simple keyword filter
EPOCHS         = int(os.getenv("EPOCHS", "15"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "8"))
LEARNING_RATE  = float(os.getenv("LEARNING_RATE", "0.001"))
SAVE_MODEL     = os.getenv("SAVE_MODEL", "true").lower() == "true"
DATA_HASH      = os.getenv("DATA_HASH", "unknown")
GIT_SHA        = os.getenv("GIT_SHA", "unknown")
IMAGE_DIGEST   = os.getenv("IMAGE_DIGEST", "unknown")
DATA_BASE_DIR  = os.getenv("DATA_BASE_DIR", "")  # e.g., "/data"

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

def extract_embeddings(yamnet_model, audio_files):
    """Extract YAMNet embeddings from audio files"""
    embeddings_list = []
    labels_list = []
    
    print(f"Extracting embeddings from {len(audio_files)} audio files...")
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load audio
            if audio_file.startswith("s3://"):
                waveform = load_wav(audio_file)
                waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
            else:
                waveform, _ = librosa.load(audio_file, sr=16000, mono=True)
                waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
            
            # Get YAMNet embeddings
            scores, embeddings, spectrogram = yamnet_model(waveform)
            
            # Average embeddings across time to get a single representation per file
            avg_embedding = tf.reduce_mean(embeddings, axis=0)
            embeddings_list.append(avg_embedding.numpy())
            
            # For now, all samples are positive (dog) - label = 1
            labels_list.append(1)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")
                
        except Exception as e:
            print(f"  Error processing {audio_file}: {e}")
            continue
    
    return np.array(embeddings_list), np.array(labels_list)

def create_classifier_model(embedding_dim, learning_rate=0.001):
    """Create a simple classification model on top of YAMNet embeddings"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def create_synthetic_negative_samples(positive_embeddings, num_negative_samples):
    """Create synthetic negative samples by adding noise to positive samples"""
    print(f"Creating {num_negative_samples} synthetic negative samples...")
    
    # Add gaussian noise to positive samples to create negatives
    noise_scale = 0.5
    negative_embeddings = []
    
    for _ in range(num_negative_samples):
        # Randomly select a positive sample
        idx = np.random.randint(0, len(positive_embeddings))
        base_embedding = positive_embeddings[idx]
        
        # Add noise
        noise = np.random.normal(0, noise_scale, base_embedding.shape)
        negative_embedding = base_embedding + noise
        negative_embeddings.append(negative_embedding)
    
    return np.array(negative_embeddings)

def plot_training_curves(history):
    """Create training curve plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    return fig

# ---------- Main ----------
def main():
    # MLflow init
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXP_NAME)
    
    # Generate sequential run name
    run_name = f"training-run-{int(time.time())}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"🚀 Starting MLflow run: {run.info.run_id}")
        print(f"📊 Experiment: {EXP_NAME}")
        
        # Log parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("class_filter", CLASS_FILTER)
        mlflow.log_param("data_hash", DATA_HASH)
        mlflow.log_param("git_sha", GIT_SHA)
        mlflow.log_param("image_digest", IMAGE_DIGEST)
        
        # Load dataset metadata
        manifest_path = resolve_path(MANIFEST_JSON)
        try:
            if manifest_path.startswith("s3://") or os.path.exists(manifest_path):
                meta = json.loads(read_text(manifest_path))
                samples = meta["samples"]
                target_class = meta["dataset_info"].get("target_class", CLASS_FILTER)
                print(f"✅ Loaded manifest with {len(samples)} samples")
                mlflow.log_param("total_samples", len(samples))
                mlflow.log_param("target_class", target_class)
            else:
                # Fallback to local directory scan
                print("⚠️  Manifest not found, scanning local directory...")
                dataset_dir = resolve_path("real_audio_dataset/dog")
                if not os.path.exists(dataset_dir):
                    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
                
                audio_files = []
                for file_path in Path(dataset_dir).glob("*.wav"):
                    audio_files.append(str(file_path))
                
                if len(audio_files) == 0:
                    raise FileNotFoundError("No audio files found!")
                
                samples = [{"file_path": f, "label": CLASS_FILTER} for f in audio_files]
                target_class = CLASS_FILTER
                print(f"✅ Found {len(samples)} local audio files")
                mlflow.log_param("total_samples", len(samples))
                mlflow.log_param("target_class", target_class)
        
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            mlflow.log_param("error", str(e))
            return
        
        # Load YAMNet model
        try:
            yamnet_model = load_yamnet()
            print("✅ YAMNet model loaded successfully")
            mlflow.log_param("yamnet_model", "https://tfhub.dev/google/yamnet/1")
        except Exception as e:
            print(f"❌ Error loading YAMNet: {e}")
            mlflow.log_param("error", str(e))
            return
        
        # Extract file paths
        audio_files = []
        for sample in samples:
            file_path = sample.get("file_path", "")
            if file_path:
                if not file_path.startswith("s3://") and not os.path.isabs(file_path):
                    file_path = resolve_path(file_path)
                audio_files.append(file_path)
        
        print(f"Processing {len(audio_files)} audio files...")
        
        # Extract embeddings
        t_start = time.time()
        embeddings, labels = extract_embeddings(yamnet_model, audio_files)
        extraction_time = time.time() - t_start
        
        if len(embeddings) == 0:
            print("❌ No embeddings extracted!")
            mlflow.log_param("error", "No embeddings extracted")
            return
        
        print(f"✅ Extracted embeddings: {embeddings.shape}")
        print(f"⏱️  Embedding extraction took {extraction_time:.2f}s")
        
        mlflow.log_param("embedding_dim", embeddings.shape[1])
        mlflow.log_param("successful_extractions", len(embeddings))
        mlflow.log_metric("extraction_time_seconds", extraction_time)
        
        # Create synthetic negative samples
        num_negative_samples = len(embeddings)  # Same number as positive
        negative_embeddings = create_synthetic_negative_samples(embeddings, num_negative_samples)
        negative_labels = np.zeros(num_negative_samples)
        
        # Combine positive and negative samples
        all_embeddings = np.vstack([embeddings, negative_embeddings])
        all_labels = np.hstack([labels, negative_labels])
        
        print(f"Total dataset: {len(all_embeddings)} samples ({len(embeddings)} positive, {len(negative_embeddings)} negative)")
        mlflow.log_param("positive_samples", len(embeddings))
        mlflow.log_param("negative_samples", len(negative_embeddings))
        mlflow.log_param("total_training_samples", len(all_embeddings))
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            all_embeddings, all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=all_labels
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        
        # Create model
        embedding_dim = embeddings.shape[1]
        model = create_classifier_model(embedding_dim, LEARNING_RATE)
        
        print(f"\nModel architecture:")
        model.summary()
        
        # Log model architecture
        mlflow.log_param("model_layers", len(model.layers))
        mlflow.log_param("model_params", model.count_params())
        
        # Training callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"\n🚀 Starting training for {EPOCHS} epochs...")
        train_start = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - train_start
        print(f"⏱️  Training took {training_time:.2f}s")
        mlflow.log_metric("training_time_seconds", training_time)
        
        # Final evaluation
        print(f"\n📊 Final Evaluation:")
        train_loss, train_acc, train_prec, train_recall = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_prec, val_recall = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_recall:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_recall:.4f}")
        
        # Log final metrics
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("final_train_accuracy", train_acc)
        mlflow.log_metric("final_train_precision", train_prec)
        mlflow.log_metric("final_train_recall", train_recall)
        mlflow.log_metric("final_val_loss", val_loss)
        mlflow.log_metric("final_val_accuracy", val_acc)
        mlflow.log_metric("final_val_precision", val_prec)
        mlflow.log_metric("final_val_recall", val_recall)
        
        # Log training history metrics
        for epoch, (loss, acc, prec, rec, val_loss, val_acc, val_prec, val_rec) in enumerate(zip(
            history.history['loss'], history.history['accuracy'], 
            history.history['precision'], history.history['recall'],
            history.history['val_loss'], history.history['val_accuracy'],
            history.history['val_precision'], history.history['val_recall']
        )):
            mlflow.log_metric("epoch_train_loss", loss, step=epoch)
            mlflow.log_metric("epoch_train_accuracy", acc, step=epoch)
            mlflow.log_metric("epoch_train_precision", prec, step=epoch)
            mlflow.log_metric("epoch_train_recall", rec, step=epoch)
            mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch_val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("epoch_val_precision", val_prec, step=epoch)
            mlflow.log_metric("epoch_val_recall", val_rec, step=epoch)
        
        # Generate predictions for confusion matrix
        y_val_pred = (model.predict(X_val) > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_val, y_val_pred)
        
        # Log confusion matrix as text
        cm_text = f"Confusion Matrix:\n{cm}"
        print(f"\n{cm_text}")
        mlflow.log_text(cm_text, "training-confusion_matrix.txt")
        
        # Create and log training curves
        try:
            fig = plot_training_curves(history)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, "training-plots/training_curves.png")
                plt.close(fig)
                os.unlink(tmp.name)
            print("✅ Training curves logged to MLflow")
        except Exception as e:
            print(f"⚠️  Could not log training curves: {e}")
        
        # Save and log model
        if SAVE_MODEL:
            try:
                # Save model locally first
                model_path = f"training-dog_classifier_{int(time.time())}.h5"
                model.save(model_path)
                
                # Log model to MLflow
                mlflow.tensorflow.log_model(
                    model, 
                    "training-model",
                    registered_model_name=f"yamnet-dog-classifier"
                )
                
                # Also log the .h5 file as artifact
                mlflow.log_artifact(model_path, "training-models")
                
                print(f"✅ Model saved and logged to MLflow")
                
                # Clean up local file
                if os.path.exists(model_path):
                    os.remove(model_path)
                    
            except Exception as e:
                print(f"⚠️  Could not save/log model: {e}")
                mlflow.log_param("model_save_error", str(e))
        
        # Save training history as artifact
        try:
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'precision': [float(x) for x in history.history['precision']],
                'recall': [float(x) for x in history.history['recall']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'val_precision': [float(x) for x in history.history['val_precision']],
                'val_recall': [float(x) for x in history.history['val_recall']]
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(history_dict, tmp, indent=2)
                tmp.flush()
                mlflow.log_artifact(tmp.name, "training-history.json")
                os.unlink(tmp.name)
            
            print("✅ Training history logged to MLflow")
        except Exception as e:
            print(f"⚠️  Could not log training history: {e}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 MLflow Run ID: {run.info.run_id}")
        print(f"🔗 MLflow UI: {MLFLOW_URI}")

if __name__ == "__main__":
    main()
