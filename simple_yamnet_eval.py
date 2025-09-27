#!/usr/bin/env python3
"""
Simple YAMNet Evaluation - No tensorflow_io dependency
Uses librosa for audio processing to avoid compatibility issues
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import csv
import json
import os
from pathlib import Path

def load_yamnet_model():
    """Load YAMNet model from TensorFlow Hub"""
    print("Loading YAMNet model...")
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)
    return yamnet_model

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

def load_audio_file_librosa(file_path):
    """Load audio file using librosa (no tensorflow_io needed)"""
    # Load audio with librosa - automatically handles various formats
    waveform, sample_rate = librosa.load(file_path, sr=16000, mono=True)
    
    # Convert to float32 tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    
    return waveform

def predict_with_yamnet(yamnet_model, waveform, class_names):
    """Make predictions using YAMNet"""
    # Run the model
    scores, embeddings, spectrogram = yamnet_model(waveform)
    
    # Aggregate scores across time
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    # Get top predictions
    top_n = 10
    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    
    predictions = []
    for i, class_index in enumerate(top_indices):
        predictions.append({
            'class_name': class_names[class_index],
            'score': float(mean_scores[class_index]),
            'class_index': int(class_index)
        })
    
    return predictions

def load_dataset():
    """Load the created dataset"""
    dataset_dir = "real_audio_dataset"
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset not found! Run 'python create_dataset.py' first")
        return None
    
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_file):
        print("❌ Metadata file not found!")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def main():
    """Main evaluation function"""
    print("🎵 Simple YAMNet Evaluation (Using librosa)")
    print("=" * 60)
    print("Avoiding tensorflow_io compatibility issues")
    
    # Load YAMNet model
    try:
        yamnet_model = load_yamnet_model()
        print("✅ YAMNet model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YAMNet: {e}")
        return
    
    # Load class names
    print("Loading AudioSet class names...")
    try:
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = class_names_from_csv(class_map_path)
        print(f"✅ Loaded {len(class_names)} class names")
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return
    
    # Find dog-related classes
    dog_related = []
    for i, name in enumerate(class_names):
        if any(word in name.lower() for word in ['dog', 'bark', 'bow-wow', 'yip', 'howl', 'canidae']):
            dog_related.append((i, name))
    
    print(f"\nDog-related classes in AudioSet:")
    for class_idx, class_name in dog_related:
        print(f"  {class_idx:3d}: {class_name}")
    
    # Load dataset
    metadata = load_dataset()
    if metadata is None:
        return
    
    print(f"\nEvaluating {metadata['dataset_info']['total_samples']} audio samples...")
    target_class = metadata['dataset_info']['target_class']
    print(f"Expected class: {target_class}")
    
    print(f"\n" + "="*80)
    
    correct_predictions = 0
    total_samples = 0
    all_results = []
    
    # Process each audio file
    for i, sample in enumerate(metadata['samples']):
        filepath = sample['filepath']
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
        
        filename = os.path.basename(filepath)
        print(f"\n📁 Processing: {filename}")
        
        try:
            # Load audio file with librosa
            waveform = load_audio_file_librosa(filepath)
            print(f"   Audio shape: {waveform.shape}")
            
            # Make predictions
            predictions = predict_with_yamnet(yamnet_model, waveform, class_names)
            
            # Display top predictions
            print(f"   Top 5 predictions:")
            found_dog = False
            
            for j, pred in enumerate(predictions[:5]):
                is_dog = any(word in pred['class_name'].lower() for word in ['dog', 'bark', 'bow-wow', 'yip', 'howl'])
                marker = "🐕" if is_dog else "  "
                print(f"     {j+1}. {marker} {pred['class_name']:<35} ({pred['score']:.4f})")
                
                if j == 0 and is_dog:
                    found_dog = True
            
            # Check if prediction is correct
            if found_dog:
                correct_predictions += 1
                print(f"   ✅ Correctly identified as dog-related!")
            else:
                print(f"   ❌ Not identified as dog-related")
            
            # Store results
            all_results.append({
                'filename': filename,
                'predictions': predictions[:5],
                'correct': found_dog
            })
            
            total_samples += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            continue
    
    # Final results
    print(f"\n" + "="*80)
    print(f"🎉 YAMNet Evaluation Results")
    print(f"="*80)
    print(f"Total samples processed: {total_samples}")
    print(f"Correctly identified: {correct_predictions}")
    
    if total_samples > 0:
        accuracy = correct_predictions / total_samples
        print(f"Accuracy: {accuracy:.1%}")
        
        # Show some statistics
        print(f"\n📊 Model Statistics:")
        print(f"• Total AudioSet classes: {len(class_names)}")
        print(f"• Dog-related classes found: {len(dog_related)}")
        print(f"• Audio processing: librosa (16kHz mono)")
        
        # Save results
        results_file = "yamnet_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'total_samples': total_samples,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'target_class': target_class,
                'dog_related_classes': dog_related,
                'detailed_results': all_results
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    print(f"\n🔍 Technical Details:")
    print(f"• Model: YAMNet from TensorFlow Hub")
    print(f"• Audio processing: librosa (no tensorflow_io)")
    print(f"• Input: 16kHz mono audio")
    print(f"• Output: 521 AudioSet class scores")
    print(f"• Compatibility: Avoids tensorflow_io issues")

if __name__ == "__main__":
    main()
