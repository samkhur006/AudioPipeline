#!/usr/bin/env python3
"""
Audio Dataset Creator

This script downloads real-world audio samples for training.
Uses the ESC-50 dataset (Environmental Sound Classification) to get real audio samples.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import requests
import zipfile
import pandas as pd
from urllib.parse import urlparse

def download_esc50_dataset(download_dir="esc50_raw"):
    """
    Download the ESC-50 dataset (Environmental Sound Classification).
    
    Args:
        download_dir: Directory to download the dataset
        
    Returns:
        Path to downloaded dataset
    """
    
    download_path = Path(download_dir)
    download_path.mkdir(exist_ok=True)
    
    # ESC-50 dataset URLs
    dataset_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = download_path / "esc50.zip"
    extract_path = download_path / "ESC-50-master"
    
    print("📥 Downloading ESC-50 dataset...")
    print("This is a real-world environmental sound classification dataset")
    
    # Check if already downloaded
    if extract_path.exists() and (extract_path / "audio").exists():
        print("✅ ESC-50 dataset already exists")
        return extract_path
    
    try:
        # Download the dataset
        print(f"Downloading from: {dataset_url}")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print(f"\n✅ Downloaded {downloaded / (1024*1024):.1f} MB")
        
        # Extract the dataset
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        
        # Clean up zip file
        zip_path.unlink()
        
        print("✅ Dataset extracted successfully")
        return extract_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def load_esc50_samples(dataset_path, target_class="dog", num_samples=15):
    """
    Load samples from ESC-50 dataset for a specific class.
    
    Args:
        dataset_path: Path to ESC-50 dataset
        target_class: Class to extract (e.g., 'dog', 'cat', 'rain', 'sea_waves')
        num_samples: Number of samples to extract
        
    Returns:
        List of audio file paths and metadata
    """
    
    # ESC-50 class mapping (simplified for common classes)
    class_mapping = {
        'dog': 'dog',
        'cat': 'cat', 
        'rain': 'rain',
        'sea_waves': 'sea_waves',
        'wind': 'wind',
        'fire': 'crackling_fire',
        'water': 'water_drops',
        'bird': 'rooster',
        'music': 'piano',
        'speech': 'crying_baby'  # closest to speech in ESC-50
    }
    
    audio_dir = dataset_path / "audio"
    meta_file = dataset_path / "meta" / "esc50.csv"
    
    if not audio_dir.exists() or not meta_file.exists():
        print(f"❌ ESC-50 files not found in {dataset_path}")
        return []
    
    # Load metadata
    try:
        df = pd.read_csv(meta_file)
        print(f"📊 ESC-50 dataset loaded: {len(df)} total samples")
        print(f"Available classes: {sorted(df['category'].unique())}")
        
        # Find target class
        esc_class = class_mapping.get(target_class, target_class)
        class_samples = df[df['category'] == esc_class]
        
        if len(class_samples) == 0:
            print(f"❌ No samples found for class '{target_class}' (ESC class: '{esc_class}')")
            print(f"Available classes: {list(class_mapping.keys())}")
            return []
        
        print(f"✅ Found {len(class_samples)} samples for '{target_class}'")
        
        # Select random samples
        selected_samples = class_samples.sample(n=min(num_samples, len(class_samples)), random_state=42)
        
        sample_info = []
        for _, row in selected_samples.iterrows():
            audio_file = audio_dir / row['filename']
            if audio_file.exists():
                sample_info.append({
                    'filename': row['filename'],
                    'filepath': str(audio_file),
                    'label': target_class,
                    'original_class': row['category'],
                    'esc_category': row['category'],
                    'fold': row['fold']
                })
        
        return sample_info
        
    except Exception as e:
        print(f"❌ Error loading ESC-50 metadata: {e}")
        return []

def create_real_dataset(output_dir="real_audio_dataset", target_class="dog", num_samples=15):
    """
    Create a dataset from real ESC-50 audio samples.
    
    Args:
        output_dir: Directory to save the processed dataset
        target_class: Class to extract from ESC-50
        num_samples: Number of samples to extract
        
    Returns:
        Path to created dataset
    """
    
    print(f"🎵 Creating real audio dataset for class: {target_class}")
    print("=" * 50)
    
    # Download ESC-50 dataset
    esc50_path = download_esc50_dataset()
    if esc50_path is None:
        return None
    
    # Load samples for target class
    sample_info = load_esc50_samples(esc50_path, target_class, num_samples)
    if not sample_info:
        return None
    
    # Create output directory
    dataset_path = Path(output_dir)
    dataset_path.mkdir(exist_ok=True)
    
    # Create subdirectory for the class
    audio_dir = dataset_path / target_class
    audio_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Processing {len(sample_info)} samples...")
    
    processed_samples = []
    
    for i, sample in enumerate(sample_info):
        try:
            # Load and process audio
            original_path = sample['filepath']
            audio, sr = librosa.load(original_path, sr=22050, mono=True)
            
            # Ensure consistent duration (pad or trim to 5 seconds)
            target_length = 5 * 22050  # 5 seconds at 22050 Hz
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
            # Save processed audio
            new_filename = f"{target_class}_{i:03d}.wav"
            new_filepath = audio_dir / new_filename
            
            sf.write(new_filepath, audio, sr)
            
            # Update sample info
            processed_sample = {
                'filename': new_filename,
                'filepath': str(new_filepath),
                'label': target_class,
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'original_file': sample['filename'],
                'esc_category': sample['esc_category'],
                'fold': sample['fold']
            }
            
            processed_samples.append(processed_sample)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(sample_info)} samples")
                
        except Exception as e:
            print(f"  ❌ Error processing {sample['filename']}: {e}")
            continue
    
    if not processed_samples:
        print("❌ No samples were successfully processed!")
        return None
    
    # Save dataset metadata
    metadata = {
        'dataset_info': {
            'total_samples': len(processed_samples),
            'target_class': target_class,
            'sample_rate': 22050,
            'duration_seconds': 5.0,
            'source_dataset': 'ESC-50',
            'created_date': str(np.datetime64('now'))
        },
        'samples': processed_samples
    }
    
    metadata_path = dataset_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Real audio dataset created successfully!")
    print(f"📁 Location: {dataset_path.absolute()}")
    print(f"📊 Samples: {len(processed_samples)}")
    print(f"🏷️  Label: {target_class}")
    print(f"📄 Metadata: {metadata_path}")
    print(f"🎧 Source: ESC-50 Environmental Sound Classification dataset")
    
    return dataset_path


def main():
    """Main function to create dataset."""
    
    print("🎵 Real Audio Dataset Creator")
    print("=" * 50)
    print("This script downloads real-world audio samples from ESC-50 dataset")
    print("ESC-50: Environmental Sound Classification with 50 classes")
    
    # Available classes for single-label classification
    available_classes = {
        'dog': 'Dog barking sounds',
        'cat': 'Cat meowing sounds', 
        'rain': 'Rain and precipitation sounds',
        'sea_waves': 'Ocean wave sounds',
        'wind': 'Wind blowing sounds',
        'fire': 'Crackling fire sounds',
        'water': 'Water drop sounds',
        'bird': 'Bird sounds (rooster)',
        'music': 'Piano music',
        'speech': 'Human vocal sounds (baby crying)'
    }
    
    print(f"\nAvailable audio classes:")
    for class_name, description in available_classes.items():
        print(f"  • {class_name:10s} - {description}")
    
    # Let user choose class or use default
    print(f"\nChoose an audio class for training:")
    selected_class = input(f"Enter class name (default: 'dog'): ").strip().lower()
    
    if not selected_class:
        selected_class = 'dog'
    
    if selected_class not in available_classes:
        print(f"⚠️  Class '{selected_class}' not available. Using 'dog' instead.")
        selected_class = 'dog'
    
    # Choose number of samples
    try:
        num_samples = input(f"Number of samples (default: 15): ").strip()
        num_samples = int(num_samples) if num_samples else 15
        num_samples = max(5, min(num_samples, 40))  # Between 5-40 samples
    except:
        num_samples = 15
    
    print(f"\n🎯 Selected: {selected_class} ({available_classes[selected_class]})")
    print(f"📊 Samples: {num_samples}")
    
    # Create the real dataset
    dataset_path = create_real_dataset(
        output_dir="real_audio_dataset",
        target_class=selected_class,
        num_samples=num_samples
    )
    
    if dataset_path:
        print("\n" + "=" * 50)
        print("✅ Real audio dataset creation complete!")
        print(f"📁 Dataset location: {dataset_path}")
        print(f"🏷️  Class: {selected_class}")
        print(f"📊 Samples: {num_samples}")
        print(f"🎧 Source: ESC-50 Environmental Sound Classification")
        
        print(f"\n🚀 Next steps:")
        print("1. Run: python simple_yamnet_eval.py")
        print("2. The evaluation script will automatically use this real dataset")
        print("3. YAMNet will classify these real audio samples")
        
        # Show sample file info
        metadata_file = dataset_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n📄 Dataset details:")
            print(f"   • Total samples: {metadata['dataset_info']['total_samples']}")
            print(f"   • Sample rate: {metadata['dataset_info']['sample_rate']} Hz")
            print(f"   • Duration: {metadata['dataset_info']['duration_seconds']} seconds each")
            print(f"   • Source: {metadata['dataset_info']['source_dataset']}")
    
    else:
        print("\n❌ Failed to create dataset!")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
