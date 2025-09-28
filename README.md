# Simple Audio Classification with YAMNet

This project creates a basic audio classification system using Google's YAMNet as a feature extractor, following the approach from the GeeksforGeeks article on audio classification.

## Overview

The system consists of two main scripts:

1. **`create_dataset.py`** - Creates a simple audio dataset for training
2. **`train_model.py`** - Trains a classifier using YAMNet features

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create Dataset
```bash
python create_dataset.py
```

This will create a synthetic music dataset with 15 samples for demonstration. The script:
- Generates synthetic audio samples (musical tones)
- Saves them as WAV files
- Creates metadata for training

### 3. Train Model
```bash
python train_model.py
```

This will:
- Load the YAMNet model from TensorFlow Hub
- Extract 1024-dimensional embeddings from each audio sample
- Train a neural network classifier
- Save the trained model and generate training plots

## How It Works

### YAMNet Feature Extraction
Following the GeeksforGeeks approach:

1. **Load YAMNet**: Pre-trained model from TensorFlow Hub
2. **Extract Features**: Get 1024-dimensional embeddings for each audio file
3. **Average Pooling**: Average embeddings across time frames for fixed-size features
4. **Train Classifier**: Use features to train a simple neural network

### Architecture
```
Audio File → YAMNet → Embeddings (1024D) → Neural Network → Classification
```

### Neural Network Structure
- Input: 1024 features (YAMNet embeddings)
- Hidden layers: 512 → 256 → 128 neurons with ReLU activation
- Dropout layers for regularization
- Output: Softmax for multi-class (or sigmoid for binary)

## Dataset Structure

The created dataset follows this structure:
```
synthetic_music_dataset/
├── metadata.json          # Dataset information
└── music/                 # Audio samples
    ├── music_000.wav
    ├── music_001.wav
    └── ...
```

## Output Files

After training, you'll get:
- `audio_classifier_model.h5` - Trained Keras model
- `label_encoder.pkl` - Label encoder for class mapping
- `training_history.png` - Training accuracy/loss plots

## Customization

### Using Real Audio Files
The `create_dataset.py` script can also process real audio files:

1. Place audio files in a directory (e.g., `my_audio/`)
2. Run the script and it will ask if you want to use real files
3. It will process and organize them for training

### Adding More Classes
To add more audio classes:

1. Modify `create_dataset.py` to create different audio types
2. Or organize real audio files into class subdirectories
3. The training script will automatically handle multiple classes

### Example with Multiple Classes
```python
# In create_dataset.py, create multiple types:
create_dataset("dataset", 10, "music")
create_dataset("dataset", 10, "speech") 
create_dataset("dataset", 10, "noise")
```

## Key Features

- **Simple Setup**: Just two scripts to run
- **YAMNet Integration**: Uses Google's pre-trained model
- **Automatic Feature Extraction**: No manual feature engineering
- **Flexible Dataset**: Works with synthetic or real audio
- **Training Visualization**: Plots and metrics
- **Model Persistence**: Save and load trained models

## Technical Details

- **Sample Rate**: 16kHz (required by YAMNet)
- **Feature Dimension**: 1024 (YAMNet embedding size)
- **Audio Duration**: 2.5-3.5 seconds per sample
- **Training**: Early stopping and learning rate reduction
- **Evaluation**: Classification report and confusion matrix

## Limitations

This is a basic demonstration system:
- Single label per audio file
- Limited to short audio clips
- Synthetic dataset for demo purposes
- Simple neural network architecture

## Next Steps

To improve the system:
1. Add more diverse real audio data
2. Implement data augmentation
3. Try different classifier architectures
4. Add support for longer audio files
5. Implement real-time prediction

## Troubleshooting

### Common Issues

1. **TensorFlow/YAMNet loading errors**: 
   - Ensure stable internet connection for model download
   - Check TensorFlow installation

2. **Audio loading errors**:
   - Verify audio file formats are supported
   - Check file paths in metadata

3. **Memory issues**:
   - Reduce dataset size for testing
   - Use smaller batch sizes

### System Requirements

- Python 3.7+
- TensorFlow 2.8+
- At least 2GB RAM
- Internet connection (for YAMNet download)

## References

- [GeeksforGeeks: Audio Classification using Google's YAMNet](https://www.geeksforgeeks.org/deep-learning/audio-classification-using-googles-yamnet/)
- [YAMNet on TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- [AudioSet Dataset](https://research.google.com/audioset/)
