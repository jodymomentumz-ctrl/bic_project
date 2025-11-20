# EEG Brain-Inspired Computing Project

Deep learning models for EEG signal classification using CNN, TinyCNN, and EEGNet architectures with optional PCA dimensionality reduction.

## Overview

This project implements multiple neural network architectures for processing and classifying EEG signals:
- **CNN**: Standard convolutional neural network
- **TinyCNN**: Lightweight CNN architecture
- **EEGNet**: Specialized architecture for EEG signal processing
- **Spiking Neural Network**: Brain-inspired spiking CNN with surrogate gradients

## Features

- **Signal Processing**: Bandpass filtering (1-40 Hz) and notch filtering (50 Hz)
- **Dimensionality Reduction**: Optional PCA for channel reduction
- **Flexible Windowing**: Configurable window size and step for temporal segmentation
- **Multiple Models**: Support for various architectures optimized for EEG data
- **PyTorch Implementation**: GPU-accelerated training with modern PyTorch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-inspired-computing.git
cd brain-inspired-computing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- SciPy

## Dataset Structure

The project expects EEG data in CSV format with the following structure:
```
datasets/
  bci_challenge/
    train/
      Data_S02_Sess01.csv
      Data_S02_Sess02.csv
      ...
    TrainLabels.csv
    ChannelsLocation.csv
```

Each CSV file should contain:
- `Time` column: timestamps
- EEG channel columns (e.g., Fz, C3, Cz, C4, Pz, etc.)
- `EOG` column (optional)
- `FeedBackEvent` column (optional)

## Usage

### Training TinyCNN with PCA

```bash
python main.py \
  --data_dir datasets/bci_challenge/train \
  --labels_csv datasets/bci_challenge/TrainLabels.csv \
  --model tinycnn \
  --pca_components 16 \
  --epochs 30 \
  --batch_size 64 \
  --window_sec 2.0 \
  --step_sec 0.5
```

### Training CNN without PCA

```bash
python main.py \
  --data_dir datasets/bci_challenge/train \
  --model cnn \
  --pca_components 0 \
  --epochs 20
```

### Training Spiking Neural Network

```bash
python main.py \
  --data_dir datasets/bci_challenge/train \
  --model spiking \
  --snn_steps 10 \
  --epochs 25
```

### Training EEGNet

```bash
python main.py \
  --data_dir datasets/bci_challenge/train \
  --model eegnet \
  --epochs 30
```

## Command Line Arguments

- `--data_dir`: Directory containing CSV files with EEG data
- `--labels_csv`: Path to labels CSV file (optional)
- `--model`: Model architecture (`cnn`, `tinycnn`, `eegnet`, `spiking`)
- `--pca_components`: Number of PCA components (0 to disable)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--window_sec`: Window size in seconds
- `--step_sec`: Step size for sliding window
- `--bandpass`: Enable bandpass filtering (1-40 Hz)
- `--notch`: Enable notch filtering (50 Hz)
- `--lr`: Learning rate
- `--snn_steps`: Number of simulation steps for spiking model

## Model Architectures

### TinyCNN
Lightweight CNN with:
- Temporal convolution (kernel size 16)
- Spatial convolution
- Pooling and dropout layers

### CNN
Deeper architecture with:
- Multiple convolutional layers
- Batch normalization
- Adaptive pooling

### EEGNet
Specialized for EEG with:
- Depthwise and separable convolutions
- Optimized for temporal and spatial features

### Spiking CNN
Brain-inspired architecture using:
- Leaky Integrate-and-Fire neurons
- Surrogate gradient descent
- Temporal dynamics

## Output

Trained models are saved as:
- `best_model_cnn.pt`
- `best_model_tinycnn.pt`
- `best_model_eegnet.pt`

Training logs include:
- Accuracy and F1 scores
- Classification reports
- Loss curves

## License

This project is available for educational and research purposes.

## Acknowledgments

- BCI Challenge dataset
- PyTorch team
- EEGNet architecture inspiration from Lawhern et al.
