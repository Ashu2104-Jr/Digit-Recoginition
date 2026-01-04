# Handwritten Digit Recognition

A neural network-based handwritten digit recognition system with a Tkinter GUI interface.

## Files

- `cptraining.py` - Neural network training script
- `test.py` - GUI application for drawing and recognizing digits
- `weights.npz` - Saved neural network weights (generated after training)

## Requirements

```bash
pip install pillow numpy tkinter
```

## Dataset

Download the MNIST training images:
```bash
wget https://zenodo.org/records/13292895/files/mnist_images.zip?download=1 -O mnist_images.zip
unzip mnist_images.zip
```

## Usage

### 1. Training the Model

```bash
python cptraining.py
```

This will:
- Load training images from `images/training/` directory
- Train a 4-layer neural network (784→392→196→98→10)
- Save trained weights to `weights.npz`
- Show training progress with a progress bar

### 2. Using the GUI

```bash
python test.py
```

Features:
- Draw digits (0-9) on black canvas with white pen
- Automatic cropping with 15px margin
- Resize to 28x28 pixels for recognition
- Real-time digit prediction
- Save drawn images as PNG files

## Neural Network Architecture

- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer 1**: 392 neurons with ReLU activation
- **Hidden Layer 2**: 196 neurons with ReLU activation  
- **Hidden Layer 3**: 98 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation

## Training Details

- **Epochs**: 15
- **Batch Size**: 40
- **Learning Rate**: 0.05
- **Optimizer**: Gradient Descent with Backpropagation
- **Loss Function**: Cross-entropy

## How It Works

1. **Training**: Images are loaded, flattened to 784 pixels, normalized (÷255), and fed to the neural network
2. **Recognition**: Drawn digits are cropped, resized to 28x28, normalized, and classified using trained weights
3. **GUI**: Tkinter canvas allows drawing with mouse, processes image, and displays prediction