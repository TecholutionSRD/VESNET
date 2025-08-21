# VESNET - Vessel Level Detection

A deep learning-based solution for detecting liquid levels in transparent vessels using computer vision.

## Overview

VESNET (Vessel Segmentation Network) is designed to accurately detect and measure liquid levels in transparent containers. It uses a combination of semantic segmentation and computer vision techniques to:

1. Detect vessels in images
2. Identify liquid content
3. Calculate fill levels with percentage accuracy

## Model Architecture

The system uses two main neural network models:

- **Semantic Segmentation Model**: A fully convolutional network with Pyramid Scene Parsing (PSP) based on ResNet101 backbone
- **YOLO Models**: Used for vessel detection and localization

## Pre-trained Models

The repository includes the following pre-trained models (stored using Git LFS):

- `checkpoints/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch` - Main segmentation model
- `checkpoints/yolo11n.pt` - YOLO model for vessel detection
- `checkpoints/yolo11s.pt` - YOLO model variant

## Directory Structure

```
├── checkpoints/          # Pre-trained model weights
├── data/                 # Data directory
│   ├── input_data/      # Input images
│   └── output_data/     # Processing results
├── src/                 # Source code
│   ├── CategoryDictionary.py
│   ├── model.py         # Neural network architecture
│   ├── prepare_kaggle_data.py
│   └── visualize.py
├── main.py             # Main inference script
├── cam.py              # Camera interface
└── live.py            # Real-time processing
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VESNET.git
cd VESNET
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Inference

```bash
# Run on a single image
python main.py --image path/to/image.jpg

# Run on a folder of images
python main.py --folder path/to/folder

# Run on the full dataset
python main.py --dataset
```

### Live Camera Mode

```bash
python live.py
```

## Results

The system outputs:
- Segmentation masks for vessels and liquids
- Calculated fill levels (continuous percentage and nearest discrete level)
- Visualization overlays
- JSON results with detailed measurements

## License

[Add your chosen license here]

## Citation

If you use this work, please cite:

[Add citation information if applicable]

## Acknowledgments

[Add any acknowledgments or credits]
