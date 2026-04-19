# Emotion Detector

A high-performance, real-time emotion recognition system featuring a robust PyTorch Deep Learning backend and a sophisticated web-based visualization dashboard.

## 📋 Overview

Emotion Detector leverages state-of-the-art Computer Vision and Deep Learning techniques to identify human emotions from live video streams. The project is designed with a modular architecture, separating the high-concurrency inference engine from the modern, responsive user interface.

## 🚀 Key Features

- **Real-time Inference**: Low-latency face detection and emotion classification from webcam feeds.
- **Advanced CNN Architecture**: Custom Convolutional Neural Network optimized for facial expression recognition.
- **Modern Dashboard**: A premium, glassmorphism-inspired web interface with real-time analytics.
- **Comprehensive Training Pipeline**: End-to-end model training script with automated data augmentation and Focal Loss for handling class imbalances.
- **Cross-Platform Support**: Seamlessly runs on local environments via Python or through a web browser.

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV (Open Source Computer Vision Library)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Image Processing**: Pillow (PIL)
- **Utilities**: NumPy, Torchvision

## 📁 System Architecture

```text
emotion-detector/
├── frontend/               # Professional web dashboard
│   ├── index.html          # Application entry point
│   ├── style.css           # Premium styling and layout
│   └── app.js              # Real-time visualization logic
├── model/                  # Core AI components
│   ├── model.py            # Neural Network architecture
│   ├── train.py            # Model training & optimization pipeline
│   └── inference.py        # Local inference engine
├── utils/                  # Data preprocessing and utility functions
├── config.py               # System-wide configuration
├── main.py                 # Multi-mode execution entry point
└── model.pth               # Compiled model weights (Pre-trained)
```

## 🏁 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam accessibility
- (Recommended) CUDA-enabled GPU for optimal training performance

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sarveshkk05/emotion-detector.git
   cd emotion-detector
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision opencv-python pillow numpy
   ```

### Operational Modes

#### Local Inference
To launch the real-time detector with face-tracking overlays:
```bash
python main.py
# Select 'infer' when prompted
```

#### Model Training
To train the model on a custom dataset:
1. Configure `DATA_DIR` in `config.py`.
2. Execute:
   ```bash
   python main.py
   # Select 'train' when prompted
   ```

#### Web Dashboard
To access the modern analytics interface:
1. Navigate to the `frontend` directory.
2. Open `index.html` in an updated web browser.

---

Maintained by [Sarvesh](https://github.com/Sarveshkk05)
