# Cotton Disease Classification with EfficientNetB0 and Vision Transformer (ViT)

## Overview
This project leverages state-of-the-art deep learning techniques to classify cotton plant diseases using images. The solution now includes both EfficientNetB0 (transfer learning) and a custom Vision Transformer (ViT) implementation in PyTorch. Both models distinguish between four classes: **Fusarium wilt, Bacterial blight, Curl virus, and Healthy**.

## Project Structure
```
EfficientNet/
├── 3-EfficientNetB0_for_transfer_learning.ipynb   # EfficientNetB0 transfer learning notebook
├── image_classifier_from_scratch.ipynb            # Vision Transformer (ViT) from scratch (PyTorch)
├── Cotton_Dataset/                                # Image dataset (not tracked by git)
│   ├── train/
│   │   ├── bacterial_blight/
│   │   ├── curl_virus/
│   │   ├── fussarium_wilt/
│   │   └── healthy/
│   └── val/
│       ├── bacterial_blight/
│       ├── curl_virus/
│       ├── fussarium_wilt/
│       └── healthy/
├── requirements.txt                               # Python dependencies
├── .gitignore                                     # Git ignore file
└── README.md                                      # This file
```

## Dataset
- **Location:** `Cotton_Dataset/`
- **Structure:**
  - `train/` and `val/` directories, each containing one subfolder per class.
  - Each subfolder contains images of that class.
- **Classes:**
  - `bacterial_blight`
  - `curl_virus`
  - `fussarium_wilt`
  - `healthy`

> **Note:** The dataset is not tracked by git (see `.gitignore`).

## Models & Approach
### EfficientNetB0 (TensorFlow/Keras)
- **Architecture:** EfficientNetB0 (with transfer learning from ImageNet)
- **Input Size:** 224x224 pixels
- **Workflow:**
  1. Load and preprocess images using Keras' `image_dataset_from_directory`.
  2. Apply data augmentation for better generalization.
  3. Use EfficientNetB0 as a feature extractor (initially frozen).
  4. Add custom classification head for 4 classes.
  5. Train the top layers first, then fine-tune the entire model.
  6. Evaluate using accuracy, confusion matrix, and visualizations.

### Vision Transformer (ViT, PyTorch)
- **Architecture:** Custom Vision Transformer (ViT) implemented in PyTorch
- **Input Size:** 224x224 pixels
- **Workflow:**
  1. Load and preprocess images using PyTorch's `ImageFolder` and `DataLoader`.
  2. Apply data augmentation (random flip, rotation, zoom) using torchvision transforms.
  3. Build a ViT model from scratch (no pre-trained weights).
  4. Train and validate the model, tracking accuracy and loss.
  5. Evaluate with accuracy/loss curves, confusion matrix, and predictions on new images.
  6. Includes a function to predict the class of any image by file path, with probability breakdown and visualization.

## How to Run
### 1. **Setup Environment**
- Install Python 3.10 or 3.9 (TensorFlow does **not** support Python 3.13+).
- Create and activate a virtual environment:
  ```bash
  python -m venv venv
  # On Git Bash/Unix
  source venv/Scripts/activate
  # or on Windows CMD/PowerShell
  venv\Scripts\activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Run the Notebooks**
- Start Jupyter:
  ```bash
  jupyter notebook
  ```
- Open `3-EfficientNetB0_for_transfer_learning.ipynb` for EfficientNetB0, or `image_classifier_from_scratch.ipynb` for ViT, and follow the step-by-step cells.

## Features
- **Transfer Learning (EfficientNetB0):** Leverages pre-trained weights for improved accuracy and faster convergence.
- **Custom ViT (PyTorch):** Implements a Vision Transformer from scratch for comparison.
- **Data Augmentation:** Reduces overfitting and improves model robustness.
- **Fine-Tuning:** Unfreezes base layers for even better performance (EfficientNetB0).
- **Visualization:** Plots training/validation accuracy and loss, confusion matrix, and displays predictions with images.
- **Image Path Prediction (ViT):** Predicts class and probabilities for any given image file.
- **Model Saving/Loading:** Save and reload models for inference or further training.

## Results & Evaluation
- The notebooks provide code to:
  - Plot accuracy/loss curves
  - Evaluate on validation data
  - Display confusion matrix and classification report
  - Predict and visualize results on new images (including single image prediction by path in ViT)

## Tips & Troubleshooting
- **TensorFlow Installation Issues:**
  - Ensure you are using Python 3.10 or 3.9.
  - If you get `No matching distribution found for tensorflow`, downgrade your Python version.
- **Dataset Not Found:**
  - Make sure `Cotton_Dataset` is in the project root and follows the correct structure.
- **Virtual Environment Issues:**
  - Delete and recreate `venv` if you switch Python versions.

## Credits
- Dataset: [Provide source if public]
- Model: [EfficientNetB0](https://arxiv.org/abs/1905.11946), [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)

## License
This project is for educational and research purposes. Please check dataset and model licenses for any restrictions.
