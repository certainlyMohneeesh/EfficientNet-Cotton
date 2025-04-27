# Cotton Disease Classification with EfficientNetB0

## Overview
This project leverages state-of-the-art deep learning techniques to classify cotton plant diseases using images. The core of the solution is based on the EfficientNetB0 architecture and transfer learning, enabling high accuracy even with limited data. The model distinguishes between four classes: **Fusarium wilt, Bacterial blight, Curl virus, and Healthy**.

## Project Structure
```
EfficientNet/
├── 3-EfficientNetB0_for_transfer_learning.ipynb   # Main notebook for transfer learning
├── 2-efficientnetB0_Custom_dataset.ipynb          # Notebook for training from scratch
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

## Model & Approach
- **Architecture:** EfficientNetB0 (with transfer learning from ImageNet)
- **Input Size:** 224x224 pixels
- **Workflow:**
  1. Load and preprocess images using Keras' `image_dataset_from_directory`.
  2. Apply data augmentation for better generalization.
  3. Use EfficientNetB0 as a feature extractor (initially frozen).
  4. Add custom classification head for 4 classes.
  5. Train the top layers first, then fine-tune the entire model.
  6. Evaluate using accuracy, confusion matrix, and visualizations.

## How to Run
### 1. **Setup Environment**
- Install Python 3.10 or 3.9 (TensorFlow does **not** support Python 3.13+).
- Create and activate a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/Scripts/activate  # On Git Bash/Unix
  # or
  .venv\Scripts\activate        # On Windows CMD/PowerShell
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. **Run the Notebook**
- Start Jupyter:
  ```bash
  jupyter notebook
  ```
- Open `3-EfficientNetB0_for_transfer_learning.ipynb` and follow the step-by-step cells.

## Features
- **Transfer Learning:** Leverages pre-trained weights for improved accuracy and faster convergence.
- **Data Augmentation:** Reduces overfitting and improves model robustness.
- **Fine-Tuning:** Unfreezes base layers for even better performance.
- **Visualization:** Plots training/validation accuracy and loss, confusion matrix, and displays predictions with images.
- **Model Saving/Loading:** Save and reload models for inference or further training.

## Results & Evaluation
- The notebook provides code to:
  - Plot accuracy/loss curves
  - Evaluate on validation data
  - Display confusion matrix and classification report
  - Predict and visualize results on new images

## Tips & Troubleshooting
- **TensorFlow Installation Issues:**
  - Ensure you are using Python 3.10 or 3.9.
  - If you get `No matching distribution found for tensorflow`, downgrade your Python version.
- **Dataset Not Found:**
  - Make sure `Cotton_Dataset` is in the project root and follows the correct structure.
- **Virtual Environment Issues:**
  - Delete and recreate `.venv` if you switch Python versions.

## Credits
- Dataset: [Provide source if public]
- Model: [EfficientNetB0](https://arxiv.org/abs/1905.11946) via TensorFlow/Keras

## License
This project is for educational and research purposes. Please check dataset and model licenses for any restrictions.
