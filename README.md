# Intraoperative Hypotension Prediction

This repository contains the code for predicting hypotension based on physiological monitoring data using machine learning techniques. The system processes data from the VitalDB dataset and trains models for classification and regression.

## Workflow Overview
1. **Download Data**: Use `download.py` to fetch data from the VitalDB.
2. **Preprocess Data**: Use `preprocessing.py` to transform raw data into a format suitable for model training.
3. **Train Models**: Use `CNN.py` or `Transformer.py` to train deep learning models for hypotension prediction.

## Setup

### Prerequisites
- Python 3.8 or higher
- Required libraries (install using `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Files and Directories
- `download.py`: Script to download data from VitalDB.
- `preprocessing.py`: Script to preprocess raw data for model input.
- `CNN.py`: Code to train a Convolutional Neural Network model.
- `Transformer.py`: Code to train a Transformer-based model.
- `converted/`: Directory to store downloaded data.
- `processed/`: Directory to store processed data.
- `model/`: Directory to save trained model checkpoints.
- `curve/`: Directory to save result figures.

## Instructions

### 1. Download VitalDB Data
Run the following command to download data:
```bash
python download.py
```
The data will be saved in the `converted` directory.

### 2. Preprocess Data
Process the downloaded data into a model-friendly format:
```bash
python preprocessing.py
```
Preprocessed data will be stored in the `processed` directory.

### 3. Train Models
Choose a model to train:

#### Train CNN Model
```bash
python CNN.py
```

#### Train Transformer Model
```bash
python Transformer.py
```

Trained models will be saved in the `model` directory.


## References
- [VitalDB Dataset](https://vitaldb.net)

