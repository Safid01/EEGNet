# EEGNet for EEG Motor Imagery Classification

This repository contains my implementation and experiments with **EEGNet**, a compact convolutional neural network designed for EEG signal classification. The project focuses on motor imagery brain-computer interface (BCI) research using deep learning and transfer learning techniques.

## Overview

The primary goal of this project is to develop an efficient EEG classification pipeline capable of recognizing motor imagery tasks from electroencephalography (EEG) signals. The repository includes data preprocessing, model training, evaluation, hyperparameter optimization, and transfer learning experiments.

## Features

- EEGNet implementation in PyTorch
- EEG signal preprocessing pipeline
- Motor imagery classification
- Subject-wise training and evaluation
- Transfer learning experiments
- Hyperparameter optimization
- Performance evaluation using standard metrics
- Modular training pipeline

## Dataset

Designed for the BCI Competition IV 2a dataset.

- 22 EEG channels
- 4 motor imagery classes
- Left Hand
- Right Hand
- Feet
- Tongue

The preprocessing pipeline can be adapted for other EEG datasets.

## Project Workflow

1. EEG preprocessing
2. Signal filtering and artifact removal
3. Epoch generation
4. Model training
5. Hyperparameter optimization
6. Transfer learning
7. Model evaluation
8. Performance analysis

## Technologies

- Python
- PyTorch
- NumPy
- MNE-Python
- Scikit-learn
- Optuna
- Matplotlib

## Repository Structure

- Data preprocessing scripts
- EEGNet model implementation
- Training pipeline
- Evaluation scripts
- Utility functions
- Saved models
- Experimental notebooks

## Results

This repository is intended for ongoing research and experimentation. Performance depends on preprocessing strategy, subject variability, training configuration, and transfer learning setup.

## Future Improvements

- Attention-enhanced EEGNet
- Domain adaptation
- Cross-subject generalization
- Real-time inference
- ONNX export
- Explainable AI visualizations
- Additional benchmark models

## References

Lawhern, V. J., et al. EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces.

## License

This project is intended for research and educational purposes.

## Author

**Safid Hasan**

Computer Science Student | AI Research | Brain-Computer Interfaces