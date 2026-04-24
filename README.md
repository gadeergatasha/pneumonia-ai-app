# PneumoRay

## Deep Learning-Based Detection of Pneumonia in Pediatric Chest Radiographs  
### A Validation Study Using Cases Collected from Hebron City

PneumoRay is an interactive medical imaging application developed to detect pneumonia from pediatric chest X-ray images using deep learning. The system is designed as a graduation project in Medical Imaging and provides both classification results and Grad-CAM visual explanations to support interpretation.

## Features

- Detection of pneumonia from pediatric chest X-ray images
- Interactive web interface built with Streamlit
- Deep learning classification using MobileNetV2
- Grad-CAM visualization for model interpretability
- Uncertainty zone to reduce overconfident misclassification

## Model Information

- Model architecture: `MobileNetV2`
- Input image size: `224 x 224`
- Classification classes:
  - `NORMAL`
  - `PNEUMONIA`
- Image preprocessing: pixel normalization using `1/255`
- Grad-CAM layer: `out_relu`

## Decision Logic

The application uses three decision zones:

- `Pneumonia` if the prediction score is `>= 0.60`
- `Normal` if the prediction score is `<= 0.40`
- `Uncertain` if the prediction score is between `0.40` and `0.60`

This uncertainty zone helps reduce incorrect confident predictions in unclear or borderline cases.

## Project Team

- Ghadeer Ahmad Ghatasha
- Rasha Nayef Almashni
- Rana Fakhri Shalalda

## Supervisor

Dr. Bassam Arqoub

## Year

2026

## Installation

Install the required packages:

```bash
pip install -r requirements.txt