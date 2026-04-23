# AI-Based Pneumonia Detection System

## 1. Introduction

This project presents an artificial intelligence system for detecting pneumonia from chest X-ray images using deep learning techniques. The system is designed to assist medical professionals by providing preliminary diagnostic support.

---

## 2. Dataset

The dataset used in this project was obtained from Kaggle. It contains chest X-ray images categorized into two classes:

* Normal
* Pneumonia

The dataset was divided into:

* Training set
* Validation set
* Test set

All images were resized to 224×224 pixels.

---

## 3. Data Preprocessing

The following preprocessing steps were applied:

* Resizing images to 224×224
* Normalization (pixel values scaled to [0,1])
* Data augmentation:

  * Rotation
  * Horizontal flipping
  * Zoom

---

## 4. Model Architecture

The model is based on MobileNetV2 using transfer learning.

* Pre-trained on ImageNet
* Final layers modified for binary classification
* Output layer uses sigmoid activation

---

## 5. Model Training

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Epochs: 20 (with Early Stopping)
* Batch Size: 32

Transfer learning was applied followed by partial fine-tuning.

---

## 6. Model Evaluation

The model achieved:

* Test Accuracy: ~94%

Limitations:

* Only accuracy was used
* Sensitivity and specificity were not fully evaluated

---

## 7. Application Development

The model was deployed as a web application using Streamlit.

Features:

* Upload X-ray image
* Display prediction result
* Show confidence score
* Uncertainty zone implemented

---

## 8. System Features

* Real-time prediction
* User-friendly interface
* Online deployment
* AI-assisted diagnosis

---

## 9. Limitations

* Binary classification only
* No support for other lung diseases
* Possible shortcut learning
* Limited dataset

---

## 10. Future Work

* Add multi-class classification
* Improve evaluation metrics
* Implement explainability (Grad-CAM)
* Train on larger and local datasets

---

## 11. Ethical Considerations

This system is intended to assist medical professionals and should not replace clinical judgment.

---

## 12. Conclusion

The project demonstrates the use of deep learning in medical image analysis and highlights the potential of AI in supporting healthcare applications.
