# MNIST Digit Classifier (TensorFlow CNN)

This project builds and trains a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits (0–9) from the classic MNIST dataset.

---

## Dataset

- **MNIST**: 70,000 grayscale images (28×28 pixels)
  - 60,000 training images
  - 10,000 test images
- Built into TensorFlow/Keras: no external download needed.

---

## Objective

Predict the correct digit (0–9) from a 28×28 grayscale image using a deep learning model.

---

## Model result

Achieved 98.69% accuracy on the test dataset

---

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py

