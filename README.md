# AI-Image-Detection-Explainable-DL
CNN-based AI-generated image detection using Explainable AI (Grad-CAM, LIME, SHAP) with up to 98% accuracy
# AI-Generated Image Detection using Explainable Deep Learning


# Overview

This project focuses on detecting AI-generated images using deep learning models and Explainable AI techniques. The system classifies images as real or synthetic and provides visual explanations for predictions.

# Features

* CNN-based image classification
* Detection of AI-generated vs real images
* Explainability using Grad-CAM, LIME, and SHAP
* Visualization of important regions influencing predictions

# Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas
* Matplotlib

# Model Performance

* NASNet: 98% accuracy
* DenseNet121: 94% accuracy

# How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the model:
   python train.py

# Project Structure

* src/ → source code
* dataset/ → input data (not included)
* results/ → output results
* README.md → project documentation

# Future Improvements

* Real-time detection system
* Web-based deployment
* Improved generalization for new AI models
