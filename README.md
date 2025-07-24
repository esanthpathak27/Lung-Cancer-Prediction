# Lung Cancer Stages Prediction using CNN

This is a GUI-based Python application built using **Tkinter** that allows users to predict the stages of lung cancer from medical images using a **Convolutional Neural Network (CNN)**.

The tool supports dataset preprocessing, training a deep learning model, visualizing accuracy/loss graphs, and predicting cancer stages for new images.

---

## Features

- Upload and preprocess image datasets of lung cancer.
- Train a CNN to classify images into:
  - Normal
  - Stage1
  - Stage2
  - Stage3
- Visualize confusion matrix and accuracy/loss graphs.
- Predict the stage of cancer in real-time from an image.
- GUI for user-friendly interaction.

---

## Major Libraries Used

 `TensorFlow / Keras`  Building and training the CNN model 
 `NumPy`  Numerical operations 
 `OpenCV (cv2)`  Image processing and display 
 `Matplotlib` & `Seaborn`  Visualization (graphs and confusion matrix) 
 `scikit-learn`  Evaluation metrics and train-test splitting 
 `Tkinter`  GUI development 
 `pickle`  Saving training history 

---

## You can install all the required libraries using pip:
pip install tensorflow keras numpy opencv-python matplotlib seaborn scikit-learn


### Execute the main script:

  ```bash
  python LungCancerStages.py 

