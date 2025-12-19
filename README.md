# ASL-Type: Real-Time Sign Language Typing System

ASL-Type is a **real-time Sign Language Typing System** that translates **American Sign Language (ASL)** hand gestures into text using a **Convolutional Neural Network (CNN)** and webcam-based input.

The project combines **deep learning–based image classification** with **real-time computer vision** to build an assistive typing interface for gesture-based communication.

---

## Features

- Recognizes **29 ASL gesture classes**
  - Alphabets **A–Z**
  - Special tokens: **space**, **del**, **nothing**
- **CNN-based image classification** trained on ASL Alphabet dataset
- **Real-time gesture capture** using OpenCV and webcam
- **Stability-based prediction mechanism** to reduce noisy predictions
- **Live text generation** from hand gestures

---

## Tech Stack

- **Programming Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Numerical Computing:** NumPy  
- **Training Environment:** Google Colab (GPU)

---

## Dataset

- **Source:** [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Total Classes:** 29
- **Image Size:** 100 × 100 × 3
- 

### Data Split
- **80%** Training  
- **20%** Validation  

### Preprocessing & Augmentation
- Image resizing and normalization  
- Random horizontal flip  
- Random rotation  
- Random zoom  

---

## Model Architecture

The system uses a **multi-layer CNN** with the following structure:

1. Conv2D (32 filters, 3×3) + ReLU  
2. MaxPooling2D (2×2)  
3. Conv2D (64 filters, 3×3) + ReLU  
4. MaxPooling2D  
5. Conv2D (128 filters, 3×3) + ReLU  
6. MaxPooling2D  
7. Flatten  
8. Dense (512 units) + ReLU  
9. Dropout (0.5)  
10. Dense Output Layer (Softmax, 29 units)

**Loss Function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy

---

## Results

- Achieved **~97% validation accuracy** after training for 20 epochs  
- Model performs well on **clean, well-lit hand gestures**  
- Integrated successfully with webcam for **real-time predictions**

---

## Real-Time Prediction Pipeline

1. Capture frames using webcam  
2. Define and extract **Region of Interest (ROI)**  
3. Resize ROI to **100 × 100**  
4. Normalize image and pass to CNN  
5. Predict gesture class  
6. Apply **stability threshold** to confirm predictions  
7. Generate text output  

---

## Limitations & Observations

Performance degrades under:
- Poor lighting conditions  
- Complex backgrounds  
- Inconsistent hand positioning  

Additional challenges:
- Some ASL gestures have **high visual similarity** (e.g., M/N, D/G)  
- Dynamic gestures (e.g., **J**, **Z**) are difficult to recognize using a frame-based CNN  
- Model experiences **domain shift** between dataset images and real-world webcam input  

These limitations are typical in real-time vision systems and highlight areas for improvement.

---

## Future Improvements

- Hand segmentation or background removal  
- Temporal modeling (CNN + LSTM / 3D CNN) for dynamic gestures  
- Training on more diverse real-world datasets  
- Adaptive ROI and automatic hand detection  
- Text-to-speech output  
- Mobile or web deployment  

---

## Applications

- Assistive technology for deaf or speech-impaired users  
- Gesture-based typing interfaces  
- Educational tools  
- Human–computer interaction systems  

---

