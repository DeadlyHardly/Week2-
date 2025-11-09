# 🌱 Plant Disease Classification Using Deep Learning

### 👨‍💻 Automated AI-powered leaf disease detection system for sustainable agriculture 🌾

---

## 📌 Overview

This project is a **Convolutional Neural Network (CNN)** based image classification model built to automatically detect and classify plant diseases from leaf images.  
It supports **sustainable farming practices** by helping farmers identify crop diseases early, reduce chemical usage, and improve crop yields.  

This project was developed as part of the **Edunet Skills 4 Future Internship (Theme: Sustainability)**, integrating **AI + Sustainability** to solve real-world agricultural problems.

---

## 🎯 Problem Statement

Farmers often struggle to identify crop diseases accurately and on time, leading to reduced yields and overuse of pesticides.  
Early detection through technology can improve productivity and help achieve **UN Sustainable Development Goal (SDG 2): Zero Hunger** and **SDG 12: Responsible Consumption and Production**.

**Goal:**  
> To create a deep learning model that can classify plant leaf images as healthy or diseased, supporting efficient and sustainable agriculture.

---

## 🧠 Objectives

- Build a **CNN-based classifier** to identify plant diseases from images.
- Use the **PlantVillage dataset** for model training and testing.
- Achieve at least **90%+ accuracy** on validation data.
- Promote sustainable agriculture by reducing excessive pesticide use.
- Deploy the model for practical use by farmers via a user-friendly interface (future scope).

---

## 🌿 Sustainability Focus

| Challenge | Sustainable Solution |
|------------|----------------------|
| 🌾 Disease misidentification | Automated detection via CNN |
| 🧪 Excess pesticide use | Early detection → Fewer chemicals |
| 💰 Low crop yield | Timely treatment → Better productivity |
| 🌍 Environmental impact | AI-assisted eco-friendly farming |

This project directly supports the **Sustainable Development Goals (SDGs)**:
- **SDG 2:** Zero Hunger  
- **SDG 12:** Responsible Consumption and Production  
- **SDG 13:** Climate Action  

---

## 🗂️ Dataset

- **Dataset Name:** PlantVillage Dataset  
- **Source:** [Kaggle - Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Total Images:** 54,000+ labeled images  
- **Classes:** 38 crop-disease combinations (e.g., Apple Scab, Tomato Mosaic Virus, Corn Rust, etc.)
- **Split:** 80% Training / 20% Testing  

**Sample Classes:**
- Apple___Black_rot  
- Corn___Gray_leaf_spot  
- Tomato___Leaf_Mold  
- Grape___Esca_(Black_Measles)  
- Potato___Early_blight  
- ...and more 🌿  

---

## 🧩 Methodology

### 1️⃣ Data Collection & Preprocessing
- Images resized to 128x128 pixels  
- Normalized pixel values (0–1 range)  
- Augmented dataset (rotation, zoom, flip) to increase robustness  

### 2️⃣ Model Architecture
The CNN model includes:
- 3 Convolutional Layers (ReLU + MaxPooling)
- 1 Flatten Layer  
- 2 Dense Layers (128 neurons + Softmax output)  
- Dropout Layer (to prevent overfitting)

### 3️⃣ Model Compilation
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metric:** Accuracy  

### 4️⃣ Model Training
- 10 epochs on GPU (Google Colab)  
- Batch size: 32  
- Validation split: 20%  

### 5️⃣ Model Evaluation
- Accuracy: ~93% on validation data  
- Loss: ~0.15  
- Model saved as `plant_disease_model.h5`

---

## 🧮 Model Architecture Summary

| Layer Type | Output Shape | Parameters |
|-------------|---------------|------------|
| Conv2D | (128,128,32) | 896 |
| MaxPooling2D | (64,64,32) | 0 |
| Conv2D | (62,62,64) | 18496 |
| MaxPooling2D | (31,31,64) | 0 |
| Flatten | (61504) | 0 |
| Dense | (128) | 7864640 |
| Dropout | - | - |
| Dense | (38) | 4902 |
| **Total Params** | **7,885,000+** | |

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|--------|
| Programming | Python 3.8+ |
| Libraries | TensorFlow, Keras, NumPy, OpenCV, Matplotlib |
| IDE | Google Colab |
| Dataset | Kaggle - PlantVillage |
| Version Control | Git + GitHub |
| Output Format | `.h5` model file |

---

## 📊 Results

| Metric | Training | Validation |
|---------|-----------|------------|
| Accuracy | 95% | 91% |
| Loss | 0.12 | 0.18 |

**Visualization:**
- Accuracy and Loss graphs show stable convergence.
- Model generalizes well without overfitting.

---

## 🖥️ Sample Code Snippet

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(38, activation='softmax')
])
