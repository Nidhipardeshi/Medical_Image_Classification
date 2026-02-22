#  Medical Image Classification

A Deep Learning based web application for classifying medical images (Chest X-Rays) into:
-  NORMAL
-  PNEUMONIA

This project uses a Convolutional Neural Network (CNN) trained using TensorFlow/Keras and deployed using Flask with an interactive web interface.

---

##  Project Overview
Medical Image Classification is designed to assist in detecting Pneumonia from Chest X-Ray images.

Users can:
- Upload an X-Ray image
- Get instant prediction
- View confidence score
- See visually highlighted results (Green = Normal, Red = Pneumonia)

This project demonstrates the practical integration of:
- Deep Learning
- Backend development
- Frontend UI
- Model deployment

---

##  Tech Stack
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- Flask
- HTML
- Git & GitHub

---

##  Project Structure
Medical_Image_Classification/
│
├── dataset/
│ ├── train/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ ├── val/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ └── test/
│ └── PNEUMONIA/
│
├── model/
│ └── medical_model.h5 (Not included due to size limit)
│
├── results/
│ └── accuracy_plot.png
│
├── src/
│ ├── train_model.py
│ └── predict.py
│
├── static/
│
├── templates/
│ └── index.html
│
├── app.py
├── requirements.txt
└── README.md


---

##  Dataset Information
This project uses the **Chest X-Ray Pneumonia Dataset**.

###  Dataset Source

The dataset can be downloaded from:
 Kaggle:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

###  Dataset Structure

After downloading and extracting, the dataset folder should look like:
dataset/
│
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
│
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
│
└── test/
├── NORMAL/
└── PNEUMONIA/

---

##  How To Run The Project

### 1️ Clone Repository
git clone https://github.com/your-username/Medical_Image_Classification.git
cd Medical_Image_Classification

### 2️ Create Virtual Environment
python -m venv venv

### 3️ Install Dependencies
pip install -r requirements.txt

### 4️ Train The Model (First Time Only)
python src/train_model.py
    This will create:
      model/medical_model.h5
      results/accuracy_plot.png

### 5️ Run Flask Application
python app.py

### 6️ Open In Browser
