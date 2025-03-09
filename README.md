# Fashion Product Classification API

## This project is a **Deep Learning-based Flask API** for predicting **fashion product attributes** from an image. 

**The model classifies an image into four categories:**  
**Color** (e.g., Black, White, Red)  
**Season** (e.g., Summer, Winter, All-Season)  
**Gender** (e.g., Men, Women, Unisex)  
**Product Type** (e.g., T-shirt, Shoes, Jacket)  

---

## Features
- **Multi-class Classification** using Deep Learning  
- **Lightweight & Fast API** using Flask  
- **Pre-trained Model Loading** for instant predictions  
- **Deployable as a Web API**  

---

## Project Structure
- **`research/1. data-processing-and-eda.ipynb`**: Notebook for data preprocessing and exploratory data analysis (EDA).  
- **`research/2. model-training-inference.ipynb`**: Notebook for model training and inference implementation.  
- **`app.py`**: Flask API for serving the model.  
- **`predict.py`**: Contains functions for loading the trained model and making predictions.  
- **`model_architecture.py`**: Defines the deep learning model architecture.  
- **`models/fashion_classifier.pth`**: Saved weights of the trained model.  
- **`models/label_encoders.pth`**: Label encoders for decoding class predictions. 

---

## Installation Guide

### **1️ Create & Activate a Virtual Environment**
```bash
conda create -p {env_name} python==3.10.16 -y
conda activate {env_name}/
```

### **2 Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3 How to run the Flask API**
```bash
python app.py
```

### **4 API TestPoint**
```bash
import requests

url = "http://127.0.0.1:5000/predict"
file_path = "/path/to/image.jpg"

files = {"file": open(file_path, "rb")}
response = requests.post(url, files=files)

print(response.json())  # Print API response

```