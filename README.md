# Two-Stage Skin Disease Classification System

This project presents a **Two-Stage Deep Learning System** for skin disease detection and classification.

The system works in two steps:
1. Detects whether the skin is **Healthy or Diseased**
2. If diseased, classifies it into a specific disease category

This two-stage approach improves prediction accuracy and reliability.

---

## 📌 System Architecture

### 🧪 Stage 1: Skin Disease Detection
The first model classifies the image into:
- Healthy
- Diseased

**Datasets Used:**
- DermNet Dataset (23 disease categories) → Diseased class  
- Healthy Skin Dataset → Healthy class  

The model learns to distinguish between normal and abnormal skin conditions.

---

### 🔬 Stage 2: Disease Type Classification
If the image is predicted as **Diseased**, it is passed to Stage 2.

The model classifies into:
- Bacterial  
- Fungal  
- Viral  
- Other Skin Diseases  

Selected categories from DermNet were grouped into these 4 classes.

---

## 📂 Project Structure
project-root/
│
├── assets/ # UI images and static files
├── model/ #models
├── pages/
│ └── Analyze.py
├── scripts/ # Training scripts
├── app.py # Main application file


---

## 📊 Dataset

### 1. DermNet Dataset
- Contains multiple skin disease categories
- Used for:
  - Stage 1 → Diseased detection
  - Stage 2 → Grouped classification

🔗 https://www.kaggle.com/datasets/shubhamgoel27/dermnet

---

### 2. Healthy Skin Dataset
- Used as **Healthy class** in Stage 1

🔗 https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset

---

## 🤖 Trained Models

Due to GitHub size limitations, models are hosted on Google Drive.

### 📥 Download Models
  👉 [Download Link](https://drive.google.com/file/d/1iRfBqxMoshvulD6DiDwaW52pVmp7mESa/view?usp=sharing)
  👉 [Download Link](https://drive.google.com/file/d/1nQQ5aZRwx1cWRqVsyiEymjTjzXHLNepx/view?usp=sharing)
  👉 [Download Link](https://drive.google.com/file/d/1i6VC9_-JTVnKCNwEzLoZ_DQD9fZXrCJM/view?usp=sharing)

📌 After downloading, place all models inside: model/


---

## 🚀 Running the Application

### Step 1: Clone the Repository
git clone https://github.com//nyfaasminpathan/AI-Driven-Skin-Disease-Diagnosis-Framework.git
cd AI-Driven-Skin-Disease-Diagnosis-Framework

###Step 2: Install Dependencies
pip install tensorflow keras numpy pandas matplotlib scikit-learn streamlit

###Step 3: Run the App
python app.py
