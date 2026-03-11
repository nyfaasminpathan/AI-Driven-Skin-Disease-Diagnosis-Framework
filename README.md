Two-Stage Skin Disease Classification System

This project presents a Two-Stage Deep Learning System for Skin Disease Detection and Classification.
The system first determines whether a skin image is healthy or diseased. If the image is identified as diseased, the system then performs a second level classification to determine the type of disease.
This approach improves prediction reliability by separating disease detection and disease type classification into two stages.

System Architecture

The project is implemented as a two-stage classification pipeline.

Stage 1: Skin Disease Detection
The first model determines whether the given skin image is:
->Healthy
->Diseased
Dataset used for this stage:
DermNet Dataset (23 disease categories) → used as diseased images
Healthy Skin Dataset → used as healthy class
The model learns to distinguish between normal skin and abnormal skin conditions


Stage 2: Disease Type Classification
If the image is predicted as Diseased in Stage 1, it is passed to the Stage 2 model.
The Stage 2 model classifies the disease into four categories:
->Bacterial
->Fungal
->Viral
->Other Skin Diseases
For this stage, specific disease categories from the DermNet dataset were selected and grouped into the above four classes.


Project Structure
project-root 
│ 
├── assets/ 
│       └── UI images and static resources 
│ ├── model/ 
│       ├── stage1_model.h5 
│       └── stage2_model.h5 
│ ├── pages/ 
│         └── Analyze.py 
│ ├── scripts/ 
│         └── training scripts
│ ├── app.py 
│         └── Main application file that runs the UI and integrates both models 


Dataset

The datasets used in this project include:
1. DermNet Dataset
The DermNet dataset contains multiple categories of skin disease images.
For this project:
All 23 disease categories were used in Stage 1 for diseased detection.
Selected categories were grouped into Bacterial, Viral, Fungal, and Other classes for Stage 2 classification.
2. Healthy Skin Dataset
Healthy skin images were collected from an external dataset and used as the healthy class for Stage 1 training.
Due to GitHub size limitations, the datasets are hosted externally.
Dataset Link:
https://www.kaggle.com/datasets/shubhamgoel27/dermnet
https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset

Trained models are stored inside the model/ directory.


---Running the Application---

Step 1: Clone the Repository
cd your-repository-name
Step 2: Install Required Libraries(pip install tensorflow keras numpy pandas matplotlib scikit-learn streamlit)
Step 3: Run the Application
python app.py
After running the command, the application interface will start in your browser.
