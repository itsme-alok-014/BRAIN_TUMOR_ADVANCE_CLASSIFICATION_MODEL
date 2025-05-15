# 🧠 Advanced Brain Tumor Classification System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# 📌 Project Overview

Advanced Brain Tumor Classification System is a Deep Learning based medical imaging application developed to classify brain MRI scans into four categories:

- 🧠 Glioma
- 🧠 Meningioma
- 🧠 Pituitary Tumor
- ✅ No Tumor

The application provides an interactive Streamlit interface that allows users to upload MRI images, choose between multiple trained models, and obtain predictions with confidence scores.

The project demonstrates an end-to-end Deep Learning pipeline including image preprocessing, model training, evaluation, and deployment.

---

# 🚀 Features

- Multi-Class Brain Tumor Classification
- CNN Model
- Hybrid Deep Learning Model
- Brain Contour Detection & Cropping
- MRI Image Preprocessing
- Confidence Score Visualization
- Interactive Streamlit Dashboard
- Model Selection (CNN / Hybrid)
- Fast Real-Time Prediction
- Medical Information Panel
- User-Friendly Interface

---

# 🖥️ Application Preview

## Home Page

> *(Add screenshot later)*

```
screenshots/home_page.png
```

---

## Prediction Result

> *(Add screenshot later)*

```
screenshots/prediction_result.png
```

---

# 🧠 Model Performance

| Model | Test Accuracy | F1 Score | Test Loss |
|--------|--------------:|----------:|----------:|
| CNN | **97.26%** | **97.25%** | **0.0915** |
| Hybrid Model | **98.47%** | **98.47%** | **0.0342** |

The Hybrid Deep Learning model achieved the highest performance on the test dataset.

---

# 📂 Dataset

The project classifies MRI images into the following four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

> **Note:** The dataset is not included in this repository due to GitHub size limitations. Add the dataset to the appropriate folder before retraining the models.

---

# 🛠️ Technology Stack

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV
- Pillow
- Imutils

### Data Processing

- NumPy
- Pandas
- Scikit-Learn

### Deployment

- Streamlit

---

# 📁 Project Structure

```text
Advanced_Brain_Tumor_Classification_System/

│
├── Brain_Tumor_Classification_Models/
│     ├── final_cnn_model.keras
│     └── final_hybrid_model.keras
│
├── sample_images/
│
├── screenshots/
│
├── streamlitapp.py
├── Brain_Tumor_Detection_Model_Builder.ipynb
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── .gitattributes
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/itsme-alok-014/BRAIN_TUMOR_ADVANCE_CLASSIFICATION_MODEL.git
```

Move into the project folder

```bash
cd BRAIN_TUMOR_ADVANCE_CLASSIFICATION_MODEL
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# 📈 Workflow

```
MRI Image
      │
      ▼
Image Upload
      │
      ▼
Brain Contour Detection
      │
      ▼
Image Preprocessing
      │
      ▼
CNN / Hybrid Model
      │
      ▼
Prediction
      │
      ▼
Confidence Score
      │
      ▼
Result Display
```

---

# 🔮 Future Improvements

- Model Explainability using Grad-CAM
- Cloud Deployment
- DICOM Image Support
- PDF Medical Report Generation
- Multi-Language Support
- Mobile Responsive Interface

---

# ⚠️ Medical Disclaimer

This application is intended for educational and research purposes only.

It is **not** a substitute for professional medical diagnosis or clinical decision-making.

---

# 👨‍💻 Author

**Alok Prakash Dubey**

GitHub:

https://github.com/itsme-alok-014

---

# 📜 License

This project is licensed under the MIT License.
