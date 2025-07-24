# 🧠 Brain Tumor Advance Classification Model

This project presents a deep learning solution for **multi-class classification of brain tumors** using MRI images. It includes **data augmentation**, a **Convolutional Neural Network (CNN)** model, and a **Streamlit-based web app** for real-time prediction.

---

## 🚀 Live Demo

🔗 [Launch the App on Streamlit](https://<your-app-link>)  
_Use the interface to upload an MRI image and predict tumor class._

---

## 📂 Project Structure

```bash
brain-tumor-cnn/
├── model/
│   └── brain_model.h5                # Trained model (tracked via Git LFS)
├── notebooks/
│   ├── Brain_Tumor_Data_Augmentation.ipynb
│   └── Brain_Tumor_Detection_Model_Builder.ipynb
├── app/
│   └── streamlit_app.py              # Streamlit frontend
├── utils/
│   └── preprocessing.py              # Image preprocessing scripts
├── data/
│   └── (placeholders or link to dataset)
├── README.md
├── .gitignore
├── .gitattributes
````

---

## 🧪 Features

* 📊 **MRI Image Classification** for 3+ tumor types
* 🧠 Built with a **custom CNN architecture**
* 🔄 **Data augmentation** for improved generalization
* 🖼️ Real-time prediction via **Streamlit app**
* 📦 Model hosted with **Git LFS** (900MB)

---

## 🖥️ Notebooks Included

1. **Brain\_Tumor\_Data\_Augmentation.ipynb**
   ➤ Prepares augmented training data using Keras’ ImageDataGenerator

2. **Brain\_Tumor\_Detection\_Model\_Builder.ipynb**
   ➤ Builds, trains, and evaluates the CNN model for classification

---

## 🛠️ Technologies Used

* Python 3
* TensorFlow / Keras
* NumPy, Pandas, OpenCV
* Matplotlib, Seaborn
* Streamlit (for UI)
* Git LFS (for model storage)

---

## 🧠 Model Performance

* Input: MRI Image (resized)
* Output: Tumor Class Label
* Accuracy: \~90% (on test data)

---

## 🔗 Dataset

If you want to train the model from scratch, use a publicly available dataset such as:

* [Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

> **Note**: Dataset is not included in the repo due to size. Link provided above.

---

## ⚙️ How to Run Locally

1. Clone the repository

```bash
git clone https://github.com/itsme-alok-014/BRAIN_TUMOR_ADVANCE_CLASSIFICATION_MODEL.git
cd BRAIN_TUMOR_ADVANCE_CLASSIFICATION_MODEL
```

2. Set up environment

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 📦 Download Trained Model

This repo uses Git LFS to store the model (`brain_model.h5`). If it's not downloading automatically:

```bash
git lfs install
git lfs pull
```

Or download manually from the [Releases](https://github.com/itsme-alok-014/BRAIN_TUMOR_ADVANCE_CLASSIFICATION_MODEL/releases).

---

## 🙌 Contribution

Feel free to fork the project, open issues, or submit pull requests for improvements.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 👨‍💻 Author

Made with ❤️ by **Alok Raj Dubey**

* Help upload the model via GitHub Releases instead of LFS

I'm here to assist!
