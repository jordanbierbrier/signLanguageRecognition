# Real-Time Sign Language Recognition

![Demo Banner](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📄 Full Report
Find detailed report [here](report.pdf).

---

## 🎬 Demo

![Demo](./demo.gif) 

---

**Computer Vision** + **Machine Learning** models for recognizing American Sign Language (ASL) letters in real-time using a webcam.  
Implemented three approaches:
- 🧠 **HOG + SVM**
- 🧠 **SIFT + Bag of Words + SVM**
- 🧠 **ResNet-50 Transfer Learning**

---

## 📂 Project Structure

| File / Folder | Description |
|:---|:---|
| `/models/` | Saved trained models (`.joblib`) |
| `train_resnet.ipynb` | Train a ResNet-50 model on ASL letter images |
| `train_svm_hog.ipynb` | Train an SVM using HOG features extracted from images |
| `train_svm_sift.ipynb` | Train an SVM using SIFT features + Bag of Words |
| `predict_hog.py` | Real-time ASL prediction using HOG features |
| `predict_sift.py` | Real-time ASL prediction using SIFT features |
| `predict_resnet.py` | Real-time ASL prediction using a trained ResNet-50 |
| `Project_Report.pdf` | Full project report with methodology and results |

---

## 🚀 How to Run

1. **Clone this repository:**
    ```bash
    git clone https://github.com/jordanbierbrier/signLanguageRecognition.git
    cd signLanguageRecognition
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run real-time prediction:**
    ```bash
    # Choose one of the models:
    python predict_hog.py
    python predict_sift.py
    python predict_resnet.py
    ```

4. **Webcam Controls during Prediction:**

| Key | Action |
|:---|:---|
| `b` | Capture background (must press for first prediction) |
| `space` | Add space to the predicted word |
| `s` | Toggle displaying current word |
| `backspace` | Delete last predicted letter |
| `r` | Reset the entire word |
| `ESC` | Exit program |


## 📊 Results and Observations
- **HOG + SVM** provided a good tradeoff between speed and accuracy for real-time use.
- **SIFT features** achieved high validation accuracy but were unstable in live prediction (sensitive to small hand changes).
- **ResNet-50** transfer learning underperformed due to a small dataset and domain difference from ImageNet.

---
Feel free to reach out if you have any questions or suggestions!