# Sign-Language-Detection-System


# 🧠 Real-Time Sign Language Detection System

A real-time Sign Language Recognition System using hand gestures, built with Python, OpenCV, TensorFlow, MediaPipe, and Flet. This project allows users to detect American Sign Language (ASL) gestures through their webcam and translate them into readable text, with optional word suggestion and voice feedback.

---

## 📁 Project Structure

```
├── app.py                  # Main Application File (UI + Prediction)
├── data_collect.py         # Data Collection Script for Custom Training
├── models/
│   ├── keras_model.h5      # Pre-trained Keras Model
│   └── labels.txt          # Class Labels (A–Z, 1–9, Space)
├── data/
│   ├── A/ ...              # Collected gesture images per label
│   ├── B/ ...
│   ├── ...
│   └── SPACE/
└── README.md               # Project Overview
```

---

## 🚀 Features

- 🔤 Real-Time ASL Gesture Prediction (A–Z, 1–9, Space)
- 📷 Webcam-based Live Detection using OpenCV
- ✋ Hand Tracking with MediaPipe
- 🧠 Model Trained using Teachable Machine
- 🧩 Trie-based Word Suggestions from Common Vocabulary
- 🗣️ Text-to-Speech Feedback (Optional)
- 🧭 Auto & Manual Prediction Modes
- 🖥️ Modern UI with Flet

---

## 🧪 Model Details

- **Model Type:** Keras (TensorFlow backend)
- **Input Size:** 224x224
- **Classes:** 36 (A–Z, 0–9, Space)
- **Trained On:** ~300–400 images per class via Teachable Machine

---

## 🛠️ Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```
Note: You need Meidapipe 0.8.10.1, otherwise it won't run. Since the version is unavailable now, so here is the link: https://drive.google.com/file/d/1PcFUY8UGcgMVqvy47H4pMOXYGlZpf0vc/view?usp=drive_link

Typical requirements:

```
opencv-python
flet
tensorflow
numpy
mediapipe
pyttsx3
```

---

## ▶️ How to Run

### 1. Launch the Application
```bash
python app.py
```

### 2. Run Data Collection (Optional)
```bash
python data_collect.py
```

> Follow the prompts and show gestures to collect new images. Stored in `data/`.

---

## 🧠 Trie-Based Word Suggestion

- A list of 100–200+ common words is loaded into a Trie data structure.
- As the user signs letters, the system auto-suggests likely matching words.

---

## 🧑‍💻 Author

- **Name:** Harsh Damame  
- **Project Type:** Final Year BSc IT Project  
- **Developed Using:** Python, OpenCV, TensorFlow, Flet, MediaPipe  

---

## 📸 Screenshots

![Screenshot 2025-04-12 193743](https://github.com/user-attachments/assets/1bfaf610-1375-47dc-85f2-5ae8c89f8411)
![Screenshot 2025-04-12 193950](https://github.com/user-attachments/assets/c1ec0a73-99a2-4408-bcfb-ee58cd18504e)
![Screenshot 2025-04-12 194100](https://github.com/user-attachments/assets/b3efe065-6786-4913-9935-def493213884)
![Screenshot 2025-04-12 194212](https://github.com/user-attachments/assets/024785fb-d38f-49cf-b830-2aa116830ab6)
![Screenshot 2025-04-12 194419](https://github.com/user-attachments/assets/21b147e9-f8c6-47e3-b3cc-84c9472526c6)
![Screenshot 2025-04-12 194434](https://github.com/user-attachments/assets/b0483fe0-8ca2-4e8c-b86c-21f52d44ce28)


---

## 📌 Future Enhancements

- Adding dynamic learning for new words
- Enhance accuracy with more data and augmentations
- Improve GUI interactivity and responsiveness
- Adding multilingual voice output

---
