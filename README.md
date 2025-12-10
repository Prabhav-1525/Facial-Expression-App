# Facial Expression Recognition App (FER2013 → Flask)
Real-time emotion recognition (happy, sad, angry, neutral) from webcam frames, using TensorFlow/Keras and OpenCV.

## 📖 Description

**Emotion Recognition** is a cutting-edge AI and Machine Learning project designed to detect human emotions through facial expressions.

This system leverages **Deep Learning (CNN + Transfer Learning)** to identify emotions such as *happy*, *sad*, *angry*, and *neutral* with high accuracy. It can be integrated into multiple domains — from mental health analysis and sentiment detection to human-computer interaction and real-time communication.

The modular, scalable architecture enables seamless updates and multi-platform deployment, including web (**Flask**) and **Streamlit** applications.

---

## ✨ Features

* **Emotion Detection** — Recognizes emotions such as happy, sad, angry, and neutral from facial expressions.
* **Deep Learning-Based** — Powered by Convolutional Neural Networks (CNNs) with Transfer Learning for robust accuracy.
* **Large Dataset** — Trained on the FER2013 dataset of labeled facial expressions.
* **Real-time Recognition** — Supports live camera or snapshot-based prediction.
* **Cross-Platform Scalability** — Works seamlessly across Windows, macOS, and Linux.
* **Modular Architecture** — Separate modules for preprocessing, training, and inference for easy maintainability.
* **User-Friendly UI** — Simple and intuitive interface built using Flask and Streamlit.
* **Web Deployed** — Fully deployed on Streamlit Cloud for public testing and demonstration.

---

## 🌐 Live Demo

You can try the deployed Streamlit version of the Emotion Recognition app here:

👉 **(https://facial-expression-app-prabhav.streamlit.app/)**

*This deployment supports real-time image-based emotion prediction directly in your browser — no setup needed!*

---

## 🧰 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask |
| **Web App** | Streamlit |
| **Machine Learning** | TensorFlow, Keras |
| **Computer Vision** | OpenCV |
| **Libraries** | NumPy, Pandas |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

```text
Facial-Expression-App/
│
├── app/                # Flask backend (routes, templates)
├── models/             # Trained CNN/Transfer Learning models
├── training/           # Model training and preprocessing scripts
├── static/             # CSS, JS, and assets for Flask UI
├── templates/          # HTML templates for Flask
├── utils/              # Helper utilities
├── streamlit_app.py    # Streamlit application (public deployment)
├── requirements.txt    # Dependencies
└── README.md
```

## ⚙️ How to Run

### 1️⃣ Local Setup
Clone the repository and install dependencies:

```bash
git clone [https://github.com/Prabhav-1525/Facial-Expression-App.git](https://github.com/Prabhav-1525/Facial-Expression-App.git)
cd "Facial Expression Recognition App"
pip install -r requirements.txt
#mac source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Run Flask App

To run the Flask backend:

```bash
python app/server.py
```

### 3️⃣ Run Streamlit App

To launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

## 🚀 Deployment

This project is deployed live on **Streamlit Cloud**.

The deployment uses:
* `requirements.txt` for dependency management.
* `.streamlit/config.toml` for Streamlit configuration.
* `secrets.toml` for API credentials (if needed).

No manual setup required — visit the **Live Streamlit App** link above to explore the hosted demo.

## 🧪 Testing Instructions

1. Launch the Flask or Streamlit app.
2. Upload a facial image (JPG/PNG).
3. The app detects and classifies the dominant emotion in the uploaded image.
4. Verify prediction output and confidence scores.

   ## 📦 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | GET | Returns the home page |
| `/recognize` | POST | Accepts an image and returns detected emotion |
| `/streamlit` | GET | Loads the Streamlit interface |

## 👤 Author

Developed and maintained by **Prabhav Saxena**.

