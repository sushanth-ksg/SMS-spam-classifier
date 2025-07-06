
# 📩 SMS Spam Classifier

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Not Spam** using NLP and a Multinomial Naive Bayes model.

---

## 🚀 Live Demo

Try the interactive app here:  
👉 [SMS Spam Classifier on Hugging Face Spaces](https://huggingface.co/spaces/sushanth-ksg/SMS-spam-classifier)

---

## 📌 Project Overview

This project demonstrates a complete machine learning pipeline:
- Text preprocessing with NLTK
- Feature extraction using TF-IDF
- Model training with Multinomial Naive Bayes
- Deployment using Gradio + Hugging Face Spaces

It takes a raw SMS message and predicts whether it's spam using a simple web interface.

---

## 🛠 Tech Stack

- **Python**
- **Scikit-learn** – for model training and evaluation
- **NLTK** – for text cleaning and preprocessing
- **Gradio** – for building the interactive UI
- **Hugging Face Spaces** – for deploying the app
- **Pickle** – for saving and loading models

---

## 🧠 Model Details

- **Algorithm**: Multinomial Naive Bayes (best suited for text classification)
- **Vectorization**: TF-IDF
- **Dataset**: SMS Spam Collection Dataset (from UCI repository or Kaggle)

---

## 📂 Project Structure
sms-spam-classifier/
│
├── app.py # Gradio interface and prediction logic
├── train_model.py # Script to train model and save .pkl files
├── text_utils.py # Custom text preprocessing logic
├── model.pkl # Trained Naive Bayes model
├── vectorizer.pkl # Fitted TF-IDF vectorizer
├── spam.csv # Dataset
└── requirements.txt # Python dependencies

📚 Acknowledgements
UCI SMS Spam Collection Dataset

Gradio – for the awesome UI framework

Hugging Face Spaces – for free model hosting

🙋‍♂️ Author
Sushanth KSG
📧 ksgsushanth@gmail.com
