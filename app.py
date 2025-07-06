from text_utils import transform_text

import pickle
import gradio as gr

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_sms(text):
    vect = vectorizer.transform([text])
    result = model.predict(vect)[0]
    return "Spam 🚫" if result == 1 else "Not Spam ✅"

interface = gr.Interface(
    fn=predict_sms,
    inputs=gr.Textbox(lines=3, placeholder="Enter SMS..."),
    outputs=gr.Text(label="Prediction"),
    title="SMS Spam Classifier"
)
interface.launch()
