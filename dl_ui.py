import gradio as gr
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np  # Fixed: was 'nb'

MAX_SEQUENCE_LENGTH = 100

# Load model - Fixed path separators
model = tf.keras.models.load_model("models/lstm_fragment_detector.h5")

# Load tokenizer - Fixed path separators
with open("models/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def dl_predict(text):
    # Tokenize and pad the input
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)[0][0]
    
    # Return label
    label = "Fragment" if prediction > 0.5 else "Not a Fragment"
    return f"{label} ({prediction:.2f})"

dl_interface = gr.Interface(
    fn=dl_predict,
    inputs=gr.Textbox(label="Enter text"),
    outputs=gr.Text(label="DL Model Prediction"),
    title="Fragment DL Classifier"
)
