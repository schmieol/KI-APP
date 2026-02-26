import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    return model

model = load_model()

# Labels laden
def load_labels():
    with open("labels.txt", "r") as f:
        labels = f.readlines()
    return [label.strip() for label in labels]

labels = load_labels()

# Bild vorbereiten
def preprocess_image(image):
    image = image.resize((224, 224))  # Standardgröße Teachable Machine
    image = np.array(image)
    image = image.astype(np.float32) / 127.5 - 1  # Normalisierung wie TM
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🧠 Produkt-Klassifizierer")
st.write("Lade ein Bild hoch (T-Shirt, Trinkflasche oder Schuh).")

uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.subheader("🔎 Ergebnis:")
    st.write(f"**Klasse:** {predicted_label}")
    st.write(f"**Sicherheit:** {confidence:.2f}%")
