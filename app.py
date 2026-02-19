import streamlit as st
from utils import predict_disease, model
from gtts import gTTS
import numpy as np
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(
    page_title="Smart Crop Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# ------------------ HEADER ------------------
st.markdown("""
    <h1 style='text-align:center; color:#166534;'>
    🌾 Smart Crop Disease Detection System
    </h1>
""", unsafe_allow_html=True)

st.write("Upload a leaf image to detect disease and get treatment advice.")

# ------------------ LANGUAGE ------------------
language = st.selectbox("🌍 Select Language", ["English", "Telugu", "Hindi"])

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Leaf", width=400)

    with st.spinner("🔍 Analyzing leaf image..."):
        img = image_display.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        disease, solution = predict_disease(uploaded_file)

    st.success(f"🌿 Predicted Disease: {disease}")
    st.info(f"💊 Suggested Treatment: {solution}")
    st.write(f"📊 Confidence: {confidence:.2f}%")

    # ----------- VOICE OUTPUT -----------
    if language == "English":
        text = f"The disease is {disease}. Suggested treatment is {solution}"
        tts = gTTS(text=text, lang='en')
    elif language == "Telugu":
        text = f"ఈ మొక్కకు ఉన్న వ్యాధి {disease}. సూచించిన చికిత్స {solution}"
        tts = gTTS(text=text, lang='te')
    else:
        text = f"इस पौधे की बीमारी {disease} है। सुझाया गया उपचार {solution}"
        tts = gTTS(text=text, lang='hi')

    tts.save("output.mp3")
    st.audio("output.mp3")

    # ----------- DOWNLOAD PDF REPORT -----------
    if st.button("📄 Download Report as PDF"):

        file_name = "Crop_Disease_Report.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        elements = []

        styles = getSampleStyleSheet()
        elements.append(Paragraph("Crop Disease Detection Report", styles['Heading1']))
        elements.append(Spacer(1, 0.5 * inch))

        elements.append(Paragraph(f"Predicted Disease: {disease}", styles['Normal']))
        elements.append(Paragraph(f"Suggested Treatment: {solution}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))

        doc.build(elements)

        with open(file_name, "rb") as file:
            st.download_button(
                label="⬇ Download PDF",
                data=file,
                file_name=file_name,
                mime="application/pdf"
            )
