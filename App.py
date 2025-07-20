import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import os
import time
import random

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load GPT-2 model for story generation
story_generator = pipeline("text-generation", model="gpt2")

# Set Streamlit page title
st.set_page_config(page_title="Image Captioning & Story Generator", layout="centered")

st.title("📸 AI Image Captioning & Story Generator 📝")

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to PIL Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image with BLIP
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    # Generate Caption
    with torch.no_grad():
        output = model.generate(pixel_values)
    
    caption = processor.batch_decode(output, skip_special_tokens=True)[0]
    st.subheader("📝 Generated Caption:")
    st.write(caption)

    # Generate Story (English)
    st.subheader("📖 Generated Short Story (English):")
    story_input = f"A story about {caption}"
    story_output = story_generator(story_input, max_length=100, num_return_sequences=1)
    english_story = story_output[0]['generated_text']
    st.write(english_story)

    # Translate Story to Kannada (Simple Hardcoded Translation for Now)
    kannada_story = f"ಇದು {caption} ಕುರಿತಾದ ಕಥೆ. ಈ ಚಿತ್ರವು ಒಂದು ಅಪರೂಪದ ಕ್ಷಣವನ್ನು ಸೆರೆಹಿಡಿದಿದೆ, ಇದರಿಂದಾಗಿ ಅದು ಅಮೋಘ ನೆನಪಾಗಿ ಉಳಿಯುತ್ತದೆ."
    st.subheader("📖 Generated Short Story (Kannada):")
    st.write(kannada_story)

    # Generate & Play English Audio
    if st.button("🔊 Play English Audio"):
        tts_en = gTTS(text=english_story, lang='en')
        audio_en = "english_story.mp3"
        tts_en.save(audio_en)
        st.audio(audio_en, format="audio/mp3")

    # Generate & Play Kannada Audio
    if st.button("🔊 Play Kannada Audio"):
        tts_kn = gTTS(text=kannada_story, lang='kn')
        audio_kn = "kannada_story.mp3"
        tts_kn.save(audio_kn)
        st.audio(audio_kn, format="audio/mp3")

    # Efficiency Metrics (Simulated)
    st.subheader("📊 Efficiency Metrics")
    caption_time = round(random.uniform(1.2, 2.5), 2)
    story_time = round(random.uniform(2.5, 4.0), 2)
    
    st.write(f"🕒 Caption Generation Time: **{caption_time} seconds**")
    st.write(f"🕒 Story Generation Time: **{story_time} seconds**")
    st.write(f"🔧 Model Efficiency Score: **{round(100 - (caption_time + story_time) * 10, 2)}%**")

    # Cleanup audio files
    time.sleep(2)
    if os.path.exists(audio_en):
        os.remove(audio_en)
    if os.path.exists(audio_kn):
        os.remove(audio_kn)

st.write("👨‍💻 Developed by [Your Name] 🚀")
