    # 🎙️ English Accent Detection Tool

This is a Streamlit-based web application that detects the **English accent** from a speaker in a video. It accepts videos from public URLs, YouTube, or local uploads and performs transcription and accent classification using AI models.

---

## 👨‍💻 Created by
**Umair Khalid**  
📍 Cottbus, Germany  
📧 Email: sking3061@gmail.com

---

## 🚀 Features

- Upload a video file or provide a link (MP4 or YouTube).
- Transcribes audio using OpenAI's Whisper model.
- Classifies English accents using a fine-tuned HuggingFace model.
- Displays:
  - Detected accent (e.g., American, British, Indian, etc.)
  - Confidence score (0–100%)
  - Transcript and explanation

---

## 📦 Installation

Make sure you have Python 3.8+ installed. Then:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv accent-env
source accent-env/bin/activate  # On Windows: accent-env\Scripts\activate

# Install all required packages
pip install -r requirements.txt
streamlit run accent_detection_app.py