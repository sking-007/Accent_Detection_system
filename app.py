import streamlit as st
import tempfile, requests, os, glob, subprocess
import whisper
import librosa, torch
import yt_dlp
from transformers import AutoProcessor, AutoModelForAudioClassification

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Load Accent Classifier
@st.cache_resource
def load_accent_model():
    model_id = "dima806/english_accents_classification"
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# Download from direct video link
def download_video(url):
    try:
        response = requests.get(url, stream=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(tmp.name, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        return tmp.name
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

# Download from YouTube
def download_youtube_audio(youtube_url):
    tmp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tmp_dir, 'audio.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        files = glob.glob(os.path.join(tmp_dir, 'audio.*'))
        if not files:
            raise FileNotFoundError("Audio download failed.")
        return files[0]
    except Exception as e:
        raise RuntimeError(f"YouTube download failed: {e}")

# Extract audio via ffmpeg
def extract_audio(input_path):
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-vn", "-y", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        if os.path.getsize(output_path) == 0:
            raise ValueError("Audio extraction output is empty.")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")

# Accent classification
def classify_accent(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)
    if waveform.size == 0:
        raise ValueError("Audio file is empty or unreadable.")

    duration = librosa.get_duration(y=waveform, sr=sr)
    if duration < 1.0:
        raise ValueError("Audio is too short for accent classification.")

    model, processor = load_accent_model()
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    if inputs['input_values'].shape[1] == 0:
        raise ValueError("Processed audio has no content.")

    with torch.no_grad():
        logits = model(**inputs).logits
        pid = logits.argmax().item()
        conf = torch.softmax(logits, dim=1)[0][pid].item()

    label = model.config.id2label[pid]
    readable = {
        "us": "American", "england": "British", "indian": "Indian",
        "australia": "Australian", "canada": "Canadian"
    }.get(label, label.capitalize())

    explanation = f"Accent classified as **{readable}** with **{conf * 100:.2f}%** confidence."
    return readable, round(conf * 100, 2), explanation, duration

# ───────────────────── Streamlit UI ─────────────────────
st.title("🎙️ English Accent Detection Tool")

uploaded = st.file_uploader("📁 Upload a video", type=["mp4", "mov", "webm"])
url_mp4 = st.text_input("🔗 Public MP4 URL")
url_yt = st.text_input("▶ YouTube URL")

if st.button("Analyze"):
    if not uploaded and not url_mp4 and not url_yt:
        st.warning("Please upload a video or provide a URL.")
    else:
        with st.spinner("Processing..."):
            video_path, audio_path = None, None
            try:
                if uploaded:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp.write(uploaded.read())
                    video_path = tmp.name
                elif url_yt:
                    video_path = download_youtube_audio(url_yt)
                else:
                    video_path = download_video(url_mp4)

                audio_path = extract_audio(video_path)
                st.audio(audio_path)

                # Transcript
                whisper_model = load_whisper_model()
                result = whisper_model.transcribe(audio_path)
                st.subheader("📝 Transcript")
                st.write(result["text"])

                # Accent
                accent, score, explanation, duration = classify_accent(audio_path)
                st.subheader("🧠 Accent Analysis")
                st.write(f"**Detected Accent:** {accent}")
                st.write(f"**Confidence Score:** {score}%")
                st.write(f"**Audio Duration:** {duration:.2f} seconds")
                st.write(explanation)

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
            finally:
                for f in [video_path, audio_path]:
                    if f and os.path.exists(f):
                        os.remove(f)
