import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from transformers import pipeline
import whisper
from moviepy import VideoFileClip
import os
import yt_dlp

nltk.download("punkt")

# Load Whisper model
model = whisper.load_model("tiny")

# BART Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extractive summarization
def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    word_frequencies = Counter(word_tokenize(text.lower()))
    ranked_sentences = sorted(sentences, key=lambda s: sum(word_frequencies[w] for w in word_tokenize(s.lower())), reverse=True)
    return " ".join(ranked_sentences[:num_sentences])

# BART summarization
def bart_summarize(text):
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

# Whisper transcription
def whisper_transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# YouTube to audio using yt-dlp
def download_youtube_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'yt_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "yt_audio.mp3"
    except Exception as e:
        st.error(f"yt-dlp error: {e}")
        return None

# Convert local video to audio
def video_to_audio(video_file):
    clip = VideoFileClip(video_file)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path)
    return audio_path

# Streamlit UI
st.title("🎙️ NotesBuddy - Summarize Notes, Videos & Audio")
st.write("Use AI & NLP to summarize text, YouTube videos, or your own notes.")

# Text summarization
st.header("📝 Text Input")
user_input = st.text_area("Enter your notes:")
summary_method = st.selectbox("Choose summarization method", ["Extractive (NLTK)", "AI-Based (BART)"])

if st.button("Summarize Text"):
    if user_input:
        summary = summarize_text(user_input) if summary_method == "Extractive (NLTK)" else bart_summarize(user_input)
        st.success("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text.")

# Audio upload
st.header("🎧 Upload Audio (WAV)")
uploaded_audio = st.file_uploader("Upload audio file (WAV only)", type=["wav"])
if uploaded_audio is not None:
    try:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_audio.read())
        transcribed = whisper_transcribe("uploaded_audio.wav")
        st.success("Transcribed Text:")
        st.write(transcribed)
    except Exception as e:
        st.error(f"Audio processing failed: {e}")

# YouTube summarization
st.header("📺 YouTube Video")
youtube_url = st.text_input("Enter YouTube URL")
if st.button("Summarize YouTube Video"):
    if youtube_url:
        try:
            audio_path = download_youtube_audio(youtube_url)
            if audio_path:
                transcribed = whisper_transcribe(audio_path)
                st.success("Transcript:")
                st.write(transcribed)
                summary = bart_summarize(transcribed)
                st.success("Summary:")
                st.write(summary)
        except Exception as e:
            st.error(f"YouTube processing failed: {e}")
    else:
        st.warning("Please enter a YouTube URL.")

# Local video upload
st.header("🎥 Upload Local Video")
video_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])
if video_file is not None:
    try:
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())
        audio_path = video_to_audio("uploaded_video.mp4")
        transcribed = whisper_transcribe(audio_path)
        st.success("Transcript:")
        st.write(transcribed)
        summary = bart_summarize(transcribed)
        st.success("Summary:")
        st.write(summary)
    except Exception as e:
        st.error(f"Video processing failed: {e}")

# Instructions
st.caption("Run this app using: `streamlit run notesbuddy.py`")
