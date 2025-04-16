import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from transformers import pipeline
import speech_recognition as sr

# Download NLTK data
nltk.download("punkt")

# Extractive Summarization (NLTK)
def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    word_frequencies = Counter(word_tokenize(text.lower()))
    ranked_sentences = sorted(sentences, key=lambda s: sum(word_frequencies[w] for w in word_tokenize(s.lower())), reverse=True)
    return " ".join(ranked_sentences[:num_sentences])

# AI-Based Summarization (Using smaller BART model: facebook/bart-base)
summarizer = pipeline("summarization", model="facebook/bart-base")

def bart_summarize(text):
    return summarizer(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

# Speech-to-Text
def audio_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

# Streamlit UI
st.title("Smart Notes Summarizer")
st.write("Summarize your notes quickly using NLP and AI!")

# Text input
user_input = st.text_area("Enter your notes here")
summary_method = st.selectbox("Choose summarization method", ["Extractive (NLTK)", "AI-Based (BART)"])

if st.button("Summarize"):
    if user_input:
        if summary_method == "Extractive (NLTK)":
            summary = summarize_text(user_input)
        else:
            summary = bart_summarize(user_input)
        
        st.write("### Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Audio File Upload
uploaded_file = st.file_uploader("Upload an audio file (WAV format) for Speech-to-Text", type=["wav"])
if uploaded_file is not None:
    try:
        text_from_audio = audio_to_text(uploaded_file)
        st.write("### Transcribed Text:")
        st.write(text_from_audio)
    except Exception as e:
        st.error(f"Error processing audio: {e}")

# Run with: streamlit run app.py

#python -m streamlit run "test4.py"
st.caption("Run this app using: streamlit run notesbuddy.py")
# python -m streamlit run "C:\Users\KOUSHIK\OneDrive\Desktop\files\predictioneer\notesbuddy.py"
