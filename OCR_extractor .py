import streamlit as st
import openai
import pytesseract
from PIL import Image
import os
from dotenv import load_dotenv
import base64
import io
import csv
import json
import torch
import numpy as np
from langdetect import detect
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import sqlite3
import unicodedata
from datetime import datetime
import regex
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers.json import JsonOutputParser

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Ge'ez dictionary from a JSON file
try:
    with open("geez_dict.json", "r", encoding="utf-8") as f:
        geez_dict = json.load(f)
except FileNotFoundError:
    geez_dict = {}

st.set_page_config(page_title="Ancient & Modern OCR", layout="wide")
st.sidebar.title("🔧 Settings")

# Auto-detect script option
auto_detect_script = st.sidebar.checkbox("🧠 Auto-detect Script Type", value=True)

# Default or manual script selection (if auto-detect is off)
manual_script_type = st.sidebar.radio("🧬 Script Type", ['Modern (OCR)', 'Ancient (GPT-4 Vision)'])

# Confidence threshold
min_confidence = st.sidebar.slider("🧐 Min OCR Confidence", 0, 100, 50)

# Translation options
translate = st.sidebar.checkbox("🌐 Translate OCR text?")
available_langs = ['en', 'fr', 'es', 'de', 'it', 'pt', 'ar', 'zh', 'hi', 'am']
target_lang = st.sidebar.selectbox("📥 Translate To (Auto overrides if language is detected)", available_langs, index=1)

# Optional GPT-4 prompt
user_prompt = st.sidebar.text_area("✍️ GPT-4 Prompt (for Ancient Scripts)", height=140, value="This image contains Geʿez or another ancient script. Please interpret or describe the content.")

# Ge'ez dictionary editing
if st.sidebar.checkbox("✏️ Edit Geʽez Dictionary"):
    new_char = st.sidebar.text_input("Character")
    new_translit = st.sidebar.text_input("Transliteration")
    new_meaning = st.sidebar.text_input("Meaning")
    if st.sidebar.button("Add to Dictionary"):
        geez_dict[new_char] = {"transliteration": new_translit, "meaning": new_meaning}
        with open("geez_dict.json", "w", encoding="utf-8") as f:
            json.dump(geez_dict, f, ensure_ascii=False, indent=2)
        st.sidebar.success(f"Added {new_char} to dictionary.")

# Ge'ez dictionary search
if st.sidebar.checkbox("🔍 Search Geʽez Dictionary"):
    search_query = st.sidebar.text_input("🔎 Search character or transliteration")
    if search_query:
        st.sidebar.markdown("### Search Results")
        for char, data in geez_dict.items():
            if search_query in char or search_query in data.get("transliteration", ""):
                st.sidebar.markdown(f"**{char}** → {data.get('transliteration')} — {data.get('meaning')}")

# Upload image
st.title("📜 Multilingual OCR + Ancient Script Reader")
uploaded_files = st.file_uploader("📤 Upload image(s) (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Manual correction
st.sidebar.title("🛠 Manual Correction")
manual_text_input = st.sidebar.text_area("✏️ Manually correct OCR output")

# Database setup
conn = sqlite3.connect("ocr_results.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        filename TEXT, 
        raw_text TEXT, 
        translated_text TEXT, 
        transliteration TEXT, 
        meaning TEXT,
        timestamp TEXT
    )
""")
conn.commit()

# Helper: Check if Ge'ez script is likely present
def contains_geez(text):
    for char in text:
        try:
            if 'ETHIOPIC' in unicodedata.name(char):
                return True
        except ValueError:
            continue
    return False

def detect_script(text):
    if regex.search(r'\p{Script=Ethiopic}', text):
        return "Ethiopic"
    return "Unknown"

# GPT-4 Vision Interaction using updated OpenAI SDK
def call_gpt4_vision(prompt, image):
    base64_image = base64.b64encode(image).decode('utf-8')
    try:
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ GPT-4 Vision Error: {str(e)}")
        return ""

# OCR Extraction using pytesseract
def extract_text(image):
    gray = image.convert('L')  # convert to grayscale for better OCR
    return pytesseract.image_to_string(gray, lang='eng')

# Translation using OpenAI
def translate_text(text, target_lang):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Translate this text to {target_lang}: {text}"}],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"❌ Translation Error: {str(e)}")
        return ""

# Entity extraction for scientific texts
def extract_entities(text):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    template = PromptTemplate(
        input_variables=["text"],
        template="""Extract key entities (names, dates, terms, etc.) from the scientific passage:

        {text}

        Return in structured JSON format."""
    )
    chain = LLMChain(llm=llm, prompt=template)
    return chain.run(text)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_bytes = uploaded_file.read()

        if auto_detect_script:
            extracted_text = extract_text(image)
        else:
            if manual_script_type == 'Modern (OCR)':
                extracted_text = extract_text(image)
            else:
                extracted_text = call_gpt4_vision(user_prompt, image_bytes)

        if manual_text_input:
            extracted_text = manual_text_input

        st.subheader(f"📄 OCR Result for: {uploaded_file.name}")
        st.text_area("📝 Extracted Text", value=extracted_text, height=200)

        if translate:
            translated = translate_text(extracted_text, target_lang)
            st.text_area("🌍 Translated Text", value=translated, height=200)
        else:
            translated = ""

        if contains_geez(extracted_text):
            st.markdown("### Geʽez Dictionary Lookup")
            for char in extracted_text:
                if char in geez_dict:
                    info = geez_dict[char]
                    st.write(f"{char}: {info.get('transliteration')} — {info.get('meaning')}")

        entities = extract_entities(extracted_text)
        st.markdown("### 🔍 Extracted Entities")
        st.json(entities)

        cursor.execute("""
            INSERT INTO results (filename, raw_text, translated_text, transliteration, meaning, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            uploaded_file.name, extracted_text, translated, "", "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

conn.close()
