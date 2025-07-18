import spacy
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
import os
# === Gemini / Gemma-3-12b-it ===

load_dotenv()

api_key =  os.getenv("GEMINI_API_KEY")
generative_model = "gemini-1.5-flash"
genai.configure(api_key=api_key)
model = genai.GenerativeModel(generative_model)
print(generative_model + " model loaded")

# === NLP / SpaCy ===
# spacy_model = "en_core_web_trf"
# nlp = spacy.load(spacy_model)
# stop_words = set(stopwords.words("english"))
# lemmatizer = WordNetLemmatizer()
# print(spacy_model + " model loaded")