import spacy
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Gemini / Gemma-3-12b-it ===
api_key = "AIzaSyByTfCk2a6m4gkeJAuCpWGmWi8qfyHBQ3w"
generative_model = "gemma-3-12b-it"
genai.configure(api_key=api_key)
model = genai.GenerativeModel(generative_model)
print(generative_model + " model loaded")

# === NLP / SpaCy ===
spacy_model = "en_core_web_trf"
nlp = spacy.load(spacy_model)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
print(spacy_model + " model loaded")