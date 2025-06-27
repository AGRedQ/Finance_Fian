import spacy
import google.generativeai as genai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import dotenv
# === Gemini / Gemma-3-12b-it ===

api_key =  dotenv.load_dotenv().get("GEMINI_API_KEY")
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