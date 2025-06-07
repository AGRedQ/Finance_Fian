import spacy
import spacy_transformers

print("spaCy loaded.")
print("Transformers module:", spacy_transformers)
nlp = spacy.load("en_core_web_trf")
print("Transformer model loaded successfully.")
