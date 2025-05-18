import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)
def load_models():
    intent_classifier = pipeline("zero-shot-classification",
                                 model="facebook/bart-large-mnli")
    ner_tagger = pipeline("ner", grouped_entities=True)
    return intent_classifier, ner_tagger

def detect_intent(text, intent_classifier, candidate_intents):
    result = intent_classifier(text, candidate_intents)
    return result['labels'][0], result['scores'][0]

def detect_entities(text, ner_tagger, indicators, companies):
    entities = ner_tagger(text)
    orgs = [ent['word'] for ent in entities if ent['entity_group'] == 'ORG']
    indicators_found = [ind for ind in indicators if ind in text.lower()]
    companies_found = [comp for comp in companies if comp in text.lower()]
    return orgs + companies_found, indicators_found

def main():
    st.title("Finance Fian - NLP Intent & Entity Detector")

    intent_classifier, ner_tagger = load_models()

    candidate_intents = [
        "greet",
        "goodbye",
        "ask_stock_price",
        "ask_indicator",
        "ask_sentiment",
        "ask_general_info"
    ]

    INDICATORS = ["sma", "rsi", "macd", "bollinger", "moving average"]
    COMPANIES = ["apple", "tesla", "microsoft", "google", "amazon"]

    user_input = st.text_input("Ask Finance Fian anything:")

    if user_input:
        intent, confidence = detect_intent(user_input, intent_classifier, candidate_intents)
        entities, indicators = detect_entities(user_input, ner_tagger, INDICATORS, COMPANIES)

        st.markdown(f"**Detected Intent:** {intent} (confidence: {confidence:.2f})")
        st.markdown(f"**Detected Companies:** {entities if entities else 'None'}")
        st.markdown(f"**Detected Indicators:** {indicators if indicators else 'None'}")

        if intent == "ask_stock_price" and entities:
            st.success(f"Fetching stock price for {entities[0].title()}...")
        elif intent == "ask_indicator" and entities and indicators:
            st.success(f"Calculating {', '.join(indicators)} for {entities[0].title()}...")
        elif intent == "greet":
            st.info("Hello! How can I assist with stocks today?")
        elif intent == "goodbye":
            st.info("Goodbye! Have a great day!")
        else:
            st.warning("Sorry, I didn't quite get that. Could you please rephrase?")

if __name__ == "__main__":
    main()
