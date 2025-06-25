
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Bian.configs import indicator_plot_config
from Bian.resources import nlp, stop_words, lemmatizer

# import libs
import yfinance as yf
import re



def extract_data_yf(tickers, Period="1y"):  # Backend 
    # Note: Remember to make a way to delete these files after use. Since they are only temporary files.
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, period=Period, interval="1d", auto_adjust=True, progress=False)
        filename = f"temp_{ticker}_{Period}.csv"
        df.to_csv(filename)
        data[ticker] = df
    return data




# ====================================================================================================
# Section: Natual Language Understanding (NLU)
# ====================================================================================================

## == == == -- -- -- Helper Functions -- -- -- == == == ##

def preprocess_query(text): # Backend
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


def run_NER(text): # Backend
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_entities(entities, label): # Backend
    return [ent_text for ent_text, ent_label in entities if ent_label == label]

def to_yf_period(text): # Backend
    match = re.search(r'(\d+)\s*(year|years|month|months|week|weeks|day|days)', text, re.IGNORECASE)
    if match:
        number = match.group(1)
        unit = match.group(2).lower()

        if 'year' in unit:
            return f"{number}y"
        elif 'month' in unit:
            return f"{number}mo"
        elif 'week' in unit:
            days = int(number) * 7
            return f"{days}d"
        elif 'day' in unit:
            return f"{number}d"
    return None

def yfinance_search_company(company_names): # Backend
    results = {}
    for name in company_names:
        s = yf.Search(name, max_results=1)
        if s.quotes:
            results[name] = s.quotes[0].get("symbol")
        else:
            results[name] = None
    # Return a list of ticker symbols (filtering out any None values)
    return [ticker for ticker in results.values() if ticker]



