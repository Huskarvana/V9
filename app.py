
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import random
import transformers
import torch
from transformers import pipeline
import feedparser

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille DS Automobiles", layout="wide")
st.title("🚗 Agent de Veille – DS Automobiles (APIs multiples)")

API_KEY_NEWSDATA = st.secrets["API_KEY_NEWSDATA"]
MEDIASTACK_API_KEY = st.secrets["MEDIASTACK_API_KEY"]

NEWSDATA_URL = "https://newsdata.io/api/1/news"
MEDIASTACK_URL = "http://api.mediastack.com/v1/news"

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=DS+Automobiles&hl=fr&gl=FR&ceid=FR:fr",
    "https://www.leblogauto.com/feed"
]

MODELES_DS = ["DS N4", "DS N8", "DS7", "DS3", "DS9", "DS4", "Jules Verne", "N°4", "N°8"]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analyzer = get_sentiment_pipeline()

# --- FONCTIONS DE COLLECTE ---
def fetch_newsdata_articles(query, max_results=5, lang=None, country=None):
    params = {"apikey": API_KEY_NEWSDATA, "q": query, "language": lang or "fr"}
    try:
        response = requests.get(NEWSDATA_URL, params=params)
        data = response.json()
        return [{
            "date": item.get("pubDate", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source_id", ""),
            "lien": item.get("link", "")
        } for item in data.get("results", [])[:max_results]]
    except:
        return []

def fetch_mediastack_articles(query, max_results=5, lang=None, country=None):
    params = {
        "access_key": MEDIASTACK_API_KEY,
        "keywords": query,
        "languages": lang or "fr"
    }
    if country and country.lower() != "all":
        params["countries"] = country.lower()
    try:
        response = requests.get(MEDIASTACK_URL, params=params)
        data = response.json()
        return [{
            "date": item.get("published_at", ""),
            "titre": item.get("title", ""),
            "contenu": item.get("description", ""),
            "source": item.get("source", ""),
            "lien": item.get("url", "")
        } for item in data.get("data", [])[:max_results]]
    except:
        return []

def fetch_rss_articles(query, max_results=5):
    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_results]:
                articles.append({
                    "date": entry.get("published", ""),
                    "titre": entry.get("title", ""),
                    "contenu": entry.get("summary", ""),
                    "source": feed_url,
                    "lien": entry.get("link", "")
                })
        except:
            pass
    return articles

def detecter_modele(titre):
    for m in MODELES_DS:
        if m.lower() in titre.lower():
            return m
    return "DS Global"

def analyser_article(row):
    try:
        sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        prediction = sentiment_analyzer(row['contenu'][:512])[0]
        sentiment = sentiment_map.get(prediction['label'], "Neutral")
    except:
        sentiment = "Neutral"
    modele = detecter_modele(row['titre'])
    résumé = row['contenu'][:200] + "..."
    return pd.Series({'résumé': résumé, 'ton': sentiment, 'modèle': modele})

# --- INTERFACE UTILISATEUR ---
nb_articles = st.slider("Nombre d'articles par source", 5, 50, 10)
filtre_modele = st.selectbox("Filtrer par modèle", ["Tous"] + MODELES_DS)
filtre_ton = st.selectbox("Filtrer par ton", ["Tous", "Positive", "Neutral", "Negative"])
langue = st.selectbox("Langue", ["fr", "en", "de", "it", "es", "pt", "all"])
pays = st.selectbox("Pays (optionnel)", ["all", "fr", "us", "gb", "de", "it", "es"])

if st.button("🔍 Lancer la veille"):
    newsdata = fetch_newsdata_articles("DS Automobiles", nb_articles, lang=None if langue=="all" else langue, country=None if pays=="all" else pays)
    mediastack = fetch_mediastack_articles("DS Automobiles", nb_articles, lang=None if langue=="all" else langue, country=None if pays=="all" else pays)
    rss = fetch_rss_articles("DS Automobiles", nb_articles)

    articles = pd.DataFrame(newsdata + mediastack + rss)

    if not articles.empty:
        with st.spinner("Analyse en cours..."):
            articles[['résumé', 'ton', 'modèle']] = articles.apply(analyser_article, axis=1)

        # Trier du plus récent au plus ancien
        articles['date'] = pd.to_datetime(articles['date'], errors='coerce')
        articles = articles.sort_values(by='date', ascending=False)

        # Appliquer les filtres
        if filtre_modele != "Tous":
            articles = articles[articles['modèle'] == filtre_modele]
        if filtre_ton != "Tous":
            articles = articles[articles['ton'] == filtre_ton]

        st.dataframe(articles[['date', 'titre', 'modèle', 'ton', 'résumé', 'source', 'lien']])
    else:
        st.warning("Aucun article trouvé.")
