
import streamlit as st
import requests
import pandas as pd
import feedparser
from datetime import datetime
from transformers import pipeline

# --- CONFIGURATION ---
st.set_page_config(page_title="Veille DS Automobiles", layout="wide")
st.title("Agent de Veille – DS Automobiles (APIs multiples)")

API_KEY_NEWSDATA = "pub_afd14bba24914ca9a0c8b7d23a0fb755"
MEDIASTACK_API_KEY = "eaa929770f885d5e15b62d7d16a521d8"
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"

MODELES_DS = ["DS N4", "DS N8", "DS7", "DS3", "DS9", "DS4", "Jules Verne", "N°8", "N°4"]

@st.cache_resource
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analyzer = get_sentiment_pipeline()

def fetch_newsdata_articles(query, max_results=5):
    url = "https://newsdata.io/api/1/news"
    params = {"apikey": API_KEY_NEWSDATA, "q": query, "language": "fr"}
    response = requests.get(url, params=params)
    data = response.json()
    return [{
        "date": item.get("pubDate", ""),
        "titre": item.get("title", ""),
        "contenu": item.get("description", ""),
        "source": item.get("source_id", ""),
        "lien": item.get("link", "")
    } for item in data.get("results", [])[:max_results]]

def fetch_mediastack_articles(query, max_results=5):
    url = "http://api.mediastack.com/v1/news"
    params = {"access_key": MEDIASTACK_API_KEY, "keywords": query, "languages": "fr"}
    response = requests.get(url, params=params)
    data = response.json()
    return [{
        "date": item.get("published_at", ""),
        "titre": item.get("title", ""),
        "contenu": item.get("description", ""),
        "source": item.get("source", ""),
        "lien": item.get("url", "")
    } for item in data.get("data", [])[:max_results]]

def fetch_rss_articles(query):
    feeds = [
        "https://news.google.com/rss/search?q=DS+Automobiles&hl=fr&gl=FR&ceid=FR:fr",
        "https://www.leblogauto.com/feed"
    ]
    articles = []
    for url in feeds:
        d = feedparser.parse(url)
        for entry in d.entries:
            articles.append({
                "date": entry.get("published", ""),
                "titre": entry.get("title", ""),
                "contenu": entry.get("summary", ""),
                "source": "RSS",
                "lien": entry.get("link", "")
            })
    return articles

def detecter_modele(titre):
    for m in MODELES_DS:
        if m.lower() in titre.lower():
            return m
    return "DS Global"

def analyser_article(row):
    try:
        sentiment = sentiment_analyzer(row['contenu'][:512])[0]['label']
        label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
        sentiment = label_map.get(sentiment, "Neutral")
    except:
        sentiment = "Neutral"
    modele = detecter_modele(row['titre'])
    résumé = row['contenu'][:200] + "..."
    return pd.Series({'résumé': résumé, 'ton': sentiment, 'modèle': modele})

def envoyer_notif_slack(article):
    try:
        message = (
            "Nouvel article détecté sur *{}*
"
            "*{}*
"
            "_Ton: {}_
"
            "<{}|Lire l'article>".format(article['modèle'], article['titre'], article['ton'], article['lien'])
        )
        payload = {"text": message}
        requests.post(SLACK_WEBHOOK_URL, json=payload)
    except:
        pass

# --- INTERFACE ---
nb_articles = st.slider("Nombre d'articles à récupérer (par source)", 5, 30, 10)

if st.button("Lancer la veille"):
    data = []
    data.extend(fetch_newsdata_articles("DS Automobiles", nb_articles))
    data.extend(fetch_mediastack_articles("DS Automobiles", nb_articles))
    data.extend(fetch_rss_articles("DS Automobiles"))

    articles = pd.DataFrame(data)

    if not articles.empty:
        with st.spinner("Analyse en cours..."):
            articles[['résumé', 'ton', 'modèle']] = articles.apply(analyser_article, axis=1)

        articles['date'] = pd.to_datetime(articles['date'], errors='coerce')
        articles = articles.sort_values(by='date', ascending=False)

        for _, row in articles.iterrows():
            envoyer_notif_slack(row)

        st.dataframe(articles[['date', 'titre', 'modèle', 'ton', 'résumé', 'source', 'lien']])
    else:
        st.warning("Aucun article trouvé.")
