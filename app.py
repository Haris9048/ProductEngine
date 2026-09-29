import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Beauty Product Finder", layout="wide")
st.title("Beauty Product Intelligence")

@st.cache_resource
def load_everything():
    products = pd.read_parquet("products_slim.parquet")
    stats = pd.read_parquet("stats_slim.parquet")
    aspects = pd.read_parquet("aspect_analysis.parquet")
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(products["search_text"].fillna(""))
    return products, stats, aspects, tfidf, matrix

products, stats, aspects, tfidf, matrix = load_everything()
scores = stats.set_index("parent_asin")["weighted_score"]

tab1, tab2, tab3 = st.tabs(["Search & Similar", "Top Products", "Aspects"])

with tab1:
    query = st.text_input("Describe what you want", "gentle moisturizing lotion for sensitive skin")
    if query:
        sims = cosine_similarity(tfidf.transform([query]), matrix).flatten()
        idx = sims.argsort()[::-1][:10]
        res = products.iloc[idx][["parent_asin", "title"]].copy()
        res["match_score"] = sims[idx].round(3)
        st.dataframe(res, use_container_width=True)

        pick = st.selectbox("Find products similar to:", res["title"].tolist())
        row = res[res["title"] == pick].index[0]
        sim2 = cosine_similarity(matrix[row], matrix).flatten()
        idx2 = sim2.argsort()[::-1][1:21]
        rec = products.iloc[idx2][["parent_asin", "title"]].copy()
        rec["similarity"] = sim2[idx2]
        rec["quality"] = rec["parent_asin"].map(scores).fillna(scores.mean())
        rec["hybrid"] = (0.6 * rec["similarity"] / rec["similarity"].max()
                         + 0.4 * rec["quality"] / rec["quality"].max())
        st.subheader("Recommended")
        st.dataframe(rec.sort_values("hybrid", ascending=False).head(5)[["title", "hybrid"]],
                     use_container_width=True)

with tab2:
    min_reviews = st.slider("Minimum reviews", 1, 500, 50)
    top = stats[stats["review_count"] >= min_reviews].sort_values("weighted_score", ascending=False).head(20)
    top = top.merge(products[["parent_asin", "title"]], on="parent_asin")
    st.dataframe(top[["title", "review_count", "average_rating", "weighted_score"]], use_container_width=True)

with tab3:
    st.bar_chart(aspects.set_index("aspect")[["positive_pct", "negative_pct"]])