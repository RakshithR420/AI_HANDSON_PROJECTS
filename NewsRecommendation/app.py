
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv('./data_set/news.tsv' , sep ='\t')
df.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']

df["data"] = df["Category"] + " " + df["SubCategory"]+ " " + df["Title"].fillna("")

tfid_vectorizer = TfidfVectorizer(stop_words="english" , max_features=5000)
tfidf_matrix = tfid_vectorizer.fit_transform(df["data"])

kmeans = KMeans(n_clusters=191 , random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
df["Cluster"] = clusters


def recommend_new_news(news_id, num_recommendations=5):
    selected_articles = (df.loc[df["NewsID"] == news_id])
    cluster_label = df.loc[df["NewsID"] == news_id, "Cluster"].values[0]
    similar_articles = df[(df["Cluster"] == cluster_label) & (df["NewsID"] != news_id)]
    recommendations = similar_articles.sample(
        num_recommendations,
        replace= False
    )
    return {
        "recommendations": recommendations.to_dict(orient="records"),
        "selected_article": selected_articles.to_dict(orient="records"),
    }


app = Flask(__name__)
CORS(app)

@app.route("/search", methods=["POST"])
def search_news():
    query = request.json["query"]
    tfidf_query = tfid_vectorizer.transform([query])
    cluster_label = kmeans.predict(tfidf_query)[0]
    similar_articles = df[df["Cluster"] == cluster_label]
    result = similar_articles.sample(5)
    return jsonify(result.to_dict(orient="records"))

@app.route("/recommend", methods=["POST"])
def recommend_news():
    data = request.get_json()
    news_id = data.get("news_id")
    num_recommendations = int(data.get("num_recommendations",5))
    result = recommend_new_news(news_id,num_recommendations)
    
    return jsonify({
        "status": "success",
        "data": {
            "recommendation": result["recommendations"],
            "selected_article": result["selected_article"]
        }
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
