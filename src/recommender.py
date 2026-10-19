"""Hybrid movie recommender (content-based + user-based collaborative filtering).

Same logic as notebooks/movie_recommendation_original.py, but similarity rows are
computed on demand instead of materialising the full 27,278 x 27,278 matrix, so it
runs on a laptop (~2 GB RAM for the user-item matrix).

Usage:
    python src/recommender.py --data-dir data --user 552 --movie 592
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def load(data_dir: str):
    movies = pd.read_csv(os.path.join(data_dir, "my_movies.csv"), encoding="utf-8-sig")
    ratings = pd.read_csv(os.path.join(data_dir, "my_ratings.csv"), encoding="utf-8-sig")
    tags = pd.read_csv(os.path.join(data_dir, "my_tag.csv"), encoding="utf-8-sig")
    tags["tag"] = tags["tag"].astype(str)
    merged = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(map(str, x))).reset_index()
    movies = movies.merge(merged, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["full_metadata"] = movies["genres"] + " " + movies["tag"]
    return movies, ratings


class HybridRecommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, text_col: str = "genres"):
        self.movies = movies
        # Content-based: TF-IDF over genres (the coursework used genres only)
        self.tfidf = TfidfVectorizer(stop_words="english").fit_transform(movies[text_col])
        # Collaborative: dense user-item matrix (users x movies), 0 = not rated
        self.ui = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0).astype(np.float32)
        self.ui_norm = normalize(self.ui.values)

    def genre_similarity(self, movie_id: int, k: int = 20) -> dict:
        idx = self.movies.index[self.movies["movieId"] == movie_id][0]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).ravel()
        order = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[1 : k + 1]
        return {self.movies.iloc[i]["movieId"]: s for i, s in order}

    def user_predictions(self, user_id: int) -> dict:
        # Faithful to the coursework code: the similarity row is taken by *position*
        # (user_similarity_scores[user_id]) while "already rated" is looked up by *userId label*
        # (.loc[user_id]). This is an off-by-one quirk - see README "Known limitations".
        sim_row = self.ui_norm[user_id] @ self.ui_norm.T
        scores = sim_row @ self.ui.values
        already = self.ui.loc[user_id].values > 0
        scores[already] = 0
        return dict(zip(self.ui.columns, scores))

    def recommend(self, user_id: int, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        g = self.genre_similarity(movie_id)
        u = self.user_predictions(user_id)
        hybrid = {m: (g.get(m, 0) + u.get(m, 0)) / 2 for m in set(g) | set(u)}
        top = [m for m, _ in sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        return self.movies[self.movies["movieId"].isin(top)][["movieId", "title"]]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--user", type=int, default=552)
    p.add_argument("--movie", type=int, default=592)
    a = p.parse_args()
    movies, ratings = load(a.data_dir)
    rec = HybridRecommender(movies, ratings)
    print("User-item matrix:", rec.ui.shape, "| Genre TF-IDF:", rec.tfidf.shape)
    print(rec.recommend(a.user, a.movie))
