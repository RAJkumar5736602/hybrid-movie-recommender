# Hybrid Movie Recommender (Content-Based + Collaborative Filtering)

Top-10 movie recommender. It blends **content-based filtering** (TF-IDF over genres + cosine similarity) with **user-based collaborative filtering** (cosine similarity between users on a user-item rating matrix), using MovieLens data.

*MSc Data Science & AI coursework, Bournemouth University (2025). Individual project.*

## Problem
Streaming catalogues are huge, and users waste time choosing what to watch. A hybrid recommender uses what a movie *is* (its genres) and what *similar users* liked, so it gives better suggestions than either method alone.

## Data
- **MovieLens 20M** (GroupLens): `movies.csv` (27,278 movies), `tags.csv` (465,564 tags) and a subset of `ratings.csv`. The subset is the first 1,048,575 ratings, covering 7,120 users and 14,026 rated movies.
- The data is **not included** because the MovieLens licence doesn't allow redistribution. See `data/README.md` to download it and recreate the files.

## Approach
1. **Content-based (CBF):** fill missing genres, build a TF-IDF matrix (27,278 × 23 genre terms), and compute cosine similarity between movies. Take the top 20 neighbours of the seed movie.
2. **Collaborative (CF):** pivot ratings into a 7,120 × 14,026 user-item matrix, compute user-user cosine similarity, score each movie as the similarity-weighted sum of ratings, and mask movies the user has already rated.
3. **Hybrid:** average the CBF and CF scores for every candidate and return the top 10.
4. Tags are also merged into a `full_metadata` field (genres + user tags) for future content features.

## Results
There is no offline ranking metric yet (no RMSE, Precision@K or NDCG). The model was checked qualitatively.
Example: `user 552`, seed movie `592` (*Batman*, 1989), output in `src/recommender.py`:

> Twelve Monkeys · Star Wars IV · Forrest Gump · Speed · Schindler's List · Batman · Silence of the Lambs · Independence Day · Star Wars V · The Matrix

The full pipeline runs in seconds on vectorised NumPy and scikit-learn operations.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put my_movies.csv, my_ratings.csv, my_tag.csv into data/ (see data/README.md)
python src/recommender.py --data-dir data --user 552 --movie 592
```
`notebooks/movie_recommendation_original.py` is the original Colab version. It builds the full 27k × 27k similarity matrix, which needs about 6 GB of RAM. `src/recommender.py` gives the same output but computes similarity rows on demand.

## Known limitations and next steps
- **Score scaling:** raw CF scores (up to about 1,000+) are averaged with cosine scores (0–1), so CF dominates and the output leans towards popular titles. Next step: min-max normalise both scores and use a weighted blend (α).
- The seed movie itself can show up in the results. It should be excluded.
- There is an off-by-one between the matrix row position and `userId` in the CF step (kept as-is to match the coursework, and flagged in the code).
- Add offline evaluation (train/test split by time, Precision@10, NDCG@10, RMSE) and try matrix factorisation (SVD/ALS) or neural CF.
- Serve it as an API (FastAPI) with precomputed item embeddings.

## Tech stack
Python · pandas · NumPy · scikit-learn (TfidfVectorizer, cosine_similarity) · Google Colab
