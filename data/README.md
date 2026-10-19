Data is not committed (MovieLens licence does not allow redistribution).

1. Download **MovieLens 20M** from https://grouplens.org/datasets/movielens/20m/
2. Copy `movies.csv` -> `my_movies.csv`, `tags.csv` -> `my_tag.csv`, `ratings.csv` -> `my_ratings.csv` into this folder.
3. The coursework used only the first 1,048,575 rows of ratings.csv (users 1-7,120). To match it:
   `head -n 1048576 ratings.csv > my_ratings.csv`
