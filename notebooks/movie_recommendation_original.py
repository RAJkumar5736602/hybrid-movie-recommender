# Original Google Colab export of the coursework notebook (Colab link removed).
# Loads the full 27k x 27k genre-similarity matrix (~6 GB RAM). See src/recommender.py for a memory-light version.



"""### Section 1: In this block we are importing panadas library. It can read CSV file dataframe format, and it helps filtering,preprocessing and sorting. so very useful to our  movie recommendation system."""

#importing pandas library for read csv file and handle data and table format.
import pandas as pd

"""Section 2: In this block we are reading'my_movies.csv' data set file by using pandas and converting CSV file format in to dataframe, by using head() function we are display top 5 rows.and it helps to see hows the structure and helps to find what are the columns are availble."""

#By using pandas we are reading movies dataset from 'my_movies.csv file.
#we are displaying first 5 rows of dataset for to check the data structure.

movies_df = pd.read_csv('my_movies.csv')
movies_df.head()

"""### Section 3: Here we use the pandas library to read the file'my_ratings.csv'.  Information such as user ID, movie ID, rating, and timestamp are included in this collection.  It provides information according to the user's rating of the film.  To see the dataset's structure, we utilize the head() function."""

# We read the 'my_ratings.csv' file with pandas – it contains data like user ID, movie ID, rating, and timestamp
# We use the head() function to display the first 5 rows of the dataset

ratings_df = pd.read_csv('my_ratings.csv')
ratings_df.head()

"""Section 4: Loading the Tags Dataset.

 In this section, we will use the pandas library to read the file'my_tag.csv'.  This dataset comprises the tags that users assigned to each movie.  This gives semantic information about the movie.  We utilize the head() function to display the dataset's first rows.  This enables us to grasp the structure.
"""

#We use pandas to read the'my_tag.csv' file, which contains the user's movie tags.
#To preview the dataset, we display the first 5 rows using head().

tag_df = pd.read_csv('my_tag.csv')
tag_df.head()

"""### section 5:Loading Three Datasets and Verifying the Structure

In this section, we will use the pandas library to read three key CSV files:'my_ratings.csv','my_movies.csv', and'my_tag.csv'. Then, we use the 'print()' and 'head()' functions to see if they loaded properly and what their structure looks like. This is a data validation step before we begin preprocessing.
"""

# Using pandas to read ratings, movies, and tags datasets.
# Using head() to display leading rows in the ratings and movies dataset.
# Printing the complete tags dataset to underunderstand its content.

ratings_df = pd.read_csv('my_ratings.csv')
movies_df =pd.read_csv('my_movies.csv')
tag_df =pd.read_csv('my_tag.csv')

print(ratings_df.head())
print(movies_df.head())
print(tag_df)

import pandas as pd
movies_df = pd.read_csv('my_movies.csv')

"""Section 6: Content-Based Filtering: Genre-Based Similarity Calculation.

In this part, we can compare the similarities of movies based on their genres. First, we use pandas to load the file'my_movies.csv'. Next, we fill in the genres column with NULL entries.


The genres are then converted to TF-IDF format utilizing technology known as 'TfidfVectorizer'. This will enable us to process text data more efficiently. We use 'cosine_similarity' to calculate the similarity of movies based on their TF-IDF values. This matrix will be used to build the CBF (Content-Based Filtering) model.

"""

#To replace missing entries in the genres column, use fillna('') to create an empty string.
movies_df['genres'] = movies_df['genres'].fillna('')

# Convert genre text to numerical format by importing TfidfVectorizer.
from sklearn.feature_extraction.text import TfidfVectorizer

# Remove stop words and generate TF-IDF vectors from genres.
GenreTextVectorizer = TfidfVectorizer(stop_words='english')
genre_tfidf_values= GenreTextVectorizer.fit_transform(movies_df['genres'])


# Import and calculate Cosine similarity between TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity

# Create a content-based similarity matrix to compare movie genres.
cbf_score_map = cosine_similarity(genre_tfidf_values)

# Print matrix shape to show how large the data size is.
print("Cosine Matrix Shape:",cbf_score_map.shape)

#To get image details from the file'my_movies.csv'
import pandas as pd

movies_dataset = pd.read_csv('my_movies.csv')

"""### Section 7 :
In this section, we merge movie-oda genres and user tag data to generate a complete metadata column. This is a critical step for content-based filtering.

To begin, we convert the tag values in the'my_tag.csv' file to strings. Next, we group and connect the tags for each movie-Id. Next, we incorporate the tag information into the'my_movies.csv' dataset.

To handle missing values in the 'genres' and 'tag' columns, we use 'fillna('')'.

After that, we merge the genres and tags to form a 'full_metadata' column. This generates an enriched textual data source, which can then be used in TF-IDF to calculate similarity.
"""

# Convert the tag details from the file'my_tag.csv' into the suitable format.
tag_data = pd.read_csv('my_tag.csv')
tag_data['tag'] = tag_data ['tag'].astype(str)

# Combine all tags linked with each movieId together.
mergerd_tag = tag_data.groupby('movieId')['tag'].apply(lambda x: ''.join(x)).reset_index()

# To ensure safety, read and tag the dataset again.
import pandas as pd

movie_dataset = pd.read_csv('my_movies.csv')
tag_entries = pd.read_csv('my_tag.csv')


# Convert the tag entry to a string: tag_entries['tag']
tag_entries['tag'] = tag_entries['tag'].astype(str)


# List the tags for each movie (movieId)
mergerd_tag_list = tag_entries.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()


# Merge the two databases based on movieId (a left join).
movie_dataset = pd.merge(movie_dataset, mergerd_tag_list, on='movieId', how='left')


# Fill in empty values ('') in genre and tag columns.
movie_dataset['tag'] = movie_dataset['tag'].fillna('')
movie_dataset['genres'] = movie_dataset['genres'].fillna('')

# Combine the genres and tag columns to form a new 'full_metadata' column.
movie_dataset['full_metadata'] = movie_dataset['genres'] + ' ' + movie_dataset['tag']


# Show the title and entire metadata for the first 5 rows:
print(movie_dataset[['title', 'full_metadata']].head())

"""### Section 8:
In this step, we will generate a user-item interaction matrix using the user and ratings from the my_ratings.csv file. It will be structured similarly to a pivot table, with userId as the row and movieId as the column. We will enter the rating values.

The user's similarity score will then be calculated using the cosine_similarity technique. This will assist us in identifying similar users.

We will print the shape of this similarity matrix and determine how many rows and columns it has.
"""

# To import the cosine_similarity function from sklearn.metrics.
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Import the my_ratings.csv dataset using pandas.
user_reviews = pd.read_csv("my_ratings.csv")


# Convert rating values to a matrix (pivot_table) with userId as row and movieId as column.
user_intraction_matrix =user_reviews.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# We utilize this user-item matrix to calculate the similarity between users using cosine similarity.
user_similarity_scores = cosine_similarity(user_intraction_matrix)

# We print the shape of the resulting user interaction matrix and similarity matrix to observe how it appears.
print("User-Intraction Matrix Shape:", user_intraction_matrix.shape)
print("User Similarity scores Shape:", user_similarity_scores.shape)

"""section 9:
In this section, we construct a function that will assist us in determining the best appropriate movie for the user based on the user similarity score and ratings.

 First, we examine the similarity scores available for the target_user_id.  Then, we dot product that similarity with the interaction matrix to calculate the prediction score for each movie.

 We set the expected score to zero for movies that the user has already reviewed, allowing us to disregard them.

 Then, we arrange the top-n (top 10) projected scores in descending order and assign them movie IDs.

 These are the top movies that will be suggested to the user.
"""

# We import the numpy library, which assists with mathematical operations.
import numpy as np

# We develop a function that recommends movies to a specific user.
def generate_recommendations_for_user(target_user_id, user_similarity_scores, user_intraction_matrix,top_n=10):

  # We utilize the similarity score of that user.
  similarty_scores = user_similarity_scores[target_user_id - 1]

  # We calculate the user similarity as a dot product with the other user interaction matrix and get prediction scores.
  predicted_scores =np.dot(similarty_scores, user_intraction_matrix)

  # We discover the movies the user has already rated and set the prediction score for those movies to zero.
  alredy_rated_movies = user_intraction_matrix.loc[target_user_id]> 0
  predicted_scores[alredy_rated_movies] = 0

  # We rank the prediction score by the highest and select the top N movie index.
  top_movie_indices = np.argsort(predicted_scores)[::-1][:top_n]

  # We return the movie ID using the movie index.
  recommended_movie_ids = user_intraction_matrix.columns[top_movie_indices]


  return recommended_movie_ids

# Create recommendations for a specific user.
recommended_movie_ids = generate_recommendations_for_user(target_user_id=1,user_similarity_scores=user_similarity_scores,user_intraction_matrix=user_intraction_matrix)

# Show recommended movies with titles.
recommended_movies = movie_dataset[movie_dataset['movieId'].isin(recommended_movie_ids)]
print(recommended_movies[['movieId', 'title']])

"""### Section 10:
In this section, we'll create a hybrid recommendation system.  Here, we combine CBF (Content-Based Filtering) and CF (Collaborative Filtering) to provide movie choices that are tailored to the user's specific preferences.

 The fetch_genre_similarity() function finds genre-based similar movies to a movie and returns a top 10 list.  We utilized the Cosine similarity approach here.

 In the fetch_user_predictions() method, we investigate the user's previous rating history.  We generate the weighted average rating by comparing the similarities to check if the user's preferences match.

 The generate_hybrid_recommendations() function averages the two results and returns a final score.  Then we select the best ten movie IDs and offer the final product.
"""

# Function that examines genre similarities for each movie
# We compute similarities in content-based filtering using cosine similarity.
def fetch_genre_similarity(movie_id, cbf_score_map, movie_dataset):
    movie_index = movie_dataset[movie_dataset['movieId'] == movie_id].index[0]
    similarity_values = list(enumerate(cbf_score_map[movie_index]))
    similarity_values = sorted(similarity_values, key=lambda x: x[1], reverse=True)
    return {movie_dataset.iloc[i]['movieId']: similarity for i, similarity in similarity_values[1:21]}

# A method that detects the rating of a specific user from comparable users, gives it weight, and forecasts
#This is collaborative filtering logic.
def fetch_user_predictions(user_id, user_similarity_scores, user_intraction_matrix):
    similarity_row = user_similarity_scores[user_id]
    weighted_ratings = np.dot(similarity_row, user_intraction_matrix)
    already_rated = user_intraction_matrix.loc[user_id] > 0
    weighted_ratings[already_rated.values] = 0
    return dict(zip(user_intraction_matrix.columns, weighted_ratings))

# Creates a hybrid recommendation output by combining CBF and CF logics
#both logics average weight, giving the top 10 results.
def generate_hybrid_recommendations(user_id, movie_id):
    genre_scores = fetch_genre_similarity(movie_id, cbf_score_map, movie_dataset)
    user_scores = fetch_user_predictions(user_id, user_similarity_scores, user_intraction_matrix)

    combined_ids = set(genre_scores.keys()).union(user_scores.keys())
    hybrid_output = {}

    for m_id in combined_ids:
        genre_value = genre_scores.get(m_id, 0)
        user_value = user_scores.get(m_id, 0)
        hybrid_output[m_id] = (genre_value + user_value) / 2

    sorted_list = sorted(hybrid_output.items(), key=lambda x: x[1], reverse=True)[:10]
    top_movie_ids = [movie_id for movie_id, _ in sorted_list]

    return movie_dataset[movie_dataset['movieId'].isin(top_movie_ids)][['movieId', 'title']]

# Call the hybrid recommendation function and view the results for user_id = 552, movie_id = 592.
hybrid_output = generate_hybrid_recommendations(552,592)

# Print the final output result - id and title alone.
print(hybrid_output)

