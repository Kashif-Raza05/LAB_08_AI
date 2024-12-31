import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Given dataset with movie titles, years, and average ratings
data = {
    'title': [
        'Stiff Upper Lips',
        'Dancer, Texas Pop. 81',
        'Me You and Five Bucks',
        'Sardaarji',
        'One Man\'s Hero',
        'The Shawshank Redemption',
        'There Goes My Baby',
        'The Prisoner of Zenda',
        'The Godfather',
        'The Godfather: Part II'
    ],
    'year': [
        1998.0,
        1998.0,
        2015.0,
        2015.0,
        1999.0,
        1994.0,
        1994.0,
        1937.0,
        1972.0,
        1974.0],
    'vote_average': [
        10.0,
        10.0,
        10.0,
        9.5,
        9.3,
        8.5,
        8.5,
        8.4,
        8.4,
        8.3
    ]
}

# Create DataFrame for the movie dataset
df = pd.DataFrame(data)

# Simulated user ratings for each movie (in a real-world scenario, this would come from user data)
ratings_data = {
    "User": ["Kashif", "ali", "kashif", "Raza", "Raza", "Raza", "asghar", "asghar", "asghar", "hussain", "hussain", "hussain"],
    "Movie": ["Stiff Upper Lips", "Dancer, Texas Pop. 81", "Me You and Five Bucks", "Stiff Upper Lips", 
              "Dancer, Texas Pop. 81", "Me You and Five Bucks", "Sardaarji", "One Man's Hero", 
              "The Shawshank Redemption", "There Goes My Baby", "The Prisoner of Zenda", "The Godfather"],
    "Rating": [4, 3, 5, 2, 4, 4, 3, 5, 5, 4, 4, 5]
}

# Create DataFrame for user ratings
ratings_df = pd.DataFrame(ratings_data)

# Create user-item matrix (rows = users, columns = movies)
user_item_matrix = ratings_df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Compute cosine similarity between movies (item-based collaborative filtering)
cosine_sim = cosine_similarity(user_item_matrix.T)  # Transpose to compare movies
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Function to recommend movies based on user input
def recommend_movies_for_user(user_name, num_recommendations=5):
    if user_name not in ratings_df['User'].values:
        return f"User '{user_name}' not found in the dataset. Please check your input."
    
    # Get the movies rated by the selected user
    user_ratings = user_item_matrix.loc[user_name]
    
    # For movies rated by the user, calculate the predicted ratings for other movies
    predicted_ratings = {}
    
    for movie in user_item_matrix.columns:
        if user_ratings[movie] == 0:  # Movie not rated by the user
            # Get the similarity scores for this movie with all the other movies rated by the user
            similar_movies = cosine_sim_df[movie]
            
            # Calculate predicted rating as the weighted sum of rated movies' ratings
            weighted_ratings = sum(user_ratings * similar_movies) / sum(similar_movies) if sum(similar_movies) != 0 else 0
            predicted_ratings[movie] = weighted_ratings

    # Sort the predicted ratings in descending order and return top N recommendations
    recommended_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
    
    return recommended_movies

# Display available users and movies
print("Available users in the dataset:")
for user in ratings_df["User"].unique():
    print(f"- {user}")

# Assign user name directly instead of using input()
selected_user = "kashif"  # Change this value for different inputs

# Generate recommendations
recommendations = recommend_movies_for_user(selected_user)

# Display recommendations
if isinstance(recommendations, str):  # If an error occurred
    print(recommendations)
else:
    print(f"\nTop movie recommendations for '{selected_user}':")
    for movie, score in recommendations:
        print(f"- {movie} (Predicted Rating: {score:.2f})")