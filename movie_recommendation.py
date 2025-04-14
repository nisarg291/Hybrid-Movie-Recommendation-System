import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Step 1: Load & Preprocess Data

# Load datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Remove years from movie titles
movies['title'] = movies['title'].apply(lambda x: re.sub(r'\(\d{4}\)', '', x).strip())

# ---- FIXED ID MAPPING ----
unique_movie_ids = movies['movieId'].unique()
movie_id_mapping = {id: idx for idx, id in enumerate(unique_movie_ids)}
movie_id_reverse_mapping = {idx: id for id, idx in movie_id_mapping.items()}

# Map movies in ratings dataset safely
ratings = ratings[ratings['movieId'].isin(movie_id_mapping)]  # Remove missing movies
ratings['mapped_movieId'] = ratings['movieId'].map(movie_id_mapping)

# Ensure mapped_movieId values are within range
assert ratings['mapped_movieId'].max() < len(unique_movie_ids), "Movie ID mapping error!"

# Get correct number of movies for embedding layer
num_movies = len(unique_movie_ids)

# Create user ID mappings
unique_user_ids = ratings['userId'].unique()
user_id_mapping = {id: idx for idx, id in enumerate(unique_user_ids)}

# Map user IDs
ratings['mapped_userId'] = ratings['userId'].map(user_id_mapping)

# Ensure no NaN values
ratings.dropna(inplace=True)

# Get unique counts
num_users = len(unique_user_ids)

# Split data
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)


# Step 2: Content-Based Filtering

# TF-IDF on movie genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))  # Handling missing genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(movie_title, top_n=5):
    """Recommend movies based on genres using TF-IDF cosine similarity."""
    if movie_title not in movies['title'].values:
        return ["Movie not found in dataset."]
    
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return movies.iloc[movie_indices]['title'].tolist()


# Step 3: Deep Learning Model

# User Input
user_input = Input(shape=(1,))
user_embedding = Embedding(input_dim=num_users, output_dim=10)(user_input)
user_flatten = Flatten()(user_embedding)

# Movie Input
movie_input = Input(shape=(1,))
movie_embedding = Embedding(input_dim=num_movies, output_dim=10)(movie_input)
movie_flatten = Flatten()(movie_embedding)

# Fully Connected Layers
concat_layer = Concatenate()([user_flatten, movie_flatten])
dense_1 = Dense(64, activation='relu')(concat_layer)
dense_2 = Dense(32, activation='relu')(dense_1)
output = Dense(1, activation='linear')(dense_2)

# Build & Compile Model
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train Model
model.fit(
    x=[train_data['mapped_userId'], train_data['mapped_movieId']],
    y=train_data['rating'],
    batch_size=64,
    epochs=20,
    validation_split=0.1
)

# Step 4: Model Evaluation

# Predict on test set
y_true = test_data['rating'].values
y_pred = model.predict([test_data['mapped_userId'], test_data['mapped_movieId']]).flatten()

# Convert predicted ratings into a classification problem (liked vs not liked)
threshold = 3  # Users "like" a movie if rating >= 3.5
y_true_class = (y_true >= threshold).astype(int)
y_pred_class = (y_pred >= threshold).astype(int)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
accuracy = accuracy_score(y_true_class, y_pred_class)
precision = precision_score(y_true_class, y_pred_class, zero_division=1)
recall = recall_score(y_true_class, y_pred_class, zero_division=1)

# Print evaluation results
print("\n Model Performance Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", classification_report(y_true_class, y_pred_class))


# Step 5: Hybrid Recommendations Combining collaborative and content-based filtering

def hybrid_recommendations(user_id, movie_title, model, top_n=5):
    """Hybrid approach combining collaborative and content-based filtering."""
    if movie_title not in movies['title'].values:
        return ["Movie not found in dataset."]
    
    # Ensure the user ID is valid
    mapped_user_id = user_id_mapping.get(user_id, None)
    if mapped_user_id is None:
        return ["User ID not found."]
    
    # Prepare the movie IDs (1D array)
    all_movie_ids = np.array(range(num_movies))
    
    # Prepare the user array (user_id repeated for all movie_ids)
    user_array = np.full_like(all_movie_ids, mapped_user_id)
    
    # Ensure inputs are reshaped correctly: (batch_size, 1) shape
    user_array = user_array.reshape(-1, 1)
    all_movie_ids = all_movie_ids.reshape(-1, 1)
    
    # Predict scores using the collaborative filtering model
    predicted_scores = model.predict([user_array, all_movie_ids])
    
    # Get the top N movies from collaborative filtering
    top_movie_ids_cf = predicted_scores.flatten().argsort()[-top_n:][::-1]

    # Get content-based recommendations
    content_based_movies = content_based_recommendations(movie_title, top_n)
    
    # Combine CF and content-based recommendations
    hybrid_movies = list(set(movies.iloc[top_movie_ids_cf]['title']).union(content_based_movies))
    
    return hybrid_movies[:top_n]

# Example Usage
print("\n Content-Based Recommendations for 'Toy Story':")
print(content_based_recommendations("Toy Story"))

print("\n Hybrid Recommendations for User 1, based on 'Toy Story':")
print(hybrid_recommendations(1, "Toy Story", model))
