# 🎬 Hybrid Movie Recommendation System

This project implements a **Hybrid Movie Recommendation System** that combines both **Content-Based Filtering** and **Collaborative Filtering (Deep Learning)** to recommend movies to users based on their previous ratings and movie metadata (genres). It leverages TensorFlow/Keras for deep learning and uses TF-IDF vectorization for genre-based similarity.

## 🚀 Features

- **Content-Based Filtering** using genres and TF-IDF + cosine similarity
- **Collaborative Filtering** using neural embeddings for users and movies
- **Hybrid Recommendations** that combine both filtering techniques
- Clean preprocessing and safe mapping of user and movie IDs
- Model evaluation with RMSE, MAE, Accuracy, Precision, Recall

## 📁 Dataset

The model uses the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/) which includes:

- `movies.csv` (movie titles and genres)
- `ratings.csv` (user ratings)

> Make sure to download and place the dataset in a folder named `ml-latest-small/` in your project root.

## 🧠 Model Architecture

- **User & Movie Embedding Layers**
- Fully connected Dense Layers
- Regression output predicting movie rating (1–5 scale)

## 📊 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Accuracy, Precision, Recall (using thresholded ratings)
- Classification Report

## 📦 Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/hybrid-movie-recommender.git
    cd hybrid-movie-recommender
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MovieLens dataset:
    - Visit [MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/)
    - Download `ml-latest-small.zip` and extract into the project directory

## ▶️ How to Run

```bash
python movie_recommendation.py
