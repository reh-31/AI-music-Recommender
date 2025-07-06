# AI Music Recommendation System

This repository implements a content-based music recommendation system that suggests songs based on their lyrical content, using modern machine learning and natural language processing techniques. Unlike traditional collaborative systems that depend on user interaction data, this project relies on analyzing song features directly, ensuring recommendations work even for new or niche songs.

## Overview
The system uses TF-IDF vectorization to transform lyrics or artist-title metadata into numerical features that capture each song’s unique characteristics. A Nearest Neighbors algorithm, with cosine similarity, then searches for the songs that are closest in meaning and style.

This project demonstrates how to build a user-friendly and scalable recommendation engine with a clean Streamlit web interface. It is designed for quick, real-time recommendations, supporting music exploration and playlist building based on lyrical similarity.

## Dataset
The system uses the Spotify Million Song Dataset (available on Kaggle), a rich dataset containing:

Song titles

Artist names

Lyrics for a large number of songs across genres

For songs with missing lyrics, the system uses artist + song name as a fallback to maintain recommendation quality.

You can adapt or expand this dataset to include even more metadata such as genre, mood, or tempo for hybrid recommendations in the future.

## Features
TF-IDF feature extraction to highlight important, unique words in each song
Nearest Neighbors search to identify and rank similar songs
Streamlit web app for an interactive and easy-to-use interface
SVG-based dynamic album covers to visually enrich the recommendations
Support for missing data by falling back to artist + song name metadata
Saved models for fast reuse without retraining
Modular code structure, easy to extend with audio or mood features in the future

## Technical Highlights
Preprocessing: Cleans and normalizes text data for consistency

TF-IDF Vectorization: Converts lyrics to sparse numerical vectors with a focus on unique language

Cosine Similarity with Nearest Neighbors: Measures closeness between songs in vector space

Streamlit: Provides an intuitive dropdown-driven UI for users to select a song and get recommendations instantly

Model Persistence: Uses Pickle to store the trained model and TF-IDF matrix for fast loading

## Example Usage
After launching the Streamlit app, a user can:

Select any song from the dropdown

Click “Show Recommendations”

See the top 5 most similar songs displayed with:

Song title

Artist

A unique album cover generated dynamically

A similarity match percentage

## Acknowledgments
Open data contributors on Kaggle

Inspiration from modern systems like Spotify’s and Apple Music’s recommendation pipelines

Streamlit, for making interactive ML prototypes fast and easy



