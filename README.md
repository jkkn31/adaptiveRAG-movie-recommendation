# Personalized Movie Recommendation using Advanced RAG Techniques

## Overview

The "Movie Recommender" Streamlit app represents an innovative approach to movie recommendations, leveraging advanced language models and vector databases to deliver personalized film suggestions. This application aims to simplify the movie selection process by allowing users to input their preferences and receive tailored recommendations for films, genres, and actors.


Built upon state-of-the-art language models and embedding techniques, "Movie Recommender" ensures accurate and insightful responses, enhancing user engagement through a streamlined user interface provided by Streamlit. In this report, we explore the development journey of "Movie Recommender," detailing the methodologies employed, the challenges encountered, and the innovative solutions implemented to create a versatile and user-centric movie recommendation tool.


## Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-adaptive-rag.streamlit.app/) 

* Youtube Video Link: https://youtu.be/6ZUpBJV6Vgg
* Streamlit App Link: https://movie-recommender-adaptive-rag.streamlit.app/ 
* Document Link: https://docs.google.com/document/d/1elgn5jlm53AonrWgatS69GbnjeIw5H-rV4-MsR7A70A/edit?usp=sharing


## Features

- Personalized movie recommendations based on user input
- Advanced natural language processing for query understanding
- Efficient retrieval of relevant movie information using FAISS vector store
- Handling of diverse user queries and edge cases
- Balance between popular and niche movie recommendations

## Technology Stack

- Streamlit for the user interface
- Advanced language models for natural language processing
- FAISS for efficient vector similarity search
- JSON for data storage and normalization
- Python for backend logic

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the Repository
```
git clone https://github.com/jkkn31/adaptiveRAG-movie-recommendation.git
cd adaptiveRAG-movie-recommendation
```

2. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install required packages
```
pip install -r requirements.txt
```

## Setup Guide

1. Set Up Environment Variables
   Create a `.env` file in the root directory of the project and add your API keys:
   ```
   LANGUAGE_MODEL_API_KEY=your_language_model_api_key
   EMBEDDING_API_KEY=your_embedding_api_key
   ```

2. Running the App
   Start the Streamlit App by running the following command:
   ```
   streamlit run streamlit_app.py
   ```

3. Interact with the App:
   Open your web browser and go to `http://localhost:8501` to interact with the Movie Recommender app. Enter your movie preferences or questions to receive personalized recommendations.


## Challenges Overcome

- Limited context in the movie dataset, Implemented Data Normalization to improve the retrieved data quality.
- Efficient processing of large movie databases
- Secure API key management
- Accurate response generation and formatting
- Handling ambiguous and non-relevant queries


## Future Improvements

- Expand the movie dataset
- Refine the recommendation algorithm
- Enhance the user interface
- Implement user feedback and learning mechanisms
