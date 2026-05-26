import streamlit as st
import pickle
import pandas as pd
import requests
import urllib.parse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------- API KEYS ----------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

OMDB_API_KEY = st.secrets["OMDB_API_KEY"]


# ---------------- OPENROUTER CLIENT ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ---------------- LOAD DATA ----------------

movies_dict = pickle.load(
    open('movies_dict.pkl', 'rb')
)

movies = pd.DataFrame(movies_dict)


# ---------------- CREATE SIMILARITY MATRIX ----------------

cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['tags']
).toarray()

similarity = cosine_similarity(vectors)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0B0F19;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #0B0F19;
}

/* Hide Streamlit Branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* Main Layout */

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Main Title */

.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.main-title span {
    color: #E50914;
}

/* Hero Section */

.hero {
    background:
    linear-gradient(
        to right,
        rgba(0,0,0,0.92),
        rgba(0,0,0,0.55)
    ),
    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");

    background-size: cover;
    background-position: center;

    border-radius: 24px;

    padding: 80px;

    margin-top: 20px;
    margin-bottom: 40px;
}

.hero-title {
    font-size: 64px;
    font-weight: 800;
    color: white;
    line-height: 1.1;
    max-width: 750px;
}

.hero-subtitle {
    font-size: 20px;
    color: #d1d1d1;
    margin-top: 20px;
    line-height: 1.7;
    max-width: 700px;
}

/* Selectbox */

.stSelectbox > div > div {
    background-color: #151A24;
    border: 1px solid #2B3240;
    border-radius: 12px;
}

/* Input */

.stTextInput > div > div > input {
    background-color: #151A24 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #2B3240 !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #E50914,
        #ff3131
    );

    color: white;
    border: none;

    border-radius: 14px;

    height: 50px;
    width: 180px;

    font-size: 16px;
    font-weight: 600;

    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);

    box-shadow:
    0px 0px 20px rgba(229,9,20,0.4);
}

/* Movie Card */

.movie-card {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 18px;

    padding: 15px;

    margin-bottom: 20px;

    backdrop-filter: blur(12px);

    transition: 0.3s;
}

.movie-card:hover {
    transform: translateY(-5px);

    box-shadow:
    0px 0px 25px rgba(229,9,20,0.15);
}

/* Movie Title */

.movie-name {
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 8px;
}

/* Movie Info */

.movie-info {
    color: #d1d1d1;
    font-size: 15px;
    margin-bottom: 4px;
}

/* AI Box */

.ai-box {
    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 18px;

    padding: 25px;

    margin-top: 20px;

    font-size: 16px;

    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)


# ---------------- FETCH MOVIE DETAILS ----------------

def fetch_movie_details(movie_name):

    url = (
        f"http://www.omdbapi.com/?t={urllib.parse.quote(movie_name)}"
        f"&apikey={OMDB_API_KEY}"
    )

    data = requests.get(url).json()

    poster = data.get('Poster')

    if poster == "N/A" or poster is None:

        poster = (
            "https://via.placeholder.com/300x450?text=No+Image"
        )

    rating = data.get('imdbRating', 'N/A')

    genre = data.get('Genre', 'N/A')

    year = data.get('Year', 'N/A')

    return poster, rating, genre, year


# ---------------- RECOMMEND FUNCTION ----------------

def recommend(movie):

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []
    recommended_ratings = []
    recommended_genres = []
    recommended_years = []
    recommended_similarity = []

    for i in movies_list:

        movie_title = movies.iloc[i[0]].title

        poster, rating, genre, year = fetch_movie_details(
            movie_title
        )

        recommended_movies.append(movie_title)

        recommended_posters.append(poster)

        recommended_ratings.append(rating)

        recommended_genres.append(genre)

        recommended_years.append(year)

        similarity_score = round(i[1] * 100)

        recommended_similarity.append(similarity_score)

    return (
        recommended_movies,
        recommended_posters,
        recommended_ratings,
        recommended_genres,
        recommended_years,
        recommended_similarity
    )


# ---------------- AI FUNCTION ----------------

def ask_ai(prompt):

    completion = client.chat.completions.create(
        model="openai/gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an intelligent movie assistant."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content


# ---------------- MAIN TITLE ----------------

st.markdown("""
<div class="main-title">
AI Movie <span>Recommendation</span>
</div>
""", unsafe_allow_html=True)


# ---------------- HERO SECTION ----------------

st.markdown("""
<div class="hero">
    <h1 class="hero-title">
        Discover Movies with AI
    </h1>


</div>
""", unsafe_allow_html=True)

# ---------------- MOVIE SELECT ----------------

selected_movie_name = st.selectbox(
    "Search or Select a Movie",
    movies['title'].values
)


# ---------------- RECOMMEND BUTTON ----------------

if st.button("Recommend Movies"):

    (
        names,
        posters,
        ratings,
        genres,
        years,
        similarity_scores
    ) = recommend(selected_movie_name)

    cols = st.columns(5)

    for idx, col in enumerate(cols):

        with col:

            st.markdown(
                '<div class="movie-card">',
                unsafe_allow_html=True
            )

            st.image(posters[idx])

            st.markdown(
                f"""
                <div class="movie-name">
                    {names[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="movie-info">
                    Match Score: {similarity_scores[idx]}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="movie-info">
                    IMDb Rating: {ratings[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="movie-info">
                    Genre: {genres[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="movie-info">
                    Release Year: {years[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

            query = urllib.parse.quote(
                names[idx] + " official trailer"
            )

            youtube_search = (
                f"https://www.youtube.com/results?"
                f"search_query={query}"
            )

            st.link_button(
                "Watch Trailer",
                youtube_search
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ---------------- AI ASSISTANT ----------------

st.markdown("---")

st.header("AI Movie Assistant")

user_prompt = st.text_input(
    "Ask anything about movies"
)

if st.button("Ask AI"):

    answer = ask_ai(user_prompt)

    st.markdown(f"""
    <div class="ai-box">

    {answer}

    """, unsafe_allow_html=True)