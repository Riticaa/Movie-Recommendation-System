import streamlit as st
import pickle
import pandas as pd
import requests
import urllib.parse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI


# ---------------- API KEYS ----------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# ---------------- OPENROUTER CLIENT ----------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ---------------- LOAD DATA ----------------

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))

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


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stApp {
        background-color: #0E1117;
    }

    h1, h2, h3, h4 {
        color: white;
    }

    .hero {
        background: linear-gradient(
            to right,
            rgba(0,0,0,0.9),
            rgba(0,0,0,0.4)
        ),
        url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");

        padding: 80px;
        border-radius: 20px;
        margin-bottom: 30px;
        background-size: cover;
        background-position: center;
    }

    .hero-title {
        font-size: 50px;
        font-weight: bold;
        color: white;
    }

    .hero-subtitle {
    font-size: 20px !important;
    color: white !important;
    margin-top: 10px !important;
    font-weight: 400;
}

    .movie-card {
        background-color: #1c1c1c;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        transition: 0.3s;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    .movie-card:hover {
        transform: scale(1.03);
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 180px;
        font-size: 16px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #ff1f1f;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- FETCH MOVIE DETAILS ----------------

def fetch_movie_details(movie_name):

    url = (
        f"http://www.omdbapi.com/?t={movie_name}"
        f"&apikey={OMDB_API_KEY}"
    )

    data = requests.get(url).json()

    poster = data.get('Poster')

    rating = data.get('imdbRating')

    genre = data.get('Genre')

    year = data.get('Year')

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


# ---------------- TITLE ----------------

st.title("🎬 AI Movie Recommendation System")


# ---------------- HERO BANNER ----------------

st.markdown("""
<div class="hero">
    <h1 class="hero-title">
        Unlimited Movies, AI Recommendations & Trailers
    </h1>


</div>
""", unsafe_allow_html=True)
# ---------------- SELECT MOVIE ----------------

selected_movie_name = st.selectbox(
    "Search or Select a Movie",
    movies['title'].values
)


# ---------------- RECOMMEND BUTTON ----------------

if st.button('Recommend'):

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

            st.subheader(names[idx])

            st.write(
                f"🔥 {similarity_scores[idx]}% Match"
            )

            st.write(
                f"⭐ IMDb Rating: {ratings[idx]}"
            )

            st.write(
                f"🎭 Genre: {genres[idx]}"
            )

            st.write(
                f"📅 Year: {years[idx]}"
            )

            query = urllib.parse.quote(
                names[idx] + " trailer"
            )

            youtube_search = (
                f"https://www.youtube.com/results?search_query={query}"
            )

            st.markdown(
                f"[▶️ Watch Trailer]({youtube_search})"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ---------------- AI CHATBOT ----------------

st.markdown("---")

st.header("🤖 AI Movie Assistant")

user_prompt = st.text_input(
    "Ask AI for movie suggestions"
)

if st.button("Ask AI"):

    answer = ask_ai(user_prompt)

    st.write(answer)