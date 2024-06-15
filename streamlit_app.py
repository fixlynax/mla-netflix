import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("netflix_titles.csv")

df = load_data()

# Data preprocessing
# Handling missing values
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['rating'].fillna('No Rating', inplace=True)

# Label encoding categorical features
label_encoders = {}
for feature in ['rating', 'country']:
    label_encoders[feature] = LabelEncoder()
    df[feature] = label_encoders[feature].fit_transform(df[feature])

# Train Random Forest Classifier
X = df[['duration', 'release_year', 'country', 'rating']]
y = df['type'].map({'Movie': 0, 'TV Show': 1})

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Streamlit UI
st.title('Netflix Movie/TV Show Prediction')

# Input form
st.sidebar.header('User Input')
duration = st.sidebar.number_input('Duration (in minutes for movies, seasons for TV shows)', min_value=1)
release_year = st.sidebar.number_input('Release Year', min_value=1800, max_value=2023)
country = st.sidebar.text_input('Country')
rating = st.sidebar.selectbox('Rating', ['G', 'PG', 'PG-13', 'R', 'TV-Y', 'TV-Y7', 'TV-G', 'TV-PG', 'TV-14', 'TV-MA', 'NC-17', 'UR', 'NR'])

# Make prediction
def predict_movie_or_tv_show(duration, release_year, country, rating):
    rating_encoded = label_encoders['rating'].transform([rating])[0] if rating in label_encoders['rating'].classes_ else 0
    country_encoded = label_encoders['country'].transform([country])[0] if country in label_encoders['country'].classes_ else 0
    
    input_data = pd.DataFrame({'duration': [duration], 'release_year': [release_year], 'country': [country_encoded], 'rating': [rating_encoded]})
    prediction = clf.predict(input_data)
    return 'Movie' if prediction[0] == 0 else 'TV Show'

if st.sidebar.button('Predict'):
    prediction = predict_movie_or_tv_show(duration, release_year, country, rating)
    st.write(f"Prediction: {prediction}")
