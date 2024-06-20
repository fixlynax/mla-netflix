import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
@st.cache_resource
def load_data():
    df = pd.read_csv("netflix_titles.csv", encoding='latin1')
    return df

# Preprocess the data
@st.cache_data
def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['rating'] = df['rating'].fillna(df['rating'].mode()[0])
    df['duration'] = df['duration'].fillna('0 min')

    # Convert duration to numeric (minutes for movies, seasons for TV shows)
    df['duration'] = df['duration'].apply(lambda x: int(x.split()[0]) * 60 if 'Season' not in x else int(x.split()[0]))

    # Encode categorical variables
    label_encoders = {}
    for feature in ['type', 'country', 'rating']:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])

    return df, label_encoders

# Load and preprocess data
df = load_data()
df, label_encoders = preprocess_data(df)

# Separate features and target variable
X = df[['duration', 'release_year', 'country', 'rating']]
y = df['type']

# Train the classifier
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X, y)

# Title and description
st.title("Netflix Movie or TV Show Classifier")
st.write("This app predicts whether a given input is a movie or a TV show based on its features.")

# Input form
st.sidebar.header('User Input')
duration = st.sidebar.number_input('Duration (in minutes for movies, seasons for TV shows)', min_value=1)
release_year = st.sidebar.number_input('Release Year', min_value=int(X['release_year'].min()), max_value=int(X['release_year'].max()))
country = st.sidebar.selectbox('Country', label_encoders['country'].classes_)
rating = st.sidebar.selectbox('Rating', label_encoders['rating'].classes_)

# Make prediction
def predict_movie_or_tv_show(duration, release_year, country, rating):
    country_encoded = label_encoders['country'].transform([country])[0]
    rating_encoded = label_encoders['rating'].transform([rating])[0]
    input_data = pd.DataFrame({'duration': [duration], 'release_year': [release_year], 'country': [country_encoded], 'rating': [rating_encoded]})
    
    # Debugging output: print input_data and prediction probabilities
    st.write("Input Data:")
    st.write(input_data)
    
    prediction = clf.predict(input_data)
    probabilities = clf.predict_proba(input_data)
    st.write("Prediction Probabilities:")
    st.write(probabilities)

    return 'Movie' if prediction[0] == 0 else 'TV Show'

if st.sidebar.button('Predict'):
    prediction = predict_movie_or_tv_show(duration, release_year, country, rating)
    if prediction:
        st.write(f"Prediction: {prediction}")
