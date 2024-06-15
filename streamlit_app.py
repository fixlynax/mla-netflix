import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Title and description
st.title("Netflix Movies and TV Shows Analysis")
st.write("This application analyzes Netflix content and predicts content type and duration.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Netflix Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Uploaded Successfully!")

    # Show dataset
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Data preprocessing
    df['country'] = df['country'].fillna(df['country'].mode()[0])
    df['cast'].fillna('No Data', inplace=True)
    df['director'].fillna('No Data', inplace=True)
    df.dropna(inplace=True)
    df['date_added'] = pd.to_datetime(df['date_added'])
    df['month_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year

    # Show processed data
    if st.checkbox("Show Processed Data"):
        st.write(df.head())

    # Prediction section
    st.header("Prediction")
    prediction_type = st.selectbox("Select Prediction Type", ("Content Type", "Duration"))

    if prediction_type == "Content Type":
        # Prepare data
        X = df[['country', 'rating', 'duration', 'release_year']]
        X = pd.get_dummies(X, drop_first=True)
        y = df['type'].map({'Movie': 0, 'TV Show': 1})

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # User input
        country = st.selectbox("Country", df['country'].unique())
        rating = st.selectbox("Rating", df['rating'].unique())
        duration = st.number_input("Duration (minutes)")
        release_year = st.number_input("Release Year", min_value=int(df['release_year'].min()), max_value=int(df['release_year'].max()), step=1)

        # Prediction
        if st.button("Predict Content Type"):
            user_data = pd.DataFrame({
                'country': [country],
                'rating': [rating],
                'duration': [duration],
                'release_year': [release_year]
            })
            user_data = pd.get_dummies(user_data, drop_first=True)
            prediction = clf.predict(user_data)[0]
            st.write(f"The predicted content type is: {'Movie' if prediction == 0 else 'TV Show'}")

    elif prediction_type == "Duration":
        # Filter data for movies
        movies = df[df['type'] == 'Movie']
        movies['duration'] = movies['duration'].str.replace(' min', '').astype(int)

        # Prepare data
        X = movies[['country', 'rating', 'release_year']]
        X = pd.get_dummies(X, drop_first=True)
        y = movies['duration']

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X_train, y_train)

        # User input
        country = st.selectbox("Country", df['country'].unique())
        rating = st.selectbox("Rating", df['rating'].unique())
        release_year = st.number_input("Release Year", min_value=int(df['release_year'].min()), max_value=int(df['release_year'].max()), step=1)

        # Prediction
        if st.button("Predict Duration"):
            user_data = pd.DataFrame({
                'country': [country],
                'rating': [rating],
                'release_year': [release_year]
            })
            user_data = pd.get_dummies(user_data, drop_first=True)
            prediction = reg.predict(user_data)[0]
            st.write(f"The predicted duration is: {int(prediction)} minutes")
