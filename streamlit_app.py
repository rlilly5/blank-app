import streamlit as st
import pandas as pd
import datetime
import math

data = data = pd.read_csv('ProcessedTicketData.csv')

# Load your pre-trained model and encoders (assuming they are saved)
model = model
encoders = data['venue', 'artist']

# Function to predict ticket price with input validation
def predict_ticket_price(artist_name, venue_name):
    if artist_name not in encoders['artist'].classes_:
        st.error(f"Artist '{artist_name}' not found in training data.")
        return

    if venue_name not in encoders['venue'].classes_:
        st.error(f"Venue '{venue_name}' not found in training data.")
        return

    artist_encoded = encoders['artist'].transform([artist_name])[0]
    venue_encoded = encoders['venue'].transform([venue_name])[0]

    input_sample = pd.DataFrame(columns=X_train.columns)  # Create a DataFrame for input
    input_sample.loc[0, 'artist'] = artist_encoded
    input_sample.loc[0, 'venue'] = venue_encoded

    # Assuming you don't have historical data for specific dates, fill other features with mean values
    for col in ['year', 'month', 'day', 'day_of_week', 'days_since_epoch']:
        if col in input_sample:
            input_sample.loc[0, col] = X_train[col].mean()  # Use mean of training data

    predicted_price = model.predict(input_sample)[0]
    st.write(f"Predicted ticket price: ${predicted_price:.2f}")

# Streamlit App
def main():
    st.title("Concert Ticket Price Predictor")

    # User input for Artist and Venue
    artist_name = st.text_input("Enter Artist Name")
    venue_name = st.text_input("Enter Venue Name")

    # Predict button and price display
    if st.button("Predict Price"):
        predict_ticket_price(artist_name, venue_name)

if __name__ == "__main__":
    main()
