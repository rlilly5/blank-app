import streamlit as st
import pandas as pd
import datetime
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = data = pd.read_csv('ProcessedTicketData.csv')

### Load your pre-trained model and encoders (assuming they are saved)
#model = model
#encoders = data['venue', 'artist']

#########################
####FROM MODEL CODE######
#########################

# Convert 'date' column to string, take first 10 characters, and convert to datetime
data['date'] = data['date'].astype(str).str[:10]
data['date'] = pd.to_datetime(data['date'])

print(data['date'].head(10))  # first 10 rows of the date column

# Target variable
target = 'max_price'

# Drop the target column to get only the features
features = data.drop(columns=['event_id', 'max_price'])

# Dictionary to store encoders for each column
encoders = {}

# Encode categorical columns
for col in ['artist', 'venue', 'city', 'state', 'ticket_vendor']:
    if col in features:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(data[col])
        encoders[col] = encoder

print(f"Features shape: {features.shape}")

print(features.head())

# Splitting the data in test and train datasets

X = features
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% train data, 20% test data

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Splitting the data in test and train datasets

X = features
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% train data, 20% test data

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Splitting the data in test and train datasets

X = features
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% train data, 20% test data

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test)

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
