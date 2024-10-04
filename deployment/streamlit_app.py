import streamlit as st
import requests
from pydantic import BaseModel

# Define the URL for the FastAPI backend
FASTAPI_URL = "http://fastapi:80/predict"

# Set the title of the Streamlit app
st.title("Trades Classification")

class InputData(BaseModel):
    price: float
    qty: float
    isBuyerMaker: bool
    isBestMatch: bool
    percent_to_1000: float
    aggregated_trades: int
    price_seen_before: bool

# Input fields for user to enter data
price = st.number_input("Price", min_value=0.0)
qty = st.number_input("Quantity in coins", min_value=0.0)
isBuyerMaker = st.number_input("Is buyer maker?", min_value=0.0)
isBestMatch = st.number_input("Is best match?", min_value=0.0)
percent_to_1000 = st.number_input("Percent till 1000", min_value=0.0)
aggregated_trades = st.number_input("Amount of trades", min_value=0.0)
price_seen_before = st.number_input("Price seen before?", min_value=0.0)
model_name = st.text_input("Model Name")

if st.button("Predict"):
    trade_data = InputData(
        price=price,
        qty=qty,
        isBuyerMaker=int(isBuyerMaker),
        isBestMatch=int(isBestMatch),
        percent_to_1000=percent_to_1000,
        aggregated_trades=int(aggregated_trades),
        price_seen_before=int(price_seen_before)
    )

    input_data = trade_data.dict()
    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(f"{FASTAPI_URL}?model_name={model_name}", json=input_data)
    
    # Debugging: Print the response status code and content
    st.write(f"Response status code: {response.status_code}")
    st.write(f"Response content: {response.content}")

    # Check if the request was successful
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        # Display the result
        st.success(f"The model predicts class: {prediction}")
    else:
        st.error("Failed to get prediction from the server.")