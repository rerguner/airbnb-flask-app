from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# Load model and feature list
model = joblib.load("xgb_airbnb_model_final.pkl")
features = joblib.load("xgb_airbnb_model_features.pkl")

# Flask app
app = Flask(__name__)

# Prepare user input for prediction
def prepare_input(neighbourhood, amenities_bin, price_tier, beds):
    input_data = {col: 0 for col in features}
    input_data['beds'] = float(beds)

    # One-hot encode neighbourhood
    neighbourhood_col = f"{neighbourhood.lower()}"
    if neighbourhood_col in input_data:
        input_data[neighbourhood_col] = 1

    # One-hot encode amenities bin
    amenities_col = f"weighted_amenities_bin_{amenities_bin}"
    if amenities_col in input_data:
        input_data[amenities_col] = 1

    # One-hot encode price tier
    price_tier_col = f"price_tier_{price_tier}"
    if price_tier_col in input_data:
        input_data[price_tier_col] = 1

    return pd.DataFrame([input_data])

# ROI calculation
def calculate_roi(nightly_price, occupancy_rate, property_price):
    if property_price == 0 or occupancy_rate == 0:
        return float("inf"), 0.0

    annual_income = nightly_price * occupancy_rate * 365
    if annual_income == 0:
        return float("inf"), 0.0

    roi_months = (property_price / annual_income) * 12
    return round(roi_months, 1), round(annual_income, 2)

# Home route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    neighbourhood = data.get("neighbourhood")
    amenities_bin = data.get("weighted_amenities_bin")
    price_tier = data.get("price_tier")
    beds = data.get("beds")
    property_price = float(data.get("property_price"))
    occupancy_rate = float(data.get("occupancy_rate", 0.7))

    # Prepare input
    X_input = prepare_input(neighbourhood, amenities_bin, price_tier, beds)

    # Predict nightly price directly (already in price scale)
    nightly_price = model.predict(X_input)[0]

    # Estimate a ±10% price range
    price_range = (round(nightly_price * 0.9), round(nightly_price * 1.1))

    # Calculate ROI
    roi_months, annual_income = calculate_roi(nightly_price, occupancy_rate, property_price)

    return jsonify({
    "predicted_price": float(round(nightly_price, 2)),
    "price_range": [float(round(nightly_price * 0.9)), float(round(nightly_price * 1.1))],
    "roi_months": float(roi_months),
    "annual_income": float(annual_income)
})


# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False)
