from flask import Flask, render_template, request, jsonify
from model.model_loader import load_artifacts
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

model, le, preprocessor, car_models = load_artifacts()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/car_models')
def get_car_models():
    return jsonify(car_models)

@app.route('/predictprice', methods=['POST'])
def predict_price():
    car_model = request.form.get('car_model')
    vehicle_age = float(request.form.get('vehicle_age'))
    km_driven = float(request.form.get('km_driven'))
    seller_type = request.form.get('seller_type')
    fuel_type = request.form.get('fuel_type')
    transmission_type = request.form.get('transmission_type')
    mileage = float(request.form.get('mileage'))
    engine = float(request.form.get('engine'))
    max_power = float(request.form.get('max_power'))
    seats = float(request.form.get('seats'))

    car_model = le.transform([car_model])[0]

    input_data = {
        'model': [car_model],
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'seller_type': [seller_type],
        'fuel_type': [fuel_type],
        'transmission_type': [transmission_type],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats]
    }

    input_df = pd.DataFrame(input_data)
    transformed_df = preprocessor.transform(input_df)
    predicted_price = model.predict(transformed_df)
    final_price = round(predicted_price[0], 2)

    return jsonify({'predicted_price': final_price})

if __name__ == "__main__":
    app.run(debug=True)
    