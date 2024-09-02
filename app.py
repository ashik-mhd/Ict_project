from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('model (1).pkl', 'rb') as model_file:
    rf = pickle.load(model_file)

with open('scaler (4).pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [																	
        float(request.form['satisfaction_level']),
        float(request.form['last_evaluation']),
        float(request.form['number_project']),
        float(request.form['average_monthly_hours']),
        float(request.form['time_spend_company']),
        float(request.form['work_accident']),
        float(request.form['promotion_last_5years']),
        float(request.form['Department']),
        float(request.form['salary'])
    ]

    # Get input data from the request
    new_data = pd.DataFrame([features])

    # Scale the new data
    new_data_scaled = scaler.transform(new_data)

    # Predict using the loaded model
    prediction = rf.predict(new_data_scaled)

    # Extract the prediction from the array
    prediction = prediction[0]  # Get the single value (0 or 1)

    # Return the prediction and probability
    if prediction == 1:
        result = "Will Leave"
    else:
        result = "Will Stay"

    return render_template("result.html", prediction=result)


if __name__ == '__main__':
    app.run(debug=True)