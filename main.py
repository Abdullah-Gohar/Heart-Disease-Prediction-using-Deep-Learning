from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the saved deep learning model
model = load_model('heart_disease_model.keras')

# Define the preprocessing function


def preprocess_input(age, sex, cp, trestbps, fbs, thalach, exang):
    # Convert categorical variable (cp) to binary indicators
    cp_values = [0, 1, 2, 3]
    cp_indicators = [1 if cp == cp_val else 0 for cp_val in cp_values]
    
    input_data = [[age,sex,trestbps,fbs,thalach,exang]+cp_indicators]
    
    scaler = joblib.load('scaler.pkl')
    input_data = scaler.transform(input_data)
    print(input_data)
    
    return input_data
    # return np.array(input_data)

    # # Combine numeric features into input data array
    # input_data = np.array([age, trestbps, thalach])
    # input_data = input_data.reshape(1, -1)  # Convert to 2D array

    # # scaler = StandardScaler()
    # scaler = joblib.load('CEP/scaler.pkl')
    # input_data_scaled = scaler.fit_transform(input_data)
    # print(input_data_scaled)
    # # Add binary indicators to the scaled numeric features
    # input_data_scaled_with_cp = np.concatenate((input_data_scaled, cp_indicators), axis=1)

    # return input_data_scaled_with_cp
    return input_data
    input_data_scaled = (input_data - input_data.mean()) / input_data.std()  # Normalize the input data
    return input_data_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    fbs = int(request.form['fbs'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])

    # Preprocess the input data
    input_data = preprocess_input(age, sex, cp, trestbps, fbs, thalach, exang)

    # Make predictions using the loaded model
    
    print(input_data)
    predictions = model.predict(input_data)

    print(predictions)
    # Convert the prediction to a human-readable format
    prediction_text = "Heart Disease" if predictions[0][0] > 0.5 else "No Heart Disease"
    
    prediction_text = f"{predictions[0][0]*100:.2f}% chance of Heart Disease"

    # Return the prediction result to the frontend
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
    # print(predict(63, 1, 3, 145, 0, 150, 0))
