from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
from openai import OpenAI

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
def convert_to_string(age, sex, cp, trestbps, fbs, thalach, exang):
    sex_str = "Male" if sex == 0 else "Female"
    cp_str = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][int(cp)]
    fbs_str = "Less Than 120 mg/dl" if fbs == 0 else "Greater Than or Equal to 120 mg/dl"
    exang_str = "No" if exang == 0 else "Yes"
    return {
        "age": age,
        "sex": sex_str,
        "cp": cp_str,
        "trestbps": trestbps,
        "fbs": fbs_str,
        "thalach": thalach,
        "exang": exang_str
    }

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
    return render_template('result.html', prediction=prediction_text,
                           age=age, sex=sex, cp=cp, 
                           trestbps=trestbps, fbs=fbs, 
                           thalach=thalach, exang=exang)


@app.route('/recommend', methods=['GET'])
def recommend():
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    fbs = request.args.get('fbs')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    prediction = request.args.get('prediction')
    patient_data = convert_to_string(age, sex, cp, trestbps, fbs, thalach, exang)
    client = OpenAI()
    prompt_text = f"""
    You are an expert in giving people recommendations on how to improve their heart health , here are the details of a person and the chances of them having a heart disease predicted through a ML model:
    Age: {age}
    Sex: {patient_data['sex']}
    Chest Pain Type (cp): {patient_data['cp']}
    Resting Blood Pressure (trestbps): {trestbps}
    Fasting Blood Sugar (fbs): {patient_data['fbs']}
    Maximum Heart Rate Achieved (thalach): {thalach}
    Exercise Induced Angina (exang): {patient_data['exang']}
    Prediction: {prediction}
    Based on the above information give tips to the person in order to maintain or improve their heart health.Keep in mind the persons age in your recommendations, Start directly from the recommendations  without text like "here are the recommendations etc" dont apply any formatting like bold i'll do myself 
    """

    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_text}

        ]
        )

        recommendations_text = response.choices[0].message.content.strip()
        recommendations_list = recommendations_text.split('\n')

        recommendations = [rec.strip() for rec in recommendations_list if rec.strip()]
        print(recommendations)
    except Exception as e:
        print(f"OpenAI API call failed: {e}")
        recommendations = [
            'Start by consulting with your healthcare provider to create a tailored plan for your heart health. Make sure to focus on the following recommendations:',
            '1. Monitor and manage your blood pressure regularly to keep it within a healthy range.',
            '2. Aim to maintain a balanced diet that is low in saturated fats, cholesterol, and sodium.',
            '3. Focus on incorporating more fruits, vegetables, whole grains, and lean proteins into your meals.',
            '4. Stay physically active with exercises that are safe and suitable for you.',
            '5. If you smoke, consider quitting and seek support to help you with the process.'
        ]
    return render_template('recommendations.html', recommendations=recommendations)




if __name__ == '__main__':
    app.run(debug=True)
    # print(predict(63, 1, 3, 145, 0, 150, 0))
