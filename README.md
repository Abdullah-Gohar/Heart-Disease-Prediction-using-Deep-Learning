# Heart Disease Prediction App

## Overview
This project is a Flask web application that predicts the likelihood of heart disease based on user input. It utilizes a deep learning model built with TensorFlow and Keras.

## Features
- **Predictive Model**: Uses medical and demographic data to predict heart disease.
- **User Interface**: Simple form for user input and display of prediction results.
- **Recommendations**: Provides personalized health tips based on prediction results.

## Requirements
- Python 3.x
- Flask
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- OpenAI

## Setup and Installation
1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/heart-disease-prediction-app.git
    cd heart-disease-prediction-app
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Model and Data**
    - Place the `heart_disease_model.keras` in the project directory.
    - Ensure `processed_cleveland.csv` is in the project directory.
  
4. **OpenAI API Key**
    - Place your own Open AI Api key in the API_KEY variable in main.py for personalized recommendations

## Running the Application
1. **Start the Flask App**
    ```bash
    python main.py
    ```

2. **Access the App**
    - Open a web browser and go to `http://127.0.0.1:5000/`.

## Usage
1. **Home Page**
    - Fill out the form with your medical and demographic information.
    - Click the submit button to get the prediction.

2. **Prediction Result Page**
    - View the likelihood of having heart disease.
    - Click on the link to get personalized health recommendations.

## Project Structure
- `main.py`: Main Flask application.
- `dl.py`: Deep learning model code.
- `corr.py`: Data preprocessing and correlation analysis.
- `templates/`: HTML templates.
  - `index.html`: Home page form.
  - `result.html`: Prediction result page.
  - `recommendations.html`: Recommendations page.
- `static/`: Static files like CSS and images.
  - `style.css`: Stylesheet.
  - `bg-img.png`: Background image.

