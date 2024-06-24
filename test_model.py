from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib


scaler = joblib.load('CEP/scaler.pkl')
data = [63, 1, 145, 0, 150, 0,0,0,0,1]

scaled = scaler.transform([data])
print(scaled)
# def preprocess_input(age, sex, cp, trestbps, fbs, thalach, exang):
#     # Convert categorical variable (cp) to binary indicators
#     cp_values = [0, 1, 2, 3]
#     cp_indicators = [1 if cp == cp_val else 0 for cp_val in cp_values]

#     # Scale numeric features
#     scaler = MinMaxScaler()
#     numeric_features = np.array([[age, trestbps, thalach]])
#     print("Numeric Features Before Scaling:")
#     print(numeric_features)
#     numeric_features_scaled = scaler.fit_transform(numeric_features)
#     print("Numeric Features After Scaling:")
#     print(numeric_features_scaled)

#     # Combine scaled numeric features and binary indicators
#     input_data = np.concatenate((numeric_features_scaled[0], [sex, fbs, exang], cp_indicators))

#     return input_data.reshape(1, -1)

# print(preprocess_input(63, 1, 3, 145, 0, 150, 0))
