import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

hd = pd.read_csv('CEP/processed_cleveland.csv', na_values = '?')


# hd['cp'].replace({1:'typical_angina', 2:'atypical_angina', 3: 'non-anginal_pain', 4: 'asymptomatic'}, inplace = True)
# hd['restecg'].replace({0:'normal', 1:' ST-T_wave_abnormality', 2:'left_ventricular_hypertrophy'}, inplace = True)
# hd['slope'].replace({1:'upsloping', 2:'flat', 3:'downsloping'}, inplace = True)
# hd['thal'].replace({3:'normal', 6:'fixed_defect', 7:'reversible_defect'}, inplace = True)
hd['num'].replace({2:1, 3:1, 4:1}, inplace = True)

hd.dropna(how = 'any', inplace = True)
# categorical_features = ['cp', 'thal', 'restecg', 'slope']
# List of categorical features to be label encoded
categorical_features = ['cp', 'thal', 'restecg', 'slope']

# Initialize the label encoder
label_encoders = {feature: LabelEncoder() for feature in categorical_features}

# Apply label encoding to each categorical feature
for feature in categorical_features:
    hd[feature] = label_encoders[feature].fit_transform(hd[feature])
# Compute the correlation matrix
correlation_matrix = hd.corr()

# Get correlation values with the target variable
target_correlation = correlation_matrix['num'].abs().sort_values(ascending=False)

print("Columns in order of descending absolute correlation with the target:")
print(target_correlation)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()