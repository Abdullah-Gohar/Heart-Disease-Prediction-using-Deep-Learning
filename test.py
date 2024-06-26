import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# # Load the dataset
# url = "cEP/processed_cleveland.csv"
# columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
# data = pd.read_csv(url, header=None, names=columns)

# # Replace missing values denoted by "?" with NaN
# data.replace('?', np.nan, inplace=True)

# # Convert columns to numeric
# data = data.apply(pd.to_numeric)

# # Drop rows with NaN values
# data.dropna(inplace=True)

# # Convert target variable to binary (0 = no heart disease, 1 = heart disease)
# data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# # Split the dataset into features and target
# X = data.drop('target', axis=1)
# y = data['target']

import pandas as pd
import numpy as np



hd = pd.read_csv('CEP/processed_cleveland.csv', na_values = '?')


hd['cp'].replace({1:'typical_angina', 2:'atypical_angina', 3: 'non-anginal_pain', 4: 'asymptomatic'}, inplace = True)
hd['restecg'].replace({0:'normal', 1:' ST-T_wave_abnormality', 2:'left_ventricular_hypertrophy'}, inplace = True)
hd['slope'].replace({1:'upsloping', 2:'flat', 3:'downsloping'}, inplace = True)
hd['thal'].replace({3:'normal', 6:'fixed_defect', 7:'reversible_defect'}, inplace = True)
hd['num'].replace({2:1, 3:1, 4:1}, inplace = True)

hd.dropna(how = 'any', inplace = True)

features = hd.columns.to_list()
# categorical_features = ['cp', 'thal', 'restecg', 'slope']
categorical_features = ['cp']
categorical_features = pd.get_dummies(hd[categorical_features].applymap(str))
features.remove('num')

features.remove('cp')
features.remove('thal')
features.remove('restecg')
features.remove('slope')


features.remove('chol')
features.remove('oldpeak')
features.remove('ca')


FEATURES = ['age','sex','cp','trestbps','fbs','thalach','exang','num']

hd = hd[FEATURES]

print(hd.head())
y = hd['num']
y.columns = ['target']
X = pd.concat([hd[features],categorical_features], axis = 1)
X.drop([92, 138, 163, 164, 251])
print(X.head())


# Address class imbalance if needed (e.g., using SMOTE or resampling)
# Example of resampling:
if y.value_counts()[0] != y.value_counts()[1]:
    majority = hd[hd.num == 0]
    minority = hd[hd.num == 1]
    
    minority_upsampled = resample(minority, 
                                  replace=True,    # sample with replacement
                                  n_samples=len(majority), # to match majority class
                                  random_state=42) # reproducible results
    
    data = pd.concat([majority, minority_upsampled])
    X = data.drop('num', axis=1)
    y = data['num']

y.columns = ['target']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
print(X_train.head())
X_train['cp'].replace({1:'typical_angina', 2:'atypical_angina', 3: 'non-anginal_pain', 4: 'asymptomatic'}, inplace = True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=300, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Plot training history
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
model.save('heart_disease_model.h5')
