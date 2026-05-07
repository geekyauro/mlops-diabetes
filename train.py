# 1. Import Packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Split the data into training and testing sets

from sklearn.model_selection import train_test_split

# Import the Logistic Regression model and evaluation metrics from scikit-learn

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

import joblib
 
import warnings

warnings.filterwarnings("ignore")
 
# 2. Read Dataset

diabetes_data = pd.read_csv("./data/diabetes.csv",header=0, sep=',')
 
print(diabetes_data.shape)
 
# 3. Handling missing values

cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

diabetes_data[cols] = diabetes_data[cols].replace(0, np.nan)
 
# Fill missing values in the 'Insulin' column with the mean value

diabetes_data['Insulin'].fillna(round(diabetes_data['Insulin'].mean()), inplace=True)
 
imputer = SimpleImputer(strategy='median')

# Fit the imputer on the data and transform it to fill missing values by median

diabetes_data[cols] = imputer.fit_transform(diabetes_data[cols])
 
diabetes_data['Outcome'] = diabetes_data['Outcome'].astype('int')
 
# 4. Split Features and Target

X = diabetes_data.drop('Outcome', axis=1)

Y = diabetes_data['Outcome']
 
# 5. Apply SMOTE to handle imbalanced dataset

from imblearn.over_sampling import SMOTE

smote = SMOTE()

transform_feature, transform_label = smote.fit_resample(X, Y)
 
# 6. Split Train & Test Split

X_train, X_test, Y_train, Y_test = train_test_split(transform_feature, transform_label, test_size=0.2, random_state=0)
 
# 7. Feature Scaling

scaler = StandardScaler()

x_train_scaler = scaler.fit_transform(X_train)

x_test_scaler = scaler.transform(X_test)
 
# 8. Logistic Regression Model

logit_model = LogisticRegression(solver='liblinear', C=1.0, penalty='l2', max_iter=10000, random_state=0)

# Train Logistic Regression Model

logit_model.fit(x_train_scaler, Y_train)
 
# 9. Model Evaluation

print(logit_model.score(x_test_scaler, Y_test))
 
Y_pred = logit_model.predict(x_test_scaler)

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
 
# 10. Save the Model

with open("model/diabetes_model.pkl", "wb") as file:

    joblib.dump((scaler, logit_model), file)
 
print("Model saved successfully as diabetes_model.pkl in model folder.")
 
 
 
