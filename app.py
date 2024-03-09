import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_scaled, y)

# Define a function to predict outcome for new data
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    # Scale the input features
    new_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    
    # Predict outcome
    prediction = model.predict(new_data)
    return prediction[0]

# Create Streamlit app
st.title('Diabetes Prediction')

# Add input fields for user input
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=17, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, step=1)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=122, step=1)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=99, step=1)
insulin = st.number_input('Insulin Level', min_value=0, max_value=846, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, step=0.1)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, step=0.001)
age = st.number_input('Age', min_value=21, max_value=81, step=1)

# Predict outcome on button click
if st.button('Predict'):
    prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    if prediction == 1:
        st.write("The model predicts that the individual has diabetes.")
    else:
        st.write("The model predicts that the individual does not have diabetes.")
