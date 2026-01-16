import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf

##load the model
model = tf.keras.models.load_model('regressionmodel.keras',compile=False)

##Load the encoders and scaler
with open('label_encoder_gender_reg.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo_reg.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler = pickle.load(file)

st.title("Estimated salary prediction")
# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited' : [exited]
    
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

expected_columns = scaler.feature_names_in_
input_data = input_data[expected_columns]


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_salary = float(prediction[0][0])


prediction_salary = max(0, prediction_salary)
prediction_salary = min(prediction_salary, 300000)

st.write(f"Prediction salary :${prediction_salary:.2f}")
