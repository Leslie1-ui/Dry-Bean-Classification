import pandas as pd
import numpy as np
import streamlit as st
import joblib

#Loading the saved components
model=joblib.load('Logistic_Regression_dry_bean_model.pkl')
encoder=joblib.load('label_encoder.pkl')
scaler=joblib.load('scaler.pkl')

st.title('Dry Bean Classification app')
st.write('This model predicts the bean class according to the feature inputs as below. Kindly insert your value: ')

#The features you outline here should match the features used during the training of the model

#Input features

Perimeter = st.slider('Perimeter(Size/edge length): ', 538.0, 1848.0)
Eccentricity = st.slider('Eccentricity(Round vs long): ', 0.4, 1.0)
Solidity = st.slider('Regular vs dented: ', 0.9, 1.0)
roundness = st.slider('Shape type: ', 0.6, 1.0)
ShapeFactor1 = st.slider('Compact vs elongated: ', 0.00, 0.01)
ShapeFactor2 = st.slider('Slenderness: ', 0.00, 0.01)
ShapeFactor4 = st.slider('Edge complexity: ', 0.98, 1.00)

#Preparing input features for the model
features = np.array([[Perimeter, Eccentricity, Solidity, roundness, ShapeFactor1, ShapeFactor2,	ShapeFactor4]])
scaled_features = scaler.transform(features)

#Prediction
if st.button('Predict Bean Type'):
    prediction_encoded = model.predict(scaled_features)
    prediction_label = encoder.inverse_transform(prediction_encoded)[0]

    st.success(f'Predict Bean Type: {prediction_label}')