# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 23:20:13 2022

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model

loaded_model = pickle.load(open('C:/Users/DELL/Desktop/ml projects till deployment/diabetes prediction/trained_diabetic_model', 'rb'))


def diabetic_prediction(input_data):
    
    
    #changing input data to numpy array
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape array as we are predicting for one instance
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
        return 'person is non diabetic'
    else :
        return 'person is diabetic'
    
    
def main():
    
 
    st.title('DIABETES PREDICTION APP')
    
    Pregnancies = st.text_input('no of pregnancies')
    
    Glucose = st.text_input('Glucose level')
    
    BloodPressure = st.text_input('BloodPressure')
    
    SkinThickness = st.text_input('SkinThickness')
    
    Insulin = st.text_input('Insulin')
    
    BMI = st.text_input('BMI')
    
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    
    Age = st.text_input('Age')
    

    #code prediction

    diagonosis = ' '

    if st.button('Diabetes test result'):
        diagonosis = diabetic_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
    st.success(diagonosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    