"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import pickle
import json
# Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import datetime
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
#To get away with warnings for presentability 
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm, skew 
# Modelling
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler # For normalization


def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------

    test_df=pd.read_csv('https://raw.githubusercontent.com/rufusseopa/Team_18_JHB_WhatTheHack_regression-predict-api-template/master/Data/Test.csv')
    riders_df=pd.read_csv('https://raw.githubusercontent.com/rufusseopa/Team_18_JHB_WhatTheHack_regression-predict-api-template/master/Data/Riders.csv')

    test_df = test_df.merge(riders_df, on='Rider Id')
    test_df.set_index('Order No', inplace=True) 
    test_df=test_df.drop(['Precipitation in millimeters'],axis=1)
    test_df['Temperature']=test_df.Temperature.fillna(test_df.Temperature.mean())
    

    testdf_time_columns = ['Placement - Time', 'Confirmation - Time', 'Arrival at Pickup - Time', 'Pickup - Time']
    for cols in testdf_time_columns:
        test_df[cols] = pd.to_datetime(test_df[cols])

    test_df['Delivery Rate'] = test_df['No_Of_Orders']/test_df['Age']
    test_df['Scaled Rating'] = test_df['Average_Rating']*(test_df['No_of_Ratings']/test_df['No_of_Ratings'].sum())
    test_df['placement_to_confirmation_time'] =  (test_df['Confirmation - Time'] - test_df['Placement - Time']).astype('timedelta64[s]')
    test_df['confirmation_to_arrivalpickup_time'] = (test_df['Arrival at Pickup - Time'] - test_df['Confirmation - Time']).astype('timedelta64[s]')
    test_df['arrivalpickup_to_pickup_time'] = (test_df['Pickup - Time'] - test_df['Arrival at Pickup - Time']).astype('timedelta64[s]')

    def traffic_ofthe_month(input_df):
        input_df['week_no'] = ''
        for i in range(0, len(input_df['Pickup - Day of Month'])):
            if input_df['Pickup - Day of Month'][i] < 8:
                input_df['week_no'][i] = '1st week'
            elif 8 <= input_df['Pickup - Day of Month'][i] < 15:
                input_df['week_no'][i] = '2nd week'
            elif 15 <= input_df['Pickup - Day of Month'][i] < 23:
                input_df['week_no'][i] = '3rd week'
            elif 23 <= input_df['Pickup - Day of Month'][i] <= 31:
                input_df['week_no'][i] = '4th week'
        input_df['week_no'] = input_df['week_no'].astype('category')

    traffic_ofthe_month(test_df)

    test_df['pickuphour'] = test_df['Pickup - Time'].dt.hour

    def peakness_hour(input_df):
        input_df['hour_status'] = ''
        for i in range(0, len(input_df['pickuphour'])):
            if 6 <= input_df['pickuphour'][i] <= 9:
                input_df['hour_status'][i] = 'morning_peakhour'
            elif 9 < input_df['pickuphour'][i] <= 12:
                input_df['hour_status'][i] = 'morning_offpeakhour'
            elif 12 < input_df['pickuphour'][i] <= 15:
                input_df['hour_status'][i] = 'afternoon_offpeakhour'
            elif 15 < input_df['pickuphour'][i] <= 18:
                input_df['hour_status'][i] = 'afternoon_peakhour'
            elif 18 < input_df['pickuphour'][i] <= 21:
                input_df['hour_status'][i] = 'night_peakhour'  
            elif (21 < input_df['pickuphour'][i]) or (input_df['pickuphour'][i] < 7):
                input_df['hour_status'][i] = 'night_offpeakhour'
        input_df['hour_status'] = input_df['hour_status'].astype('category')

    peakness_hour(test_df)
    test_columns_drop = ['User Id', 'Vehicle Type',
       'Placement - Day of Month', 'Placement - Weekday (Mo = 1)',
       'Placement - Time', 'Confirmation - Day of Month',
       'Confirmation - Weekday (Mo = 1)', 'Confirmation - Time',
       'Arrival at Pickup - Day of Month','Rider Id',
       'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
       'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long', 'No_Of_Orders',
       'Age', 'Average_Rating']

    test_df=test_df.drop(test_columns_drop, axis=1)

    test_df['Platform Type'] = test_df['Platform Type'].astype('category')
    test_df['Personal or Business']=test_df['Personal or Business'].astype('category')

    oce = ce.OneHotEncoder(['Platform Type', 'Personal or Business','hour_status', 'week_no'])


    test_df= oce.fit_transform(test_df)
    scalerr = MinMaxScaler()
    X_scaledr = scalerr.fit_transform(test_df)
    X_normalize_test = pd.DataFrame(X_scaledr)

    # ------------------------------------------------------------------------
    predict_vector =X_normalize_test
    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
