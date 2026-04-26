import streamlit as st
import joblib as j
import matplotlib.pyplot as plt
import pandas as pd
#__________LOAD MODEL_________
@st.cache_resource
def load_model():
    
    sarima = j.load("sarima_model.joblib")
    rf = j.load("ML_TS_model.joblib")

    return sarima,rf

sarima_model, rf_model = load_model()

#_________ HEADER _____________
st.header("""-----Google Stock Price Prediction model-----""")

st.subheader("""I have built to separate model based on Traditional Time Series Model and Machine Learning Model using the Google stock price data form Yahoo finance for the past 2 year from 2024-2026 (Scroll last for Prediction).
             NOTE:
             1)- SARIMA Model is built using stock price data 
             2)- ML Model is built using daily change of price data 
                 so it will predict the percentage change from previous value.

              Model 1 = SARIMAX(0,1,0)(2,1,0,20) where seasonality is of 20 days
                        RMSE = ~22.87 , MAPE = ~ 6.7% , AIC ~ 2098.42
             
              Model 2 = Random Forest
                        RMSE = ~0.005437 , R2 = ~ 84.89%""")


df= pd.read_csv(r"google_2year_data (1).xls")
df = df.drop(index=[0,1],axis=0).reset_index(drop=True)
df['Date']= pd.to_datetime(df['Price'])
df = df.drop(columns='Price')
df['Close'] = pd.to_numeric(df['Close'])


st.line_chart(df.set_index('Date')['Close'])
st.write("Here is the data used for building the model:")
st.dataframe(df.head(10))

#_____________Creating Custom function for SARIMA MODEL___________
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

ts_data = pd.read_csv("Traditional_Model_Data_for_Prediction").drop(columns=["Unnamed: 0"])
ts_ml_data = pd.read_csv("ML_pred_data").drop(columns=["Unnamed: 0"])

ts_data['Date'] = pd.to_datetime(ts_data['Date'])
ts_ml_data['Date'] = pd.to_datetime(ts_ml_data['Date'])

# SARIMA Prediction
def input_date0(date):

    date = pd.to_datetime(date)
    
    exog = ts_data[ts_data['Date'] <= pd.to_datetime(date)].iloc[:,2:]

    exogen = np.array(exog)

    
    prediction  = sarima_model.forecast(steps = int(ts_data[ts_data['Date'] == pd.to_datetime(date)].index[0] + 1),
                           exog = exogen)
    return np.array(prediction.tail(1))[0]

#ML Prediciton
def input_date1(date):

    date = pd.to_datetime(date)
    
    exog = ts_ml_data[ts_ml_data['Date'] == pd.to_datetime(date)].iloc[:,2:]
    exogen = np.array(exog)

    
    prediction  = rf_model.predict(exogen)
    return prediction[0],exog.reset_index(drop= True).T



#_______INPUT FOR USING SARIMAX MODEL_______
st.sidebar.header("Input Date for using SARIMAX Model")
date0 = st.sidebar.date_input("Input Date ",
                              value=pd.to_datetime('2025-10-22'),
                              min_value=pd.to_datetime('2025-10-22'),
                              max_value=pd.to_datetime('2026-03-13'))

#________NOTR___________
st.header("Note: The prediction for the date will be based on the previous data, so the input date should be within the range of the data used for building the model. For SARIMAX model, the input date should be between 2025-10-22 and 2026-03-13. For ML model, the input date should be between 2026-01-07 and 2026-03-13.")
st.subheader("Please select the date for which you want to predict the stock price using the models from the sidebar. The prediction will be displayed below along with the RMSE for SARIMAX model and feature importance for ML model.")   
#_______INPUT FOR USING ML MODEL________
st.sidebar.header("Input Date for using ML Model")
date1 = st.sidebar.date_input("Input Date ",
                              min_value=pd.to_datetime('2026-01-07'),
                              value=pd.to_datetime('2026-02-22'),
                              max_value=pd.to_datetime('2026-03-13'))


#_______OUTPUT WHEN SARIMAX IS USED_______
if st.button('Predict Stock Price using SARIMAX Model'):
    col1,col2 = st.columns(2)
    col2.metric("RMSE","~22.87")
    if date0.weekday() in [5,6]:
        col1.write("Please Enter Date other than Saturday and Sunday")
    else:
        col1.metric(f"Prediction for {date0} is", input_date0(date0))


#_______OUTPUT WHEN ML IS USED_______

if st.button('Predict Stock Price using ML Model'):
    col1,col2 = st.columns(2)

    col2.write("Inputs used for prediction:")
    col2.write(input_date1(date1)[1])
    
    col2.write("Feature importance for the model:")
    f= rf_model.feature_names_in_
    i = rf_model.feature_importances_
    fi_df = pd.DataFrame({'Feature': f, 'Importance': i})
    col2.write("Feature Importance Chart:")
    col2.bar_chart(fi_df.set_index('Feature'))

    if date1.weekday() in [5,6]:
        col1.write("Please Enter Date other than Saturday and Sunday")
    else:
        col1.metric(f"Prediction for {date1} is", input_date1(date1)[0])
        
