import pickle
import streamlit as st
import pandas as pd
import numpy as np

with open('log_reg_model.pkl', 'rb') as file:
    log_reg = pickle.load(file)

st.title('Mobile Usage Categorizer')
st.write("Enter the following metrics to know your category")

apps_installed = st.number_input("Number of Apps Installed", min_value=0.0)
app_usage = st.number_input("Average App Usage Time (Min/Day)", min_value=0.0)
screen_time = st.number_input("Average Screen On Time (Hours/Day)",min_value=0.0)
data_usage = st.number_input("Average Data Usage (MB/Day)", min_value=0.0)
data_usage = np.log(data_usage)
battery_drain = st.number_input("Average Battery Drain (mAH/Day)",min_value=0.0)

def category():
    X_test = pd.DataFrame({'App Usage Time (min/day)':[app_usage],'Screen On Time (hours/day)':[screen_time],'Battery Drain (mAh/day)':[battery_drain],
                           'Number of Apps Installed':[apps_installed],'Log_Data_Usage':[data_usage]})

    y_pred = log_reg.predict(X_test)
    y = ''

    if y_pred[-1] == 1: y = 'Light Usage' 
    elif y_pred[-1] == 2: y = 'Moderate Usage'
    elif y_pred[-1] == 3: y = 'Average Usage'
    elif y_pred[-1] == 4: y = 'Heavy Usage'
    else: y = 'Extreme Usage'

    return y

if st.button('Mobile Usage'):
    if apps_installed and app_usage and screen_time and data_usage and battery_drain: 
        predicted_category = category()
        
        st.success(f'Mobile Usage Category: {predicted_category}')
    else:
        st.warning('Please enter all metrics')


st.sidebar.header('About')
st.sidebar.write('''
This app uses a logistics regression model trained on mobile usage data to predict usage categories.
The model takes into account:
- Number of Apps Installed
- App Usage Time
- Screen On Time
- Data Usage
- Battery Drain
''')

st.sidebar.header('Metric Description ')
st.sidebar.write('''
- Number of Apps Installed: Total number of apps the user has installed on the device excluding bloatware.
- App Usage Time: Average daily time spent on mobile applications, measured in minutes per day.
- Screen On Time: Average hours per day the screen is active.
- Data Usage: Average daily mobile data consumption in megabytes.
- Battery Drain: Average daily battery consumption in mAh.
''')
