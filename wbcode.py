import tensorflow as tf
import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title = "Work Burnout Prediction" , page_icon="🗒️")
st.title("Work Burnout Prediction 😵‍💫😩🗒️")
st.title("")

model = tf.keras.models.load_model('wbmodel.h5')
le0 = joblib.load('wble0.h5')
ley = joblib.load("wbley.h5")

day = st.radio("Day Type",["Weekday","Weekend"])

hours = st.slider("Work Hours" , 3.0 , 12.2)

screen = st.slider("Screen Time Hours" , 4.51 , 15.7)

meetings = st.slider("Meetings Count" , 0 , 10)

breaks = st.slider("Breaks Taken" , 1 , 5)

after = st.radio("After Hours Work" ,[0,1])

sleep = st.slider("Sleep Hours" , 4.5 , 10.8)

task = st.slider("Task Completion Rate" , 40.0 , 107.0)

burnout = st.slider("Burnout Score" , 2.5 , 144.0)




day = le0.transform([day])[0]



btn = st.sidebar.button("pred")

if btn :
    input_data = np.array([[day,hours,screen,meetings,breaks,after,sleep,task,burnout]])
    pred = model.predict(input_data)
    pred = np.argmax(pred)  
    pred = ley.inverse_transform([pred])[0]

    st.sidebar.info(pred)
