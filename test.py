import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("Employee Proficiency Prediction App")

st.sidebar.header("Input Parameters")

# Load data
data = pd.read_csv('employee.csv') 

# Get unique roles
roles = sorted(data['Job Title'].unique())
role = st.sidebar.selectbox("Select Job Role", roles)
data = data[data['Job Title'] == role]

# Get user input features
Programming_Skill = st.sidebar.slider("Programming Skill", 1, 100, 5) 
Networking_Skill = st.sidebar.slider("Networking Skill", 1, 100, 5)
Database_Management = st.sidebar.slider("Database Management", 1, 100, 5)
Communication = st.sidebar.slider("Communication", 1, 100, 5)
Teamwork_Collaboration = st.sidebar.slider("Teamwork Collaboration", 1, 100, 5)  
Emotional_Intelligence = st.sidebar.slider("Emotional Intelligence", 1, 100, 5)
Leadership = st.sidebar.slider("Leadership", 1, 100, 5)
Team_Building = st.sidebar.slider("Team Building", 1, 100, 5)
Risk_Management = st.sidebar.slider("Risk Management", 1, 100, 5)


Programming_Skill = Programming_Skill*0.3
Networking_Skill = Networking_Skill*0.2
Database_Management = Database_Management*0.1
Communication  = Communication*0.1
Teamwork_Collaboration = Teamwork_Collaboration*0.1
Emotional_Intelligence = Emotional_Intelligence*0.05
Leadership = Leadership*0.05
Team_Building = Team_Building*0.05
Risk_Management = Risk_Management*0.05



regressor = pickle.load(open('RandomForest_Regressor_model.pkl', 'rb'))
# Make prediction
new_data = [[Programming_Skill, Networking_Skill, Database_Management, Communication, 
             Teamwork_Collaboration, Emotional_Intelligence, Leadership, Team_Building, Risk_Management]]
y_pred = regressor.predict(new_data)[0]

st.subheader("Predicted Proficiency Percentage")
st.write(y_pred)

# Suggest training
if y_pred < 15:
    training = "Basic Training" 
elif y_pred < 27:
   training = "Intermediate Training"
else:
   training = "Advanced Training"
   
st.subheader("Suggested Training")   
st.write(training)