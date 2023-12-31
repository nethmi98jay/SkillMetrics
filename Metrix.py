import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

st.title('Employee Proficiency Prediction App')

st.write("""
This app predicts the proficiency percentage of an employee based on their skills!
""")

data = pd.read_csv('employee_skill_dataset.csv')

def calculate_proficiency(row):
    programming_weight = 0.3
    networking_weight = 0.2
    db_management_weight = 0.1
    communication_weight = 0.1
    teamwork_weight = 0.1
    emotional_intelligence_weight = 0.05
    leadership_weight = 0.05
    team_building_weight = 0.05
    risk_management_weight = 0.05

    proficiency_percentage = (
        row['Programming_Skill'] * programming_weight +
        row['Networking_Skill'] * networking_weight +
        row['Database_Management'] * db_management_weight +
        row['Communication'] * communication_weight +
        row['Teamwork_Collaboration'] * teamwork_weight +
        row['Emotional_Intelligence'] * emotional_intelligence_weight +
        row['Leadership'] * leadership_weight +
        row['Team_Building'] * team_building_weight +
        row['Risk_Management'] * risk_management_weight
    )

    return proficiency_percentage

    
data['Proficiency_Percentage'] = data.apply(calculate_proficiency, axis=1) 

# Split data
X = data[['Programming_Skill', 'Networking_Skill', 'Database_Management', 'Communication', 'Teamwork_Collaboration', 'Emotional_Intelligence', 'Leadership', 'Team_Building', 'Risk_Management']]
y = data['Proficiency_Percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model 
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
def get_unique_role():
    get_role = set()
    with open('employee.csv', mode='r') as file:
        reader = pd.read_csv(file)
    return sorted(set(reader['Job Title']))

# Create sidebar for user input
st.sidebar.header('Enter Employee Skills')
employee_name = st.sidebar.text_input('Enter Employee Name')
selected_job =  st.sidebar.selectbox(
    'Select your job role',
    (get_unique_role()))
programming = st.sidebar.slider('Programming', 0, 100, 50)
networking = st.sidebar.slider('Networking', 0, 100, 50) 
db = st.sidebar.slider('Database Management', 0, 100, 50)
communication = st.sidebar.slider('Communication', 0, 100, 50)
teamwork = st.sidebar.slider('Teamwork', 0, 100, 50)  
emotional_intelligence = st.sidebar.slider('Emotional Intelligence', 0, 100, 50)
leadership = st.sidebar.slider('Leadership', 0, 100, 50)
team_building = st.sidebar.slider('Team Building', 0, 100, 50)
risk_management = st.sidebar.slider('Risk Management', 0, 100, 50)

# Make predictions
if st.button('Predict'):
    skills = [programming, networking, db, communication, teamwork, emotional_intelligence, leadership, team_building, risk_management]
    pred = regressor.predict([skills])[0]
    st.write(f'Employee Name: {employee_name}')
    st.write(f'Predicted Proficiency Percentage: {pred:.2f}%')
    
    suggested_training = ""
    if pred < 50:
        suggested_training = " Basic Training"
        st.write("Suggested Training Level: Basic Training")
    elif pred < 70:
        suggested_training = "Intermediate Training"
        st.write("Suggested Training Level: Intermediate Training")
    else:
        suggested_training = "Advanced Training"
        st.write("Suggested Training Level: Advanced Training")
        
    report = f"""
    AI Based Employee Proficiency Report
    Employee Name: {employee_name} 
    Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Programming Skill: {programming}
    Networking Skill: {networking}  
    Database Management: {db}
    Communication: {communication}
    Teamwork Collaboration: {teamwork}   
    Emotional Intelligence: {emotional_intelligence}
    Leadership: {leadership}
    Team Building: {team_building}
    Risk Management: {risk_management}
    Proficiency Percentage: {pred:.2f}% 
    Suggested Training Level: {suggested_training}
    """
    
    st.download_button(
       label="Download Report",
       data=report,
       file_name=f"{employee_name}_report.txt" 
    )
