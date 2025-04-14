import streamlit as st
import pandas as pd
import pickle

# Load data (only for dropdowns)
df = pd.read_csv(r'C:\Users\Prathamesh\Desktop\123\data\heart.csv')

# Load models and encoders
with open(r'C:\Users\Prathamesh\Desktop\123\models\encoder.pickle', 'rb') as f:
    label_encoders = pickle.load(f)

with open(r'C:\Users\Prathamesh\Desktop\123\models\scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

with open(r'C:\Users\Prathamesh\Desktop\123\models\model.pickle', 'rb') as f:
    model = pickle.load(f)

st.title("Heart Disease Prediction App")
# User input form
age = st.selectbox(options=list(df['Age'].unique()), label='Choose your Age')
Sex = st.selectbox(options=list(df['Sex'].unique()), label='Choose your Sex')
ChestPainType = st.selectbox(options=list(df['ChestPainType'].unique()), label='Choose your ChestPainType')
RestingBP = st.selectbox(options=list(df['RestingBP'].unique()), label='Choose your RestingBP')
Cholesterol = st.selectbox(options=list(df['Cholesterol'].unique()), label='Choose your Cholesterol')
FastingBS = st.selectbox(options=list(df['FastingBS'].unique()), label='Choose your FastingBS')
RestingECG = st.selectbox(options=list(df['RestingECG'].unique()), label='Choose your RestingECG')
MaxHR = st.selectbox(options=list(df['MaxHR'].unique()), label='Choose your MaxHR')
ExerciseAngina = st.selectbox(options=list(df['ExerciseAngina'].unique()), label='Choose your ExerciseAngina')
Oldpeak = st.selectbox(options=list(df['Oldpeak'].unique()), label='Choose your Oldpeak')
ST_Slope = st.selectbox(options=list(df['ST_Slope'].unique()), label='Choose your ST_Slope')

# Build input DataFrame
input_data = {
    'Age': age,
    'Sex': Sex,
    'ChestPainType': ChestPainType,
    'RestingBP': RestingBP,
    'Cholesterol': Cholesterol,
    'FastingBS': FastingBS,
    'RestingECG': RestingECG,
    'MaxHR': MaxHR,
    'ExerciseAngina': ExerciseAngina,
    'Oldpeak': Oldpeak,
    'ST_Slope': ST_Slope
}

df_input = pd.DataFrame([input_data])

# Encode categorical columns using stored label encoders
for col in ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']:
    le = label_encoders[col]
    df_input[col] = le.transform(df_input[col])

# Scale numeric columns
num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df_input[num_cols] = scaler.transform(df_input[num_cols])



# Prediction
if st.button("Predict Heart Disease"):
    prediction = model.predict(df_input)
    result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
    st.success(f"Prediction: {result}")
