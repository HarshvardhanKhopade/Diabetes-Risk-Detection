import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'H:\Dataset\diabetes_dataset.csv')

le = LabelEncoder()

df['smoking_history'] = le.fit_transform(df['smoking_history'])


gender_encoder = OneHotEncoder(sparse=False)
gender_encoded = gender_encoder.fit_transform(df[['gender']])
gender_categories = gender_encoder.categories_[0]
gender_df = pd.DataFrame(gender_encoded, columns=[f'gender_{category}' for category in gender_categories])
df = pd.concat([df, gender_df], axis=1)


x = df.drop(['diabetes', 'gender'], axis=1)
y = df['diabetes']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)

st.title('Diabetes Risk Detection')

col1, col2 = st.columns(2)

with col1:
    gender_options = gender_categories.tolist()
    Gender = st.selectbox('Gender', gender_options)

with col2:
    Age = st.number_input('Age', min_value=1, max_value=110, value=40)

col3, col4 = st.columns(2)

with col3:
    hypertension = st.number_input('hypertension', min_value=0, max_value=1, step=1)

with col4:
    heart_disease = st.number_input('heart_disease', min_value=0, max_value=1, step=1)

smoking_history_options = ['never', 'No Info', 'current', 'former', 'ever', 'not current']
Smoking_history = st.selectbox('Smoking History', smoking_history_options)

# Apply label encoding to 'Smoking History'
Smoking_history_encoded = le.transform([Smoking_history])[0]

BMI = st.number_input('BMI', min_value=10.0, max_value=96.0, value=27.0)

Hemoglobin_level = st.number_input('HbA1c Level', min_value=3.5, max_value=9.0, value=5.5)

Blood_glucose_level = st.number_input('Blood Glucose Level', min_value=80, max_value=300, value=138)

if st.button('Detect'):
    query = np.array([[Age, hypertension, heart_disease, Smoking_history_encoded, BMI, Hemoglobin_level, Blood_glucose_level] + gender_encoder.transform([[Gender]]).tolist()[0]])
    result = DT.predict(query)[0]
    if result == 0:
        st.success('You have no diabetes')
    else:
        st.error('You have diabetes')

