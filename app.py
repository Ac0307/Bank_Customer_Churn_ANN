import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf

ann_model=tf.keras.models.load_model('my_model.h5')

with open('label_encoder.pkl','rb') as file:
    gen_lbl_enc=pickle.load(file)

with open('ohe.pkl','rb') as file:
    country_one_hot_enco=pickle.load(file)

with open('scale.pkl','rb') as file:
    scale=pickle.load(file)

st.title("Bank Customer Churn Prediction")


country=st.selectbox('Country',country_one_hot_enco.categories_[0])
gender=st.selectbox('Gender',gen_lbl_enc.classes_)
age=st.slider('Age',18,100)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,15)
products_number=st.slider('Products Number',1,4)
credit_card=st.selectbox('Has Credit Card',[0,1])
active_member=st.selectbox('Is Active Member',[0,1])

input_data=pd.DataFrame({
    "credit_score":[credit_score],
    "gender":[gen_lbl_enc.fit_transform([gender])[0]],
    "age":[age],
    "tenure":[tenure],
    "balance":[balance],
    "products_number":[products_number],
    "credit_card":[credit_card],
    "active_member":[active_member],
    "estimated_salary":[estimated_salary]
})

country_ohen=country_one_hot_enco.transform([[country]]).toarray()
country_en_df=pd.DataFrame(country_ohen,columns=country_one_hot_enco.get_feature_names_out(['country']))


df=pd.concat([input_data.reset_index(drop=True),country_en_df],axis=1)

df_scaled=scale.transform(df)

prediction=ann_model.predict(df_scaled)
prediction_prob=prediction[0][0]

st.write(f"Churn Probability is {prediction_prob}")

if prediction_prob>0.6:
    st.write("Customer is likely to churn")
else:
    st.write("Customer is unlikely to churn")