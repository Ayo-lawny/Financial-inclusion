# nstructions
# Install the necessary packages
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore') 

# Import you data and perform basic data exploration phase
data = pd.read_csv('Financial_inclusion_dataset.csv')
dx = data.copy()

dx.drop('uniqueid', axis = 1, inplace = True)

# Display general information about the dataset
dx.info()

# Create a pandas profiling reports to gain insights into the dataset
# %pip install pandas pandas-profiling
# import pandas_profiling as pp
# from pandas_profiling import ProfileReport

# Handle Missing and corrupted values
dx.isnull().sum()

# Remove duplicates, if they exist
dx.duplicated()
all_duplicates = dx[dx.duplicated(keep=False)]
df_no_duplicates = dx.drop_duplicates()

# Handle outliers, if they exist
def outlierRemoval(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtypes != 'O': # --------------------------------------- If the data type is not an object type
            Q1 = dataframe[i].describe()[4] # ---------------------------------- Identify lower Quartile
            Q3 = dataframe[i].describe()[6] # ---------------------------------- Identify the upper quartile
            IQR = Q3 - Q1 # ---------------------------------------------------- Get Inter Quartile Range
            minThreshold = Q1 - 1.5 * IQR # ------------------------------------ Get Minimum Threshold
            maxThreshold = Q3 + 1.5 * IQR # ------------------------------------ Get Maximum Threshold

            dataframe = dataframe.loc[(dataframe[i] >= minThreshold) & (dataframe[i] <= maxThreshold)]
    return dataframe


dx = outlierRemoval(dx)

# Encode categorical features
categoricals = dx.select_dtypes(include = ['object', 'category'])
numericals = dx.select_dtypes(include = 'number')

from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in dx.columns: # ...................................................... If the selected column is found in the general dataframe
        dx[i] = scaler.fit_transform(dx[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in dx.columns: # ...................................................... If the selected columns are found in the general dataframe
        dx[i] = encoder.fit_transform(dx[i])# .................................. encode it

# Based on the previous data exploration train and test a machine learning classifier
sel_cols = ['age_of_respondent', 'household_size','job_type', 'education_level', 'marital_status']
x = dx[sel_cols]

x = x
y = dx.bank_account
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 75, stratify = y)

# Based on the previous data exploration train and test a machine learning classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier() 
model.fit(xtrain, ytrain) 
cross_validation = model.predict(xtrain)
pred = model.predict(xtest) 

# save model
model = pickle.dump(model, open('Financial_Inclusion.pkl', 'wb'))

# STREAMLIT
# Create a streamlit application (locally) and add input fields for your features and a validation button at the end of the form
model = pickle.load(open('Financial_Inclusion.pkl','rb'))

st.markdown("<h1 style = 'color: #B06161; text-align: center;font-family: Arial, Helvetica, sans-serif; '>Financial Inclusion In Africa</h1>", unsafe_allow_html= True)
st.markdown("<h3 style = 'margin: -25px; color: #B06161; text-align: center;font-family: Arial, Helvetica, sans-serif; '> Created by Ayodeji</h3>", unsafe_allow_html= True)
st.image('Finance.png', width = 600)
st.markdown("<h2 style = 'color: #B06161; text-align: center;font-family: Arial, Helvetica, sans-serif; '>Background of the Study </h2>", unsafe_allow_html= True)

# st.markdown('<br><br>', unsafe_allow_html= True)

st.markdown("<p>The term financial inclusion means:  individuals and businesses have access to useful and affordable financial products and services that meet their needs, transactions, payments, savings, credit and insurance delivered in a responsible and sustainable way. The dataset contains demographic information and what financial services are used by approximately 33,600 individuals across East Africa. The ML model role is to predict which individuals are most likely to have or use a bank account.</p>",unsafe_allow_html= True)

st.sidebar.image('USER.png')

dx = data[['age_of_respondent', 'household_size','job_type', 'education_level', 'marital_status']]
st.write(dx.head())

# Import your ML model into the streamlit application and start making predictions given the provided features values
input_type = st.sidebar.radio("Select Your Prefered Input Style", ["Slider", "Number Input"])
if input_type == 'Slider':
    st.sidebar.header('Input Your Information')
    age_of_respondent = st.sidebar.number_input("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())
    household_size = st.sidebar.number_input("household_size", data['household_size'].min(), data['household_size'].max())
    job_type = st.sidebar.selectbox("Job Type", data['job_type'].unique())
    education_level = st.sidebar.selectbox("education_level", data['education_level'].unique())
    marital_status = st.sidebar.selectbox("marital_status", data['marital_status'].unique())

    
# else:
#     st.sidebar.header('Input Your Information')
#     age_of_respondent = st.sidebar.number_input("age_of_respondent", data['age_of_respondent'].min(), data['age_of_respondent'].max())
#     household_size = st.sidebar.number_input("household_size", data['household_size'].min(), data['household_size'].max())
#     job_type = st.sidebar.number_input("job_type", data['job_type'].min(), data['job_type'].unique())
#     education_level = st.sidebar.text_input("education_level", data['education_level'].min(), data['education_level'].unique())
#     marital_status = st.sidebar.text_input("marital_status", data['marital_status'].min(), data['marital_status'].unique())
    
st.header('Input Values')

# Bring all the inputs into a dataframe
input_variable = pd.DataFrame([{'age_of_respondent':age_of_respondent, 'household_size': household_size, 'job_type': job_type, 'education_level':education_level, 'marital_status':marital_status}])


st.write(input_variable)

# Standard Scale the Input Variable.
for i in numericals.columns:
    if i in input_variable.columns:
      input_variable[i] = StandardScaler().fit_transform(input_variable[[i]])
for i in categoricals.columns:
    if i in input_variable.columns: 
        input_variable[i] = LabelEncoder().fit_transform(input_variable[i])

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("<h2 style = 'color: #B06161; text-align: center; font-family: helvetica '>Model Report</h2>", unsafe_allow_html = True)

# Import your ML model into the streamlit application and start making predictions given the provided features values
if st.button('Press To Predict'):
    predicted = model.predict(input_variable)
    st.toast('Model Predicted')
    st.image('Done.png', width = 50)
    if predicted == 1:
       st.success(f'Will the individual have or use a bank account? Yes')
    else:
        st.success(f'Will the individual have or use a bank account? No') 

# Deploy your application on Streamlit share:
# Create a github and a streamlit share accounts
# Create a new git repo
# Upload your local code to the newly created git repo
# log in to your streamlit account an deploy your application from the git repo