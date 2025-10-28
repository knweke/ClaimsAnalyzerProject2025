import pandas as pd
#import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#import seaborn as sns
#from altair.vegalite.v5.theme import theme
#from jinja2.nodes import Slice
from scipy.stats import zscore
import joblib

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.csv'

df = pd.read_csv(filepath, encoding='cp1252')
pd.set_option('display.max_columns', None)

df.drop(['Claim_ID', 'Customer_Phone', 'Customer_Email',
         'Incident_Date', 'Claim_Submission_Date',
         'Policy_Start_Date', 'Policy_End_Date'
         ], axis=1, inplace=True)

suspicious_claims = df[df['Claim_Amount'] > 300000]
#suspicious_claims

non_suspicious_claims = df[df['Claim_Amount'] < 300000]
#non_suspicious_claims

df.drop(['Customer_Name', 'Policy_Number', 'Adjuster_Notes'], axis=1, inplace=True)

numerical_columns = ['Claim_Amount', 'Customer_Age', 'Premium_Amount']
categorical_columns = ['Location', 'Policy_Type',
                       'Claim_Type', 'Incident_Type',
                       'Claim_Status', 'Customer_Gender',
                       'Customer_Occupation'
                       ]

df_model = df

y = df_model['Fraud_Flag']
X = df_model.drop(['Fraud_Flag'], axis=1)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# reshape the array as we are predicting for one instance
#df_model.reshape(1,-1)

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_columns),
        ('categorical', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('prep4', preprocessor),
    ('clf4', XGBClassifier(objective= 'binary:logistic'))
])

pipeline.fit(X_train, y_train)

rf_predict = pipeline.predict(X_test)

print(classification_report(y_test, rf_predict))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print('Cross validation score:', cv_score)
print('Mean accuracy score:', cv_score.mean())