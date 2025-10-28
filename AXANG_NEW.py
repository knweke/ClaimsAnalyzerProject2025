import streamlit_authenticator as stauth
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO
import requests
#from scipy.stats import zscore
from streamlit_option_menu import option_menu
import base64
import json
#from fpdf import FPDF
import os
#from xgboost.testing.data import joblib
#from ClaimsAppModel import pipeline
from streamlit_authenticator import Hasher
import yaml
from yaml.loader import SafeLoader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets

# https://pypi.org/project/streamlit-authenticator/

# --- Configuration for Sign-Up and Email ---
# Use a separate YAML file for credentials in production
# For simplicity, we'll keep it in memory for this example
# In a real app, you would load and save this to a file or database

# Add widgets
image = Image.open('C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\axa-logo.png')
st.image(image, width=80)

st.markdown('<p style="font-family: calibri; color:#000080; font-size: 38px;">AXA Mansard Insurance Plc</p>', unsafe_allow_html=True)
# st.markdown('## AXA Mansard Insurance Plc')
st.markdown("##### Life and Non-Life Insurance Claims Analyzer and Anomaly Detector")

image2 = Image.open('C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\insight-logo3.png')
st.image(image2, width=150)

# Set background image
BACKGROUND_IMAGE = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\BACKGROUND2.png'

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(BACKGROUND_IMAGE)

with open('C:/Users/Springrose/PyCharmMiscProject/user_cred.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

'''authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)'''

USER_CREDENTIALS = {
    'usernames': {
        'admin': {
            'email': 'admin@axamansard.com',
            'name': 'Admin',
            'is_email_verified': True
        }
    }
}

AUTHENTICATOR = stauth.Authenticate(
    USER_CREDENTIALS,
    'my_app_cookie',
    'my_app_key',
    cookie_expiry_days=30
)

try:
    AUTHENTICATOR.login()
except Exception as e:
    st.error(e)