# Using Streamlit to build a web application that will be used to detect and flag potentially fraudulent claims

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from io import StringIO
import requests
from scipy.stats import zscore
from streamlit_option_menu import option_menu
import base64
import json
import requests
from fpdf import FPDF
import os
from xgboost.testing.data import joblib
from ClaimsAppModel import pipeline

filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.csv'

df = pd.read_csv(filepath, encoding='cp1252')
pd.set_option('display.max_columns', None)
#df.head()

joblib.dump(pipeline, 'fraud_detection_model.pkl')
model = joblib.load('fraud_detection_model.pkl')

# Add widgets
image = Image.open('C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\axa-logo.png')
st.image(image, width=70)

st.markdown('## AXA Mansard Insurance Plc')
st.markdown("##### Life and Non-Life Insurance Claims Analyzer and Anomaly Detector")

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

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        'Supporting Documentation',
        ['Methodology', 'How to Use', 'Code Base', 'Requirements'],
        icons=['book', 'person', 'code', 'book'],
        default_index=0,
    )

with st.sidebar.container():
    st.markdown("------------")  # Optional separator

# Setting Page Configuration
st.set_page_config(page_title="Life and Non-Life Insurance Claims Analyzer and Anomaly Detector",
                   layout="wide", page_icon="thumbs_up")

# ---- Load Data ----
@st.cache_data
def load_data():
    return pd.read_csv(filepath, encoding='latin-1')

# ---- User Auth (CSV-based) ----
def load_users():
    if os.path.exists("users.csv"):
        return pd.read_csv("users.csv")
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    # Load existing users or create a new DataFrame if file doesn't exist
    try:
        users = pd.read_csv("users.csv")
    except FileNotFoundError:
        users = pd.DataFrame(columns=["username", "password"])

    # Create a new row as a DataFrame
    new_user = pd.DataFrame([{"username": username, "password": password}])

    # ✅ Concatenate the new user row
    users = pd.concat([users, new_user], ignore_index=True)

    # Save the updated users list back to CSV
    users.to_csv("users.csv", index=False)
    return True

# ---- PDF Generator ----
def generate_pdf_report(data_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Claims Analysis Report", ln=True, align='C')
    pdf.ln(10)
    for line in data_summary:
        pdf.multi_cell(0, 10, line)
    filename = "claims_report.pdf"
    pdf.output(filename)
    return filename

# ---- Sidebar: Login/Register ----
st.sidebar.title("🔐 User Authentication")

users_df = load_users()
auth_mode = st.sidebar.radio("Choose Action", ["Login", "Register"])
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

auth_success = False
if auth_mode == "Login":
    if st.sidebar.button("Login"):
        if username in users_df["username"].values:
            user_row = users_df[users_df["username"] == username]
            if user_row["password"].values[0] == password:
                auth_success = True
            else:
                st.sidebar.error("Incorrect password.")
        else:
            st.sidebar.warning("User not found.")
else:
    if st.sidebar.button("Register"):
        if save_user(username, password):
            st.sidebar.success("Hurray! You have just been registered. Please login.")
        else:
            st.sidebar.warning("Username already exist.")
            auth_success = True

with st.sidebar.container():
    st.markdown("----------")
    st.markdown("#### Created and Designed By :")
    st.write("- Kenechukwu Nweke")
    st.write("- Abdulwahab Zakariyau")
    st.write("- Chinenye John")

# ---- MAIN APP ----
if auth_success:
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Dashboard", "📥 Export Report", "📄 PDF Report Summary"])

    with tab1:
        st.markdown('##### Analyze insurance claims, detect fraud, simulate claims optimization and export reports.')
        st.markdown("##### Welcome **" + username + "**! 👋")
        st_lottie = st.components.v1.html(f"""
        <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json" background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay></lottie-player>
        """, height=400)

    with tab2:
        st.header("📊 Claim Analytics Dashboard")
        st.subheader("Claims Data Preview")
        st.dataframe(df.head())

        st.subheader("High Value Claim Detection")
        threshold = df['Claim_Amount'].quantile(0.80)
        high_claims = df[df['Claim_Amount'] > threshold]
        st.write(f"- Claims above 80th percentile (₦{threshold:.2f})")
        st.dataframe(high_claims)

        threshold = df['Claim_Amount'].quantile(0.80)
        high_claims = df[df['Claim_Amount'] > threshold]
        st.write(f"- Number of suspicious high-amount claims: {len(high_claims)}")

        st.subheader("Frequent Claimants")
        frequent_claims = df['Customer_Name'].value_counts()
        frequent_customers = frequent_claims[frequent_claims > 3]
        st.write(f"- Customers with greater than 3 claims: {len(frequent_customers)}")
        st.dataframe(frequent_customers)

        st.subheader("Z-Score Anomaly Detection")
        df['z_score'] = zscore(df['Claim_Amount'])
        anomalies = df[df['z_score'].abs() > 3]
        st.write(f"- Anomalous claims (Z > 3): {len(anomalies)}")
        st.dataframe(anomalies)

        if 'Claim_Type' in df.columns:
            st.subheader("Cost Center Analysis")
            cost_centre = df.groupby('Incident_Type')['Claim_Amount'].sum().sort_values(ascending=False)
            st.bar_chart(cost_centre)

        st.subheader("Simulated Claim Capitation")
        df['capped_claim'] = np.where(df['Claim_Amount'] > 300000, 300000, df['Claim_Amount'])
        savings = df['Claim_Amount'].sum() - df['capped_claim'].sum()
        st.write(f"- Total Savings with Cap: ₦{savings:,.2f}")

        df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'])
        df['Policy_End_Date'] = pd.to_datetime(df['Policy_End_Date'])
        df['Incident_Date'] = pd.to_datetime(df['Incident_Date'])
        df['Claim_Submission_Date'] = pd.to_datetime(df['Claim_Submission_Date'])

        # Extract year and month of policy inception
        df['policy_inception_year'] = df['Policy_Start_Date'].dt.year
        df['policy_inception_month'] = df['Policy_Start_Date'].dt.month
        df['submission_delay_days'] = (df['Claim_Submission_Date'] - df['Incident_Date']).dt.days

        # Visualizing years of policy inception between 2022 to 2024
        st.subheader("Year on Year Chart by Policy Inception (2022-2024)")
        year_to_year = df.groupby('policy_inception_year')['Claim_Amount'].sum().sort_values(ascending=False)
        st.bar_chart(year_to_year)

        # Visualizing months of policy inception between 2022 to 2024
        st.subheader("Month to Month Chart by Policy Inception (2022-2024)")
        month_to_month = df.groupby('policy_inception_month')['Claim_Amount'].sum().sort_values(ascending=False)
        st.bar_chart(month_to_month)

        st.subheader("Other Cost Center Analysis")
        st.write(df['Policy_Type'].value_counts())
        #print('\n')
        corporate_claim_amount = df[df['Policy_Type'].isin(['Corporate'])]['Claim_Amount'].sum()
        st.write(f'- Corporate claims total amount: ₦{corporate_claim_amount:,.2f}')
        family_claim_amount = df[df['Policy_Type'].isin(['Family'])]['Claim_Amount'].sum()
        st.write(f'- Family claims total amount: ₦{family_claim_amount:,.2f}')
        individual_claim_amount = df[df['Policy_Type'].isin(['Individual'])]['Claim_Amount'].sum()
        st.write(f'- Individual claims total amount: ₦{individual_claim_amount:,.2f}')

        st.write(df['Claim_Type'].value_counts())
        #print('\n')
        gadget_claim_amount = df[df['Claim_Type'].isin(['Gadget'])]['Claim_Amount'].sum()
        st.write(f'- Gadget claims total amount: ₦{gadget_claim_amount:,.2f}')
        auto_claim_amount = df[df['Claim_Type'].isin(['Auto'])]['Claim_Amount'].sum()
        st.write(f'- Auto claims total amount: ₦{auto_claim_amount:,.2f}')
        Fire_claim_amount = df[df['Claim_Type'].isin(['Fire'])]['Claim_Amount'].sum()
        st.write(f'- Fire claims total amount: ₦{Fire_claim_amount:,.2f}')
        life_claim_amount = df[df['Claim_Type'].isin(['Life'])]['Claim_Amount'].sum()
        st.write(f'- Life claims total amount: ₦{life_claim_amount:,.2f}')
        health_claim_amount = df[df['Claim_Type'].isin(['Health'])]['Claim_Amount'].sum()
        st.write(f'- Health claims total amount: ₦{health_claim_amount:,.2f}')

        st.write(df['Incident_Type'].value_counts())
        #print('\n')
        death_claim_amount = df[df['Incident_Type'].isin(['Death'])]['Claim_Amount'].sum()
        st.write(f'- Death claims total amount: ₦{death_claim_amount:,.2f}')
        theft_claim_amount = df[df['Incident_Type'].isin(['Theft'])]['Claim_Amount'].sum()
        st.write(f'- Theft claims total amount: ₦{theft_claim_amount:,.2f}')
        fire_claim_amount = df[df['Incident_Type'].isin(['Fire'])]['Claim_Amount'].sum()
        st.write(f'- Fire claims total amount: ₦{fire_claim_amount:,.2f}')
        accident_claim_amount = df[df['Incident_Type'].isin(['Accident'])]['Claim_Amount'].sum()
        st.write(f'- Accident claims total amount: ₦{accident_claim_amount:,.2f}')
        illness_claim_amount = df[df['Incident_Type'].isin(['Illness'])]['Claim_Amount'].sum()
        st.write(f'- Illness claims total amount: ₦{illness_claim_amount:,.2f}')

        st.write(df['Claim_Status'].value_counts())
        #print('\n')
        approved_claims_total_amount_paid = df[df['Claim_Status'].isin(['Approved'])]['Claim_Amount'].sum()
        st.write(f'- Approved claims total amount paid: ₦{approved_claims_total_amount_paid:,.2f}')
        ave_appr_claims_total_amount_paid = df[df['Claim_Status'].isin(['Approved'])]['Claim_Amount'].sum() / 589
        st.write(f'- Average approved claims total amount paid: ₦{ave_appr_claims_total_amount_paid:,.2f}')

        # Minimum and maximum claim amount
        st.write(f"- Minimum claim amount: ₦{df['Claim_Amount'].min():,.2f}")
        st.write(f"- Maximum claim amount: ₦{df['Claim_Amount'].max():,.2f}")

        # Claim status count
        st.write(df['Claim_Status'].value_counts())
        #print('\n')
        # Customer gender count
        st.write(df['Customer_Gender'].value_counts())
        #print('\n')
        # Customer occupation count
        st.write(df['Customer_Occupation'].value_counts())

        # Customer count by Location
        st.write(df['Location'].value_counts())

        # Claim amount by Location
        claim_amount_abuja = df[df['Location'].isin(['Abuja'])]['Claim_Amount'].sum()
        st.write(f'- Abuja total claim amount: ₦{claim_amount_abuja:,.2f}')
        claim_amount_ibadan = df[df['Location'].isin(['Ibadan'])]['Claim_Amount'].sum()
        st.write(f'- Ibadan total claim amount: ₦{claim_amount_ibadan:,.2f}')
        claim_amount_kano = df[df['Location'].isin(['Kano'])]['Claim_Amount'].sum()
        st.write(f'- Kano total claim amount: ₦{claim_amount_kano:,.2f}')
        claim_amount_lagos = df[df['Location'].isin(['Lagos'])]['Claim_Amount'].sum()
        st.write(f'- Lagos total claim amount: ₦{claim_amount_lagos:,.2f}')
        claim_amount_ph = df[df['Location'].isin(['Port Harcourt'])]['Claim_Amount'].sum()
        st.write(f'- Port Harcourt total claim amount: ₦{claim_amount_ph:,.2f}')

    with tab3:
        st.header("📤 Export Data")
        if st.button("Export to Excel"):
            df.to_excel("processed_insurance.xlsx", index=False)
            with open("processed_insurance.xlsx", "rb") as file:
                b64 = base64.b64encode(file.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="processed_insurance.xlsx">📥 Download Excel</a>'
                st.markdown(href, unsafe_allow_html=True)

    with tab4:
        st.header("📄 Generate PDF Report")
    report_summary = [
        f"Total Claims: {len(df)}",
        f"High-Value Claims (>80th percentile): {len(high_claims)}",
        f"Frequent Claimants (>3 claims): {len(frequent_customers)}",
        f"Anomalous Z-Score Claims: {len(anomalies)}",
        f"Total Savings after Cap: ₦{savings:,.2f}"
    ]

    if st.button("Generate PDF Report"):
        pdf_file = generate_pdf_report(report_summary)
        with open(pdf_file, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_file}">📄 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.warning("Please, login to access the dashboard.")

st.divider()

st.markdown('###### Select, filter and view claim data by Policy ID')

# Use a form to group input widgets and a submit button # Option 1
with st.form(key="claim_form"):
    #st.write("Enter a policy number manually to search and view claim details")
    policy_number_input = st.text_input(
        "Policy Number :",
        help="e.g., POL6048764, POL7285232"
    ).upper() # Convert input to uppercase to match DataFrame

    submit_button = st.form_submit_button("Search")

    # Logic to handle the form submission
    if submit_button:
        if policy_number_input:
            # Filter the DataFrame based on the user's input
            filtered_df = df[df['Policy_Number'] == policy_number_input]

            if not filtered_df.empty:
                st.success(f"Found {len(filtered_df)} claim(s) for Policy ID: **{policy_number_input}**")
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.warning(f"No claim(s) found for Policy ID: **{policy_number_input}**")
        else:
            st.error("Please enter a policy number to search.")

# Get a list of unique policy numbers to use as options for the select box # Option 2
policy_numbers = sorted(df['Policy_Number'].unique())

# Add a "Select All" or placeholder option for the select box
policy_numbers.insert(0, 'Select a policy...')

# Create the select box widget
selected_policy_number = st.selectbox(
    "Select policy number from the list to view claim details",
    options=policy_numbers
)

# Logic to display data only after a valid policy is selected
if selected_policy_number and selected_policy_number != 'Select a policy...':
    # Filter the DataFrame based on the user's selection
    filtered_df = df[df['Policy_Number'] == selected_policy_number]

    st.success(f"Found **{len(filtered_df)}** claim(s) for Policy ID: **{selected_policy_number}**")
    #st.dataframe(filtered_df, use_container_width=True)
#elif selected_policy_number == 'Select a policy...':
    #st.info("View selected claim details")

    # Display selected fields
    selected_fields = filtered_df[['Customer_Name', 'Location', 'Policy_Type', 'Claim_Type', 'Incident_Type',
                                   'Customer_Age', 'Customer_Gender', 'Customer_Occupation',
                                   'Claim_Status', 'Premium_Amount', 'Claim_Amount', 'Incident_Date',
                                   'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date',
                                   'Fraud_Flag']]
    st.dataframe(selected_fields, use_container_width=True)
elif selected_policy_number == 'Select a policy...':
    st.info("View and extract selected claim details")

# ############################################################################### # Option 3
#if selected_policy_number:
    # Filter data by policy number
    #filtered_data = df[df["Policy_Number"] == selected_policy_number]

    #if not filtered_data.empty:
        #st.success("Policy found! Displaying selected fields:")

        # Display selected fields
        #selected_fields = filtered_data[['Location', 'Policy_Type', 'Claim_Type','Incident_Type',
                                         #'Customer_Age', 'Customer_Gender', 'Customer_Occupation',
                                         #'Claim_Status', 'Premium_Amount', 'Claim_Amount', 'Incident_Date',
                                         #'Claim_Submission_Date', 'Policy_Start_Date', 'Policy_End_Date',
                                         #'Fraud_Flag']]
        #st.dataframe(selected_fields)
    #else:
        #st.error("No claims found for the entered policy number.")

st.markdown("###### Select available options below to filter selected claim data for predictive analysis")

# Filter claims data with select boxes
# Get unique options for each filter
location = list(df['Location'].unique())
policy_type = list(df['Policy_Type'].unique())
claim_type = list(df['Claim_Type'].unique())
incident_type = list(df['Incident_Type'].unique())
customer_gender = list(df['Customer_Gender'].unique())
customer_age = df['Customer_Age'].astype(int)
customer_occupation = list(df['Customer_Occupation'].unique())
claim_status = list(df['Claim_Status'].unique())
premium_amount = df['Premium_Amount']
claim_amount = df['Claim_Amount']
policy_no = list(df['Policy_Number'].unique())
incident_date = list(df['Incident_Date'])
claim_submission_date = list(df['Claim_Submission_Date'])
policy_start_date = list(df['Policy_Start_Date'])
policy_end_date = list(df['Policy_End_Date'])

# Create the select boxes
selected_loc = st.selectbox("Location :", options=location)
selected_pt = st.selectbox("Policy type :", options=policy_type)
selected_ct = st.selectbox("Claim type :", options=claim_type)
selected_it = st.selectbox("Incident type :", options=incident_type)
selected_customer_gender = st.selectbox('Customer gender :', options=customer_gender)
selected_customer_age = st.selectbox('Customer age :', options=customer_age)
selected_customer_occupation = st.selectbox('Customer occupation :', options=customer_occupation)
selected_claim_status = st.selectbox('Claim status :', options=claim_status)
selected_premium_amount = st.number_input('Enter premium amount (₦) :',
                               min_value=0.00,
                               value=0.00,
                               step=0.01,
                               format="%.2f")
st.write("Premium amount entered : ₦", selected_premium_amount)
selected_incident_date = st.date_input("Incident date :", key='abc')
selected_claim_submission_date = st.date_input("Claim submission date :", key='def')
selected_policy_start_date = st.date_input("Policy start date :", key='ghi')
selected_policy_end_date = st.date_input("Policy end date :", key='jkl')

selected_claim_amount = st.number_input('Enter claim amount (₦) :',
                               min_value=0.00,
                               value=0.00,
                               step=0.01,
                               format="%.2f")
st.write("Claim amount entered : ₦", selected_claim_amount)
# selected_pn = st.selectbox("Policy ID :", options=policy_no)

# Creating filters
filtered_df = df.copy()

if selected_loc != '--Select--':
    filtered_df = filtered_df[filtered_df['Location'] == selected_loc]
if selected_pt != '--Select--':
    filtered_df = filtered_df[filtered_df['Policy_Type'] == selected_pt]
if selected_ct != '--Select--':
    filtered_df = filtered_df[filtered_df['Claim_Type'] == selected_ct]
if selected_it != '--Select--':
    filtered_df = filtered_df[filtered_df['Incident_Type'] == selected_it]
if selected_customer_gender != '--Select--':
    filtered_df = filtered_df[filtered_df['Customer_Gender'] == selected_customer_gender]
if selected_customer_age == 0:
    filtered_df = filtered_df[filtered_df['Customer_Age'] == selected_customer_age]
if selected_customer_occupation != '--Select--':
    filtered_df = filtered_df[filtered_df['Customer_Occupation'] == selected_customer_occupation]
if selected_claim_status != '--Select--':
    filtered_df = filtered_df[filtered_df['Claim_Status'] == selected_claim_status]
if selected_premium_amount:
    filtered_df = filtered_df[filtered_df['Premium_Amount'] == selected_premium_amount]
if selected_claim_amount:
    filtered_df = filtered_df[filtered_df['Claim_Amount'] == selected_claim_amount]
if selected_incident_date != '--Select--':
    filtered_df = filtered_df[filtered_df['Incident_Type'] == selected_incident_date]
if selected_claim_submission_date != '--Select--':
    filtered_df = filtered_df[filtered_df['Claim_Submission_Date'] == selected_claim_submission_date]
if selected_policy_start_date != '--Select--':
    filtered_df = filtered_df[filtered_df['Policy_Start_Date'] == selected_policy_start_date]
if selected_policy_end_date != '--Select--':
    filtered_df = filtered_df[filtered_df['Policy_End_Date'] == selected_policy_end_date]

#st.divider()

#filter_button = st.markdown("###### Filtered claims data :")
#if filter_button not in st.session_state:
    #st.session_state.filter_button = True
    #st.dataframe(filtered_df)

#st.divider()

# Filter claims data with policy number
#policy_number = df['Policy_Number'].unique()
#selected_id = st.selectbox(
    #"Select or enter policy number manually to filter customer claim data. This parameter selects a row from the table.",
    #policy_number)
#filtered_df = df[df['Policy_Number'] == selected_id]
#st.markdown('###### Filtered claim data by Policy ID :')
#st.dataframe(filtered_df)

st.divider()

# Create show claims data button
st.markdown("###### Click the button below to show uploaded claims data")
df = pd.read_csv(filepath)
if st.button("Show Uploaded Claims Data"):
    report = st.write(df)

# Create hide data button
if "Show Claims Data" not in st.session_state:
    st.session_state.show_button = True
def hide_session_state():
    st.session_state.show_button = False
if st.session_state.show_button:
    st.button("Hide Claims Data", on_click=hide_session_state)
else:
    st.write("Hide Claims Data")

st.divider()

# Create button to trigger prediction
st.write(f"Make sure to enter selected claim details correctly. If otherwise, default prediction result will return '0'.")
if st.button("View Prediction Result"):
    input_data = pd.DataFrame([{
        'Location' : selected_loc,
        'Policy_Type' : selected_pt,
        'Claim_Type' : selected_ct,
        'Incident_Type' : selected_it,
        'Customer_Age' : selected_customer_age,
        'Customer_Gender' : selected_customer_gender,
        'Customer_Occupation' : selected_customer_occupation,
        'Claim_Status' : selected_claim_status,
        'Premium_Amount' : selected_premium_amount,
        'Claim_Amount' : selected_claim_amount
        }])

    #if selected_claim_amount == 0.00:
        #st.info('Claim amount must be greater than 0. To get accurate prediction result, please enter selected claim details.')

    # Claim threshold
    threshold = 400000

    # Default claim prediction result : '0'
    prediction = model.predict(input_data)[0]

    if selected_claim_amount == 0.00:
        st.info('Claim amount must be greater than 0. Hence, prediction result could not be ascertained. To get accurate prediction result, please enter selected claim details.')
    #else:
        #if prediction == 0:
            #st.info('Claim amount must be greater than 0. Hence, prediction result could not be ascertained. To get accurate prediction result, please enter selected claim details.')
        #else:
            #if prediction == 1:
                #st.info('Claim amount must be greater than 0. Hence, prediction result could not be ascertained. To get accurate prediction result, please enter selected claim details.')'''

    st.markdown(f"##### Claim prediction result : '{int(prediction)}'")

    # Condition to predict if a claim is fraudulent (1) or not (0).
    # Initialize the prediction variable
    if selected_claim_amount > threshold:
        st.warning(f"The claim amount of ₦**{selected_claim_amount:,}** is above the payment threshold of ₦**{threshold:,}**. It has been flagged as potentially fraudulent.")
    else:
        if selected_claim_amount < threshold:
            st.info(f"The claim amount of ₦**{selected_claim_amount:,}** falls within the payout threshold of ₦**{threshold:,}**.")

    if prediction == 1:
        st.error('⚠️ Fraud alert : Potentially fraudulent claim has been detected. Please, exercise due diligence.')
        st.warning('❌ Remark : This claim transaction has been flagged as suspicious and requires further review.')
    else:
        st.success('✅ Claim payout request passed credibility check.')
        st.success('🔍 After careful analysis, transaction is presumed legitimate for approval and further processing.')

st.divider()

st.info(f"Disclaimer: Prediction is made by a Machine Learning model and might require further validation to ensure its accuracy.")

st.divider()

# ...............................Space for API requests

st.markdown("###### Claim Approval/Authorization Status")
claim_status_2 = st.selectbox('Approve, reject, deny or flag claim request :', ['Approved', 'Rejected',
                                                                                'Denied', 'Flagged'])
comment = st.text_area('Comment (Reason for comment) :')

st.markdown("###### Document Review Section")
review = st.text_area("Claim documentation remark(s) :")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("Approve"):
        st.success("Document Approved")
        # Add logic here to update approval status in a database
        # or trigger an approval workflow.
        st.session_state.status = "Approved for further processing."
with col2:
    if st.button("Reject"):
        st.info("Document Rejected")
        # Add logic here to update rejection status in a database
        # or trigger a rejection workflow.
        st.session_state.status = "Rejected"
with col3:
    if st.button("Deny"):
        st.info("Document Denied")
        # Add logic here to update denial status in a database
        # or trigger a denial workflow.
        st.session_state.status = "Denied"
with col4:
    if st.button("Flag"):
        st.info("Document Flagged")
        # Add logic here to update flagged status in a database
        # or trigger a flagged workflow.
        st.session_state.status = "Flagged"
with col5:
    if st.button("Awaiting Decision"):
        st.info("Document Awaiting Decision")
        # Add logic here to update awaiting decision status in a database
        # or trigger an awaiting decision workflow.
        st.session_state.status = "Document awaiting decision until confirmed."

if "status" in st.session_state:
    st.write(f"Current Status : {st.session_state.status}")

# Creating a button that triggers an API call and then displays the input results using internet connection and API services
with st.form(key='claims_form'):
    # Create the submit button
    submit_button = st.form_submit_button(label='Submit Claim Request')

if submit_button:
    #st.write(f"Location: {selected_loc}, Policy type: {selected_pt}, Claim type: {selected_ct}, Incident type: {selected_it}, Customer age: {selected_customer_age}, Customer gender: {selected_customer_gender}, Customer occupation: {selected_customer_occupation}, Claim status: {selected_claim_status}, Premium amount: {selected_premium_amount}, Claim amount: {selected_claim_amount}, Authorization: {claim_status_2}, Comment: {comment}, Current Status: {st.session_state.status}")
    st.write(f"- Policy ID: {selected_policy_number}")
    st.write(f"- Location: {selected_loc}")
    st.write(f"- Policy type: {selected_pt}")
    st.write(f"- Claim type: {selected_ct}")
    st.write(f"- Incident type: {selected_it}")
    st.write(f"- Customer age: {selected_customer_age}")
    st.write(f"- Customer gender: {selected_customer_gender}")
    st.write(f"- Customer occupation: {selected_customer_occupation}")
    #st.write(f"- Claim status: {selected_claim_status}")
    st.write(f"- Premium amount: {selected_premium_amount}")
    st.write(f"- Claim amount: {selected_claim_amount}")
    st.write(f"- Authorization: {claim_status_2}")
    st.write(f"- Comment: {comment}")
    st.write(f"- Document review: {review}")
    st.write(f"- Current Status: {st.session_state.status}")

st.divider()

# This space will be used to make API calls
# This space will also be used to invoke requests to make GET, POST, PUT, DELETE, or other HTTP requests as needed
# Code samples that will be used for API calls and requests

#url = "https://jsonplaceholder.typicode.com/todos/1"
#response = requests.get(url)
#data = response.json()
#st.write("Data from API:")
#st.write(data)

# This space will handle user input and display results: Integrate requests calls with Streamlit widgets to allow user interaction.
# For instance, you could have a button that triggers an API call and then displays the results

#if st.button("Fetch New Data"):

#new_data_url = "https://api.example.com/some_endpoint"
#try:
    #new_response = requests.get(new_data_url)
    #new_data = new_response.json()
    #st.write("Newly fetched data:")
    #st.write(new_data)
#except requests.exceptions.RequestException as e:
        #st.error(f"Error fetching data: {e}")

# This space will be used to manage secrets (for API keys)
# If your API calls require API keys or other sensitive information
# Use Streamlit's secrets management to securely store and access them
# In your .streamlit/secrets.toml file:

#my_api_key = "your_api_key_here"

# In your Streamlit app:
#api_key = st.secrets["my_api_key"]
#headers = {"Authorization": f"Bearer {api_key}"}

# Headers will be used to initiate requests.get() or requests.post() calls

# EMAIL NOTIFICATION TO CUSTOMERS WITH TIMESTAMP IS REQUIRED