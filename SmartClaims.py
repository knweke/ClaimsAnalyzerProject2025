import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import os
import time
from datetime import datetime
import pickle
from typing import List
#from openai import OpenAI
import openai
import shap
from tqdm import tqdm
from joblib import dump, load
from dotenv import load_dotenv

""" 
fraud_pipeline_with_embeddings.py 
Demonstrates use of OpenAI text-embedding-ada-002 embeddings, combined with structured features to train an 
XGBoost classifier for potentially fraudulent claims.

Project Title: Fraudulent claims detection with embeddings + XGBoost pipeline.

We’ll: 
- Use Adjuster_Notes (free-text) → embed with text-embedding-ada-002. 
- Use selected structured features (e.g., Claim_Amount, Customer_Age, Premium_Amount, 
policy/claim dates transformed to numeric). 
- Encode categorical features (Policy_Type, Claim_Type, Incident_Type, Claim_Status, 
Customer_Gender, Customer_Occupation, Location). 
- Train/test split, fit XGBClassifier, evaluate with ROC AUC + classification report.
"""

# --- CONFIG ---
# Load environment variables from .env file
load_dotenv()

# Set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.organization = "org-IMyQr2JMpeWYpnuIx1JbWhGH"
EMBED_MODEL = "text-embedding-ada-002"
CACHE_FILE = "embeddings_cache.pkl"

# print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))

# --- EMBEDDINGS FUNCTION ---
def get_embeddings(texts, model=EMBED_MODEL):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}

    results = []
    for t in tqdm(texts, desc="Embedding texts"):
        key = t if t else ""
        if key in cache:
            results.append(cache[key])
        else:
            emb = openai.Embedding.create(model=model, input=key)["data"][0]["embedding"]
            cache[key] = emb
            results.append(emb)
            time.sleep(0.05) # simple rate limit cushion

            print(len(emb))  # Should be 1536 for ada-002
            print(emb[:5])  # Preview first few values

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    return np.array(results)

# --- LOAD DATA ---
filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.xlsx'

# df = pd.read_csv(filepath, sep="\t")  # pd.read_csv
df = pd.read_excel(filepath) # pd.read_excel (.xlsx)

print("Shape:", df.shape)
print(df.head())

# --- FEATURE ENGINEERING ---
# Target
y = df['Fraud_Flag'].astype(int)

# Numeric features
num_cols = ["Claim_Amount", "Customer_Age", "Premium_Amount"]

# Date features -> transform to days
#for col in ["Incident_Date", "Claim_Submission_Date", "Policy_Start_Date", "Policy_End_Date"]:
    #df[col] = pd.to_datetime(df[col], errors="coerce")

# Date features -> transform to days
# Converting dates to datetime objects to aid claim submission analysis
df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], errors="coerce")
df['Policy_End_Date'] = pd.to_datetime(df['Policy_End_Date'], errors="coerce")
df['Incident_Date'] = pd.to_datetime(df['Incident_Date'], errors="coerce")
df['Claim_Submission_Date'] = pd.to_datetime(df['Claim_Submission_Date'], errors="coerce")

df["submission_delay_days"] = (df["Claim_Submission_Date"] - df["Incident_Date"]).dt.days
df["policy_duration_days"] = (df["Policy_End_Date"] - df["Policy_Start_Date"]).dt.days
num_cols.extend(["submission_delay_days", "policy_duration_days"])

# Categorical features
cat_cols = ["Policy_Type", "Claim_Type", "Incident_Type", "Claim_Status",
            "Customer_Gender", "Customer_Occupation", "Location"]

# Text (to embed)
df["Adjuster_Notes"] = df["Adjuster_Notes"].fillna("")

# --- EMBEDDINGS ---
# "Claim_Type", "Incident_Type", "Policy_Type" ... can be added to X_text to enrich the model
X_text = get_embeddings(df["Adjuster_Notes"].tolist())

# --- PREPROCESS STRUCTURED FEATURES ---
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), num_cols),
        ('categorical', OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

X_struct = preprocessor.fit_transform(df)

# --- CONCAT STRUCTURED + EMBEDDINGS ---
from scipy.sparse import hstack, csr_matrix
X = hstack([X_struct, csr_matrix(X_text)])  # OneHotEncoder gives sparse, embeddings give dense

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# --- TRAIN XGBOOST ---
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
    )

xgb_model.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

# --- CROSS VALIDATION ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_score = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print('Cross validation score:', cv_score)
print('Mean accuracy score:', cv_score.mean())

# --- Build feature names for SHAP ---
# Numeric + categorical names
cat_encoder = preprocessor.named_transformers_["categorical"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
feature_names_struct = num_cols + cat_feature_names

# Embedding feature names
embedding_dim = X_text.shape[1]
feature_names_embed = [f"emb_{i}" for i in range(embedding_dim)]
feature_names = feature_names_struct + feature_names_embed

# --- SHAP EXPLAINER ---
explainer = shap.TreeExplainer(xgb_model)

# shap_values will be a matrix: [n_samples, n_features]
shap_values = explainer.shap_values(X_test)

"""
Grouping all embedding dimensions as one SHAP feature (“Adjuster_Notes embedding”) so your plots are easier to read.
Grouping the 1536 embedding dimensions into a single SHAP feature (e.g. "Adjuster_Notes_Embedding"), 
makes the plots much more interpretable.  Instead of clutter with emb_0 to emb_1535, 
you’ll see just one feature summarizing the text embedding contribution.
"""

# --- Build feature names ---
cat_encoder = preprocessor.named_transformers_["categorical"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
feature_names_struct = num_cols + cat_feature_names

# Replace embedding dims with a single group name
# feature_names = feature_names_struct + ["Adjuster_Notes_Embedding"] * X_text.shape[1]

# --- SHAP Explainer ---
explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test)

# Convert sparse matrix to dense (if needed)
X_dense = X_test.toarray()

# --- GLOBAL FEATURE IMPORTANCE ---
# Group embeddings into one feature by name
"""
What changes here:
- We duplicate the label "Adjuster_Notes_Embedding" across all 1536 embedding dimensions.
- SHAP automatically aggregates identical feature names in plots → you see them as one block.

This gives clean plots like:
- Claim_Amount
- days_to_submit
- Customer_Age
- Adjuster_Notes_Embedding
- Premium_Amount

You can now interpret: “The embedding (text notes) strongly influences fraud detection” without
worrying about individual embedding dimensions.
"""

"""
let’s measure how much the embeddings contribute compared to structured features. 
We’ll use SHAP values to aggregate feature importance by group: 
1. Separate SHAP values for structured features vs embeddings. 
2. Compute the mean absolute SHAP value per group, this tells you which group contributes more on average. 
3. Express as percentages for easy reporting.
"""
# SHAP values (already computed earlier)
# shap_values = explainer.shap_values(X_test)

# Split feature indices
n_struct = len(feature_names_struct)  # numeric + categorical
n_total = X_dense.shape[1]
n_embed = n_total - n_struct

shap_values = np.array(shap_values)  # ensure numpy

# --- Group-level importance ---
abs_shap = np.abs(shap_values)

struct_importance = abs_shap[:, :n_struct].mean()
embed_importance = abs_shap[:, n_struct:].mean()

total = struct_importance + embed_importance

contrib_df = pd.DataFrame({
    "Feature_Group": ["Structured (numerical + categorical)", "Text Embedding"],
    "MeanAbsSHAP": [struct_importance, embed_importance],
    "Relative %": [100 * struct_importance / total, 100 * embed_importance / total]
    })

print(contrib_df)



"""
Example output (hypothetical):

          Feature_Group  MeanAbsSHAP  Relative %
0  Structured (num+cat)     0.0842        62.5
1        Text Embedding     0.0505        37.5

This means:
62% of predictive power comes from structured features (amounts, ages, categorical fields).
38% comes from embeddings (adjuster notes).

Why this is powerful:
- You can justify the inclusion of embeddings quantitatively: “Text contributed 37% of the fraud
signal.”
- If embeddings dominate, you might simplify structured features; if they’re weak, maybe embeddings
are not needed.
"""

"""
Simulated query-to-response flow (e.g., user asks “Why was claim 123 flagged?” → model generates 
a SHAP-based natural language summary)
Here’s a sample end-to-end flow showing how the application could handle the query.
"""
# --- Sample Query-to-Response Flow ---

# Step 1: User Query
# Why was claim 123 flagged?

# Step 2: Backend Processing
# Retrieve claim data for claim ID 123 from the database.
{
    "Claim_ID": 123,
    "Claim_Amount": 850000.0,
    "Premium_Amount": 95000.0,
    "Customer_Age": 24,
    "Policy_Type": "Individual",
    "Claim_Type": "Auto",
    "Incident_Type": "Accident",
    "Customer_Gender": "Male",
    "Customer_Occupation": "Unemployed",
    "Location": "Lagos",
    "Adjuster_Notes": "The claimant reported vague accident details and delayed submission."
}

# Run the model (XGB Classifier + embeddings).
# - Prediction: Fraudulent
# - Fraud Probability: 0.91

# Generate SHAP explanation (top contributing factors):
# - High claim amount (impact +0.37)
# - Low customer age (impact +0.21)
# - Unemployed occupation (impact +0.18)
# - Suspicious adjuster notes (“vague details”, impact +0.25)

# Step 3: Summarization Layer (LLM / Hugging Face)
''' Take the SHAP output + prediction and rephrase into a user-friendly explanation.'''

# Step 4: API Response
{
    "question": "Why was claim 123 flagged?",
    "answer": "Claim 123 was flagged as potentially fraudulent with a high probability (91%). The main reasons were the unusually large claim amount, the claimant’s young age, their unemployed status, and vague accident details reported in the adjuster notes.",
    "raw_explanation": {
        "prediction": "Fraudulent",
        "fraud_probability": 0.91,
        "top_factors": [
            "High claim amount (impact +0.37)",
            "Low customer age (impact +0.21)",
            "Unemployed occupation (impact +0.18)",
            "Suspicious wording in notes (impact +0.25)"
            ]
        }
    }

# Step 5: Analyst Sees
''' 
Claim 123 was flagged as potentially fraudulent with a 91% probability.
The unusually large claim amount, the claimant’s young age, unemployed status,
and vague accident details in the adjuster notes contributed most to the risk.
'''

# That’s the full query → model → SHAP → summary → response flow.

'''
Here's a multi-claim version of this flow (e.g., “Summarize fraud risks for last week’s claims”) 
so users can see how it scales across batches?

Let’s scale the flow up to multi-claim summaries. Instead of explaining a single claim, 
the system aggregates all claims in a time window and produces a human-readable fraud risk report.
'''

# --- Multi-Claim Query-to-Response Flow ---

# Step 1: User Query
# Summarize fraud risks for last week’s claims.

# Step 2: Backend Processing
# - Filter dataset for claims submitted last week.
# - Example (3 claims retrieved)

[
    {
        "Claim_ID": "A101",
        "Claim_Amount": 483077.79,
        "Premium_Amount": 82785.56,
        "Customer_Age": 28,
        "Occupation": "Artisan",
        "Fraud_Flag": 0,
        "Adjuster_Notes": "Normal household fire damage."
        },

    {
        "Claim_ID": "B202",
        "Claim_Amount": 313000.81,
        "Premium_Amount": 77527.65,
        "Customer_Age": 70,
        "Occupation": "Unemployed",
        "Fraud_Flag": 1,
        "Adjuster_Notes": "Unclear accident sequence, inconsistent statements."
        },

    {
        "Claim_ID": "C303",
        "Claim_Amount": 496678.56,
        "Premium_Amount": 5122.44,
        "Customer_Age": 52,
        "Occupation": "Student",
        "Fraud_Flag": 1,
        "Adjuster_Notes": "Unusually high medical costs compared to coverage."
        }
    ]

# Run model predictions (XGB + embeddings)
# - A101 → Legit (Fraud prob: 0.18)
# - B202 → Fraudulent (Fraud prob: 0.87)
# - C303 → Fraudulent (Fraud prob: 0.93)

# Generate SHAP explanations for each fraudulent case:
# - B202: Age 70 (+0.22), unemployed (+0.19), vague notes (+0.31)
# - C303: Very high claim vs premium (+0.42), occupation (student, +0.21), suspicious notes (+0.30)

# Step 3: Summarization Layer
# Combine explanations → generate a natural language report.

# Step 4: API Response
{
    "question": "Summarize fraud risks for last week’s claims.",
    "answer": "Out of 3 claims filed last week, 2 were flagged as potentially fraudulent. "
              "Claim B202 (87% fraud probability) was flagged due to vague accident descriptions,"
              "claimant’s advanced age, and unemployment status. "
              "Claim C303 (93% fraud probability) showed an unusually high claim amount compared to its low premium, "
              "a student claimant profile, and suspicious medical cost details."
              "Claim A101 was assessed as low risk (18% probability) with no major anomalies.",
    "summary_stats": {
    "total_claims": 3,
    "fraudulent": 2,
    "legit": 1,
    "avg_fraud_probability": 0.66
    }
}

# Step 5: Analyst Sees:
# - Last week, 3 claims were submitted.
# - 2 were flagged as fraudulent (B202, C303), mainly due to high claim-to-premium ratios, unusual occupations, and suspicious adjuster notes.
# - 1 claim (A101) appeared legitimate.

# This shows how the system scales from single-query explanations to batch risk monitoring.

'''
Extending this into a dashboard-style flow (e.g., analyst can query fraud summary by location, policy type, 
or time period and get a SHAP-driven narrative + chart)
let’s design a dashboard-style query-to-response flow for your fraud detection system. 
This combines natural language queries, XGB + SHAP explanations, and visual summaries,
so analysts get both narrative + charts.
'''

# --- Fraud Detection Dashboard Flow ---

# 1. User Query Examples
# Analyst types a natural query into the dashboard:
# - Show fraud summary for Lagos in the last 30 days.
# - Which policy types had the highest fraud risk last quarter?
# - Compare fraud risks across locations this week.

# 2. Backend Steps
# Filter Claims → based on query (location = Lagos, date range = last 30 days).
# Run Model (XGB Classifier + embeddings) → predict fraud probabilities.
# Generate SHAP explanations for top contributing features.
# Aggregate Results into summary stats & visualizations.

# 3. API Response (to feed dashboard)
{
    "question": "Show fraud summary for Lagos in the last 30 days.",
    "answer": "In Lagos, 12 claims were submitted in the last 30 days."
              "5 claims (42%) were flagged as potentially fraudulent."
              "The most common fraud indicators were: unusually high claim-to-premium ratios,"
              "vague adjuster notes, and younger claimants filing high-value claims."
              "Health and Auto policies contributed the most flagged cases.",
    "summary_stats": {
        "total_claims": 12,
        "fraudulent": 5,
        "legit": 7,
        "avg_fraud_probability": 0.58,
        "top_risk_factors": [
        "High claim amount vs premium",
        "Suspicious adjuster notes",
        "Young claimant with high-value claim"
            ]
        },
    "visuals": {
        "fraud_distribution_by_policy": {
        "Health": 3,
        "Auto": 2,
        "Life": 0,
        "Gadget": 0,
        "Fire": 0
            },
        "fraud_by_age_group": {
        "18-30": 3,
        "31-50": 1,
        "51+": 1
            }
        }
    }

# 4. Dashboard Display
# The dashboard presents:
# - 📊 Bar Chart → Fraud vs Legit claims by policy type.
# - 📈 Line Chart → Fraud probability trends over time.
# - 🧑‍💼 Demographic Breakdown → Age group / gender vs fraud likelihood.
# - 📝 Narrative Box → LLM summary (from JSON answer).

# 5. Analyst Sees:
# - 🔎 “In Lagos, 12 claims were filed last month.
# - 42% were flagged as potentially fraudulent.
# - Most fraud was linked to Health and Auto policies.
# - High claim-to-premium ratios and vague adjuster notes were the leading red flags.”

# This gives analysts both quantitative insights and an AI-driven narrative in one view.

