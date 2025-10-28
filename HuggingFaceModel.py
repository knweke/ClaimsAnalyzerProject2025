import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
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
import shap
import joblib
import os
import time
import pickle
from typing import List
import openai
from tqdm import tqdm
from joblib import dump, load
import tensorflow as tf

"""
How It Works:
1. Text embeddings → sentence-transformers/all-MiniLM-L6-v2 turns adjuster notes into 384
dim semantic vectors.
2. Structured features → numeric values scaled, categorical features one-hot encoded.
3. Feature fusion → concatenate structured + text embedding vectors.
4. XGBoost → learns fraud patterns using both text + structured signals.
5. Evaluation → precision, recall, F1 to measure fraud detection quality.
"""

# Load dataset
filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.csv'
df = pd.read_csv(filepath, sep=",")  # or pd.read_excel / pd.read_csv depending on file
print(df.head())

print("Shape:", df.shape)
#print(df.columns.tolist())

# Target variable
target = df["Fraud_Flag"]

# Choose structured and text columns
# text_col = df["Adjuster_Notes"]
structured_cols = ["Claim_Amount", "Premium_Amount", "Customer_Age",
                   "Policy_Type", "Claim_Type", "Incident_Type", 'Claim_Status',
                   "Customer_Gender", "Customer_Occupation", "Location"]

# Hugging Face embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Batch encode Adjuster_Notes
notes = df["Adjuster_Notes"].fillna("").tolist()
embeddings = embedder.encode(notes, show_progress_bar=True)

# Convert embeddings into a DataFrame
emb_df = pd.DataFrame(embeddings, index=df.index)
emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

# Concatenate with structured features
df_final = pd.concat([df[structured_cols], emb_df], axis=1)

# Structured features
X_structured = df[structured_cols]
y = target

# Preprocess structured features (scaling + one-hot encoding)
numeric_features = ["Claim_Amount", "Premium_Amount", "Customer_Age"]

categorical_features = ["Policy_Type", "Claim_Type", "Incident_Type", "Claim_Status",
                        "Customer_Gender", "Customer_Occupation", "Location"]

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough'
)

X_structured_transformed = preprocessor.fit_transform(X_structured)

# Combine structured + text embeddings
from scipy.sparse import hstack
X_final = hstack([X_structured_transformed, embeddings])  # embeddings are dense

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost classifier
clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))

# --- CROSS VALIDATION ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
print('Cross validation score:', cv_score)
print('Mean accuracy score:', cv_score.mean())

joblib.dump(clf, 'xgb_fraud_model.pkl')
joblib.dump(clf, "preprocessor.pkl")

"""
Next Steps:
- Add explainability with SHAP → show which features/text embeddings drove the fraud prediction. 
- Use larger models (all-mpnet-base-v2) if you want better embeddings. 
- Deploy as a real-time.

We’ll use SHAP (SHapley Additive exPlanations) for feature importance and then convert those signals 
into a natural language explanation.

Extended Pipeline with Explainability.
"""

# Initialize SHAP explainer for XGBoost
explainer = shap.TreeExplainer(clf)

# Get SHAP values for test set
shap_values = explainer.shap_values(X_test)

# Convert SHAP values into feature importance summary
def explain_prediction(index, top_k=5):
    """
    Generate a human-readable explanation for a single claim prediction
    """
    # Raw prediction
    pred = clf.predict(X_test[index])
    proba = clf.predict_proba(X_test[index])[0][1]

    # Get SHAP values for this instance
    shap_vals = shap_values[index]
    feature_names = preprocessor.get_feature_names_out().tolist() + [f"embed_{i}" for i in range(embeddings.shape[1])]

    # Pair feature names with SHAP importance
    importance = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    # Natural language explanation
    reasons = []
    for feat, val in importance:
        if "Claim_Amount" in feat:
            reasons.append(f"high claim amount (impact: {val:.2f})")
        elif "Premium_Amount" in feat:
            reasons.append(f"premium size (impact: {val:.2f})")
        elif "Customer_Age" in feat:
            reasons.append(f"customer age effect (impact: {val:.2f})")
        elif "Adjuster note" in feat or "embed" in feat:
            reasons.append("suspicious wording in adjuster notes")
        elif "Occupation" in feat:
            reasons.append(f"customer occupation risk (impact: {val:.2f})")
        else:
            reasons.append(f"{feat} (impact: {val:.2f})")

    # Prediction summary
    explanation = (
        f"Claim #{index} was predicted as {'Fraudulent' if pred == 1 else 'Legit'}"
        f"(fraud probability: {proba:.2f}).\n"
        f"Top factors: {', '.join(reasons)}."
    )

    return explanation

# Explain a few predictions
for i in range(3):
    print(explain_prediction(i))
    print("-" * 80)

"""
Example Output (natural language):
Claim #0 was predicted as Fraudulent (fraud probability: 0.87).
Top factors: high claim amount (impact: 0.45), suspicious wording in adjuster notes,
customer occupation risk (impact: 0.31), premium size (impact: 0.21).
 
Claim #1 was predicted as Legit (fraud probability: 0.12). 
Top factors: suspicious wording.

What This Adds:
- Prediction + Explanation → not just “fraud = 1”, but “why the model thinks so.” 
- Human-readable reasons → analysts can interpret results (e.g., high claim amount, suspicious 
adjuster notes). 
- Trust + Auditability → useful for regulators, compliance, and operational teams.
"""

# FAST API SERVICE
"""
How to serve this as an API (FastAPI/Flask) so users can ask real time questions like,
“Why was claim X flagged?” and get this summary back instantly? 

Let’s make this pipeline real-time with a lightweight FastAPI service. 
That way, analysts or claim officers can query: 
- “Predict fraud for claim #12345” 
- “Why was this claim flagged?” 
and get both the prediction + natural language explanation instantly.
"""

from fastapi import FastAPI
from pydantic import BaseModel

# ------------------------------
# Load models & preprocessors
# ------------------------------
# Load pretrained components (trained earlier)
clf = joblib.load("xgb_fraud_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
explainer = shap.TreeExplainer(clf)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Fraudulent Claims Detector API")

# Input schema
class ClaimRequest(BaseModel):
    Claim_Amount: float
    Premium_Amount: float
    Customer_Age: int
    Policy_Type: str
    Claim_Status: str
    Claim_Type: str
    Incident_Type: str
    Customer_Gender: str
    Customer_Occupation: str
    Location: str
    Adjuster_Notes: str

# ------------------------------
# Helper: explanation generator
# ------------------------------
def explain_prediction(X_row, embedding):
    # Combine structured + embedding
    X_final = np.hstack([X_row.toarray(), embedding.reshape(1, -1)])

    # Prediction
    proba = clf.predict_proba(X_final)[0][1]
    pred = int(proba > 0.5)

    # SHAP values
    shap_vals = explainer.shap_values(X_final)[0]

    # Feature names
    feature_names = preprocessor.get_feature_names_out().tolist() + [f"embed_{i}" for i in range(embedding.shape[0])]

    # Top features
    importance = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # Human-readable
    reasons = []
    for feat, val in importance:
        if "Claim_Amount" in feat:
            reasons.append(f"high claim amount (impact {val:.2f})")
        elif "Premium_Amount" in feat:
            reasons.append(f"premium size (impact {val:.2f})")
        elif "Customer_Age" in feat:
            reasons.append(f"customer age (impact {val:.2f})")
        elif "embed" in feat:
            reasons.append("suspicious wording in adjuster notes")
        else:
            reasons.append(f"{feat} (impact {val:.2f})")

    explanation = {
        "prediction": "Potentially Fraudulent" if pred == 1 else "Legitimate claim",
        "fraud_probability": round(proba, 3),
        "top_factors": reasons
    }
    return explanation

# ------------------------------
# API route
# ------------------------------
@app.post("/predict")
def predict_claim(claim: ClaimRequest):
    # Convert to DataFrame
    claim_df = pd.DataFrame({claim.dict()})

    # Preprocess structured features
    structured_cols = ["Claim_Amount", "Premium_Amount", "Customer_Age",
                       "Policy_Type", "Claim_Type", "Incident_Type", 'Claim_Status',
                       "Customer_Gender", "Customer_Occupation", "Location"]

    X_structured = preprocessor.transform(claim_df[structured_cols])

    # Generate embedding for notes
    embedding = embedder.encode(claim_df["Adjuster_Notes"].iloc[0])

    # Get prediction + explanation
    result = explain_prediction(X_structured, embedding)
    return result

# Save Models After Training In your training notebook/script (from earlier steps), add:
import joblib
joblib.dump(clf, "xgb_fraud_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

'''# Step 4: Run the API
# uvicorn app:app --reload --port 8000

# Run the API
#uvicorn app:app --host 0.0.0.0 --port 8000

# Example Request(with curl or Postman)
curl -X POST "http://127.0.0.1:8000/predict"
-H "Content-Type: application/json"
-d '{
"Claim_ID": "123",
"Claim_Amount": 300000,
"Premium_Amount": 20000,
"Customer_Age": 28,
"Policy_Type": "Health",
"Claim_Type": "Accident",
"Incident_Type": "Fire",
"Customer_Occupation": "Artisan",
"Adjuster_Notes": "High-value claim with vague incident description"
}'

# Example Response
{
  "claim_id": "123",
  "fraud_probability": 0.81,
  "fraud_label": 1,
  "explanation": "Claim 123 was flagged as fraudulent with probability 0.81. Top contributing factors: Claim_Amount contributed 0.34, Premium_Amount contributed -0.12, Customer_Age contributed 0.08.",
  "shap_values": {
    "Claim_Amount": 0.34,
    "Premium_Amount": -0.12,
    "Customer_Age": 0.08
    }
    }'''

from transformers import pipeline

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
@app.post("/query")
def query_claim(claim: ClaimRequest, question: str = "Why was this claim flagged?"):
    """
    Natural language query endpoint.
    Returns a summarized answer based on SHAP + claim data.
    """
    # Convert input to DataFrame
    claim_df = pd.DataFrame([claim.dict()])

    # Preprocess structured features
    structured_cols = ["Claim_Amount","Premium_Amount","Customer_Age",
                       "Policy_Type","Claim_Type","Incident_Type",
                       "Customer_Gender","Customer_Occupation","Location"]

    X_structured = preprocessor.transform(claim_df[structured_cols])

    # Embedding for notes
    embedding = embedder.encode(claim_df["Adjuster_Notes"].iloc[0])

    # Get structured explanation
    result = explain_prediction(X_structured, embedding)

    # Create context for summarizer
    context = (
    f"Prediction: {result['prediction']}, " 
    f"Fraud probability: {result['fraud_probability']}. " 
    f"Top contributing factors: {', '.join(result['top_factors'])}. " 
    f"Question: {question}"
    )

    # Summarize in natural language
    summary = summarizer(context, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
    return {
        "question": question,
        "answer": summary,
        "raw_explanation": result
    }

'''
#Step 3: Example Request
curl -X POST "http://127.0.0.1:8000/query?question=Why%20was%20this%20claim%20flagged%3F" \
-H "Content-Type: application/json" \
-d '{
  "Claim_Amount": 483077.79,
"Premium_Amount": 82785.56,
    "Customer_Age": 28,
    "Policy_Type": "Family",
    "Claim_Type": "Health",
    "Incident_Type": "Fire",
    "Customer_Gender": "Female",
    "Customer_Occupation": "Artisan",
    "Location": "Ibadan",
    "Adjuster_Notes": "Local tend employee source nature add rest human station property ability management test."
}'

# Example Response
{
"question": "Why was this claim flagged?",
"answer": "The claim was assessed as legitimate with low fraud probability. Key drivers included a high claim amount, premium size, and specific customer profile, though wording in the adjuster notes had minor suspicious signals.",
"raw_explanation": {
    "prediction": "Legit",
    "fraud_probability": 0.23,
    "top_factors": [
    "high claim amount (impact 0.42)",
    "premium size (impact 0.21)",
    "customer age (impact -0.15)",
    "suspicious wording in adjuster notes",
    "Policy_Type_Family (impact 0.12)"
        ]
    }
}'''

"""
What You Get:
/predict → structured fraud probability + top features. 
/query → natural language answers to analyst-style questions. 
SHAP stays the source of truth, LLM just makes it human-friendly.
"""

"""
Do you also want me to show you how to extend /query so it works with multi-claim queries (e.g., 
“Summarize top fraud reasons across last 50 claims”)? That would let analysts ask portfolio-level questions, 
not just single-claim ones. 

Great 🚀 — let’s extend your Fraud Claims API so it can handle multi-claim queries like: 
- “Summarize top fraud reasons across the last 50 claims” 
- “What are the common patterns in flagged claims this week?” 
- “Which features most contributed to fraudulent predictions overall?”

We’ll: 
1. Collect SHAP explanations across multiple claims. 
2. Aggregate them (e.g., most frequent or highest average impact). 
3. Use an LLM summarizer to generate a concise natural-language answer.
"""

'''# Step 1: Extend API with Portfolio Query
@app.post("/portfolio_query")
def portfolio_query(claims: list[ClaimRequest], question: str = "Summarize top fraud reasons"):
    """
    # Accepts multiple claims, runs predictions + SHAP, and summarizes the overall fraud patterns.
    """
    explanations = []

    for claim in claims:
        claim_df = pd.DataFrame([claim.dict()])

        # Structured preprocessing
        structured_cols = ["Claim_Amount", "Premium_Amount", "Customer_Age",
                           "Policy_Type", "Claim_Type", "Incident_Type", 'Claim_Status',
                           "Customer_Gender", "Customer_Occupation", "Location"]

        X_structured = preprocessor.transform(claim_df[structured_cols])

        # Embedding for notes
        embedding = embedder.encode(claim_df["Adjuster_Notes"].iloc[0])

        # Explanation
        result = explain_prediction(X_structured, embedding)
        explanations.append(result)

    # Aggregate results
    preds = [e["prediction"] for e in explanations]
    probs = [e["fraud_probability"] for e in explanations]
    factors = [", ".join(e["top_factors"]) for e in explanations]

    context = (
        f"Portfolio Query: {question}. "
        f"Out of {len(explanations)} claims, {preds.count('Fraudulent')} were flagged as fraudulent "
        f"and {preds.count('Legit')} were legitimate. "
        f"Average fraud probability: {np.mean(probs):.2f}. "
        f"Common contributing factors include: {'; '.join(factors)}."
    )

    # Summarize with LLM
    summary = summarizer(context, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
    return {
        "question": question,
        "answer": summary,
        "portfolio_stats": {
            "total_claims": len(explanations),
            "fraudulent": preds.count("Fraudulent"),
            "legit": preds.count("Legit"),
            "avg_fraud_probability": np.mean(probs)
            },
        "sample_explanations": explanations[:3]  # return first 3 detailed as sample
    }

# Step 2: Example Request
curl - X POST "http://127.0.0.1:8000/portfolio_query?question=Summarize%20top%20fraud%20reasons%20across%20claims" \
-H "Content-Type: application/json" \
-d '[
  {
    "Claim_Amount": 483077.79,
    "Premium_Amount": 82785.56,
    "Customer_Age": 28,
    "Policy_Type": "Family",
    "Claim_Type": "Health",
    "Incident_Type": "Fire",
    "Customer_Gender": "Female",
    "Customer_Occupation": "Artisan",
    "Location": "Ibadan",
    "Adjuster_Notes": "Local tend employee source nature add rest human station property ability management test."
},
{
    "Claim_Amount": 313000.81,
    "Premium_Amount": 77527.65,
    "Customer_Age": 70,
    "Policy_Type": "Family",
    "Claim_Type": "Auto",
    "Incident_Type": "Accident",
    "Customer_Gender": "Male",
    "Customer_Occupation": "Unemployed",
    "Location": "Abuja",
    "Adjuster_Notes": "International artist situation talk despite stage plant view own available."
}
]'

# Example Response
{
    "question": "Summarize top fraud reasons across claims"
    "answer": "Out of 2 claims reviewed, 1 was flagged as fraudulent and 1 legitimate. Average fraud probability was moderate. Key drivers included high claim amounts, suspicious wording in adjuster notes, and risk factors such as customer occupation and unusual incident types."
    "portfolio_stats": {
        "total_claims": 2,
        "fraudulent": 1,
        "legit": 1,
        "avg_fraud_probability": 0.55
},

"sample_explanations": [
    {
        "prediction": "Legit",
        "fraud_probability": 0.23,
        "top_factors": ["high claim amount (impact 0.42)", "premium size (impact 0.21)", "..."]
    },
    {
        "prediction": "Fraudulent",
        "fraud_probability": 0.87,
        "top_factors": ["suspicious wording in adjuster notes", "customer occupation risk", "..."]
        }
    ]
}'''

"""
# What This Adds:
- Analysts can now ask portfolio-level questions (multiple claims).
- Output gives summary + stats + sample explanations.
- Useful for weekly fraud reports or batch reviews.
"""
