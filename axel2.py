import os
import time
from PIL import Image
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import plotly.express as px
from transformers import pipeline
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

# Load environment variables from .env file
# load_dotenv()

# --- LOAD CLAIMS DATA ---
filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.xlsx'
df = pd.read_excel(filepath) # pd.read_excel (.xlsx)

df['Customer_Phone'] = (
    df['Customer_Phone']
    .astype(str)
    .str.strip()           # remove whitespace
    .str.replace(r'\.0$', '', regex=True)  # remove unwanted decimals if any
)

# --- SAVED MODELS ---
# joblib.dump(clf, 'xgb_fraud_model.pkl')
# joblib.dump(clf, "preprocessor.pkl")

# ---------------------------
# CONFIG / CACHING HELPERS
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers alias
SUMMARIZER_MODEL = "facebook/bart-large-cnn"  # optional, heavy
CACHE_FILE = "embeddings_cache.pkl"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource
def load_embedder(model_name=EMBED_MODEL_NAME):
    return SentenceTransformer(f"sentence-transformers/{model_name}")

@st.cache_resource
def load_summarizer(model_name2=SUMMARIZER_MODEL):
    try:
        return pipeline("summarization", model=model_name2)
    except Exception as e:
        st.warning("Could not load summarizer model locally (it may be large). Summaries will be disabled.")
    return None

#@st.cache_resource
#def make_shap_explainer(_model):
    #return shap.TreeExplainer(_model)

# ---------------------------
# DATA PREPROCESSING
# ---------------------------
def parse_dates(df: pd.DataFrame):
    date_cols = ["Incident_Date", "Claim_Submission_Date", "Policy_Start_Date", "Policy_End_Date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Engineered features
    if {"Incident_Date", "Claim_Submission_Date"}.issubset(df.columns):
        df["days_to_submit"] = (df["Claim_Submission_Date"] - df["Incident_Date"]).dt.days.fillna(0)
    else:
        df["days_to_submit"] = 0
    if {"Policy_Start_Date", "Policy_End_Date"}.issubset(df.columns):
        df["policy_duration_days"] = (df["Policy_End_Date"] - df["Policy_Start_Date"]).dt.days.fillna(0)
    else:
        df["policy_duration_days"] = 0
    return df

# ---------------------------
# EMBEDDINGS (CACHED)
# ---------------------------
EMBED_CACHE = os.path.join(MODEL_DIR, "embed_cache.pkl")

def get_embeddings(texts: List[str], embedder: SentenceTransformer):
    # simple file cache to avoid re-embedding same text repeatedly
    try:
        cache = joblib.load(EMBED_CACHE)
    except Exception:
        cache = {}
    embeddings = []
    to_compute = []
    idx_map = []
    for i, t in enumerate(texts):
        key = t if isinstance(t, str) else ""
        if key in cache:
            embeddings.append(cache[key])
        else:
            embeddings.append(None)
            to_compute.append(key)
            idx_map.append(i)

    # Compute in bulk if needed
    if to_compute:
        new_embs = embedder.encode(to_compute, show_progress_bar=False, convert_to_numpy=True)
        for j, emb in enumerate(new_embs):
            key = to_compute[j]
            cache[key] = emb
            embeddings[idx_map[j]] = emb
        joblib.dump(cache, EMBED_CACHE)

    # Convert to 2D numpy array
    return np.vstack(embeddings)

# ---------------------------
# FEATURE PIPELINE
# ---------------------------
def build_preprocessor(df: pd.DataFrame, numeric_cols, cat_cols):
    num_transform = StandardScaler()
    cat_transform = OneHotEncoder(handle_unknown="ignore")
    preproc = ColumnTransformer([
        ("num", num_transform, numeric_cols),
        ("cat", cat_transform, cat_cols)
    ], sparse_threshold=0.0)  # Allow dense result if small, we'll keep sparse handling
    preproc.fit(df[numeric_cols + cat_cols])
    return preproc

# ---------------------------
# TRAIN / LOAD MODEL
# ---------------------------
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_clf.joblib")
PREPROC_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")

def train_and_save_model(X_struct, embeddings, y):
    """
    X_struct: sparse matrix from preprocessor.transform
    embeddings: dense numpy array
    y: target array
    """
    # combine
    if isinstance(X_struct, csr_matrix):
        X_final = hstack([X_struct, csr_matrix(embeddings)])
    else:
        # X_struct dense
        X_final = np.hstack([X_struct, embeddings])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2,
                                                        stratify=y if len(np.unique(y)) > 1 else None,
                                                        random_state=42)
    st.info("Training XGBoost Model. It may take a few minutes for large data.")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    xgb_clf.fit(X_train, y_train)

    # --- EVALUATE ---
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    print(precision_score(y_test, y_pred))
    print(recall_score(y_test, y_pred))

    joblib.dump(xgb_clf, MODEL_FILE)
    st.success("Model trained and saved.")
    return xgb_clf

def load_model_if_exists():
    if os.path.exists(MODEL_FILE) and os.path.exists(PREPROC_FILE):
        try:
            xgb_clf = joblib.load(MODEL_FILE)
            preproc = joblib.load(PREPROC_FILE)
            return xgb_clf, preproc
        except Exception:
            return None, None
    return None, None

# ---------------------------
# SHAP EXPLAINERS
# ---------------------------
@st.cache_resource
def make_shap_explainer(_xgb_clf):
    explainer = shap.TreeExplainer(xgb_clf)
    return explainer

def explain_instance(xgb_clf, explainer, X_struct_row, emb_row, preproc, top_k=6):
    X_struct_row = X_struct_row.reshape(1, -1)  # (1, n_features)
    emb_row = emb_row.reshape(1, -1)  # (1, embedding_dim)

    X_combined = np.hstack([X_struct_row, emb_row])  # (1, n_features + embedding_dim)

    proba = float(xgb_clf.predict_proba(X_combined)[0, 1])
    pred = int(proba > 0.5)
    shap_vals = explainer.shap_values(X_combined)[0]  # shape (n_features)

    # Build feature names
    num_names = preproc.transformers_[0][2]

    # Get categorical feature names
    cat_encoder = preproc.transformers_[1][1]
    try:
        cat_names = cat_encoder.get_feature_names_out(preproc.transformers_[1][2])
        cat_names = list(cat_names)
    except Exception:
        cat_names = []
    struct_names = list(num_names) + cat_names
    embed_dim = emb_row.shape[0]
    embed_names = [f"embed_{i}" for i in range(embed_dim)]
    all_feature_names = struct_names + embed_names

    # Pair and sort
    pairs = list(zip(all_feature_names, shap_vals))
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
    formatted = [{"**Top Selected Features**": p[0], "**SHAP Values**": float(p[1])} for p in pairs_sorted]

    return {
        "prediction": "**Potentially fraudulent claim**" if pred == 1 else "**Legitimate claim**",
        "fraud_probability": round(proba, 4),
        "top_shap": formatted,
        "all_feature_names": all_feature_names,
        "shap_values": shap_vals
    }

# ---------------------------
# AGGREGATION / PORTFOLIO SUMMARY
# ---------------------------
def portfolio_aggregate(explanations: List[dict]):
    # Explanations is list of explain_instance outputs
    total = len(explanations)
    fraud_count = sum(1 for e in explanations if e["prediction"] == "**Potentially fraudulent claim**")
    avg_prob = np.mean([e["fraud_probability"] for e in explanations]) if total > 0 else 0.0

    # Aggregate top features frequency and average impact
    flat = {}
    for e in explanations:
        for feat in e["top_shap"]:
            name = feat["**Top Selected Features**"]  # ✅ use correct key
            value = feat["**SHAP Values**"]  # ✅ use correct key
            flat.setdefault(name, []).append(abs(value))

    agg = [
        {"feature": k, "count": len(v), "mean_abs_shap_value": float(np.mean(v))}
        for k, v in flat.items()]
    agg_df = pd.DataFrame(agg).sort_values(["mean_abs_shap_value", "count"], ascending=False)
    return {
        "total_claims": total,
        "fraudulent_claims": fraud_count,
        "legit_claims": total - fraud_count,
        "avg_fraud_probability": float(avg_prob),
        "top_features": agg_df.head(20)
        }

# --- NEWLY ADDED ---
# ----------------------------
# Portfolio Summary Function
# ----------------------------
def run_portfolio_summary(df, summarizer=None):
    stats = {
        "num_claims": len(df),
        "avg_claim_amount": df["Claim_Amount"].mean(),
        "fraud_rate": df["Fraud_Flag"].mean(),
    }

    base_summary = (st.info(
        f"Portfolio have **{stats['num_claims']}** claim requests. "
        f"Average claim amount: **₦{stats['avg_claim_amount']:.2f}**. "
        f"Estimated fraud rate: **{stats['fraud_rate']:.1%}**."
        )
    )

    if summarizer:
        try:
            response = summarizer(
                base_summary,
                max_length=60,
                min_length=20,
                do_sample=False
            )
            return response[0]["summary_text"]
        except Exception as e:
            return f"⚠️ Summarizer failed: {e}\n\nFallback summary: {base_summary}"
    else:
        return base_summary

# ---------------------------
# STREAMLIT UI - MAIN APP
# ---------------------------
# Add widgets
image = Image.open('C:/Users/Springrose/Downloads/FRAUD DETECTION/axa-logo.png')
st.image(image, width=80)

st.markdown('<p style="font-family: calibri; color:#000080; font-size: 38px;">AXA Mansard Insurance Plc</p>', unsafe_allow_html=True)
# st.markdown('## AXA Mansard Insurance Plc')
st.markdown("##### Life and Non-Life Insurance Claims Analyzer and Anomaly Detector")

image2 = Image.open('C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\insight-logo3.png')
st.image(image2, width=150)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Claims Portfolio", "🛰️ OSINT Risk", "🔍 Customer Search", "📥 Export Reports"])

# Step 1: Add PDF Export Dependencies
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from io import BytesIO
import datetime
import os

LOGO_PATH = "C:/Users/Springrose/Downloads/FRAUD DETECTION/axa-logo.png"

def generate_pdf_report(df, portfolio_summary_text=None, charts=[]):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18,
                                 textColor=colors.HexColor("#000080"))
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"],
                                   fontSize=14, textColor=colors.HexColor("#C00000"))
    normal_style = styles["Normal"]
    elements = []

    # Logo
    if os.path.exists(LOGO_PATH):
        elements.append(RLImage(LOGO_PATH, width=100, height=40))
        elements.append(Spacer(1, 20))

    # Title
    elements.append(Paragraph("AXA Mansard Insurance Plc", title_style))
    elements.append(Paragraph("Fraud & OSINT Risk Analysis Report", heading_style))
    elements.append(Spacer(1, 20))

    # Date
    today = datetime.datetime.now().strftime("%d %B %Y")
    elements.append(Paragraph(f"Report Date: {today}", normal_style))
    elements.append(Spacer(1, 20))

    # Portfolio Summary
    if portfolio_summary_text:
        elements.append(Paragraph("Portfolio Summary", heading_style))
        elements.append(Paragraph(portfolio_summary_text, normal_style))
        elements.append(Spacer(1, 20))

    # Risk Table
    if not df.empty:
        elements.append(Paragraph("Top Risk Customers", heading_style))
        table_data = [df.columns.tolist()] + df.head(20).values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#000080")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    # Charts
    for chart in charts:
        chart_path = f"chart_{hash(chart)}.png"
        chart.write_image(chart_path)
        elements.append(RLImage(chart_path, width=400, height=250))
        elements.append(Spacer(1, 20))

    # Footer
    elements.append(Spacer(1, 40))
    footer = Paragraph(
        "Confidential - AXA Mansard Insurance Plc © All Rights Reserved",
        ParagraphStyle("FooterStyle", parent=normal_style, fontSize=8, textColor=colors.grey, alignment=1)
    )
    elements.append(footer)

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# Step 2: Sidebar Export Tab
with tab4:
    st.markdown("###### 📥 Export Risk Reports")

    if "risk_df" in st.session_state and not st.session_state["risk_df"].empty:
        export_df = st.session_state["risk_df"]

        if st.button("⬇️ Generate PDF Summary Report"):
            portfolio_summary_text = st.session_state.get("portfolio_summary", "")
            charts = st.session_state.get("charts", [])

            pdf_bytes = generate_pdf_report(export_df, portfolio_summary_text, charts)
            st.download_button(
                label="📄 Download PDF Summary Report",
                data=pdf_bytes,
                file_name="claims_analysis_osint_report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("⚠️ No risk summary available yet. Run Portfolio Summary first.")

#st.write("**Report special features:**")
#st.write("- Header with AXA logo & blue/red corporate colors")
#st.write("- Condensed portfolio summary")
#st.write("- Styled risk table with AXA blue header row")
#st.write("- Embedded charts from active Streamlit session")
#st.write("- Rich combined report summary")
#st.write("- Export CSV, Excel and PDF reports")

# st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")
# st.title("Fraud Detection Dashboard — Streamlit")

with tab1:
    st.markdown("###### Claim Analysis, Model Interpretation and SHAP Explanation")

with st.sidebar:
    st.header("**Control Panel**")
    uploaded = st.file_uploader("**Upload Claims Data (Tabular)**", type=["csv", "txt", "xlsx"],
                                help="Claims data with columns like Claim_Amount, Adjuster_Notes, Fraud_Flag")

    use_pretrained = st.checkbox("Load saved preprocessor and model if available", value=True)
    enable_summarizer = st.checkbox("Enable natural language summaries (may load large model)", value=False)
    run_training = st.button("**Train model from uploaded data**")
    st.markdown("---")
    st.markdown("**Notes:**\n- Embeddings cached to reduce repeated calls.\n- Summarizer maybe large and slow to load.")
    st.markdown("---")

# load summarizer optionally
#summarizer = load_summarizer(SUMMARIZER_MODEL) if enable_summarizer else None
summarizer = load_summarizer() if enable_summarizer else None

# --- NEWLY ADDED ---
if "summarizer" not in st.session_state and load_summarizer:
    from transformers import pipeline
    st.session_state.summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
summarizer = st.session_state.get("summarizer", None)

# If user uploaded file
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".txt"):
            # Try comma first, fallback to tab
            try:
                df = pd.read_csv(uploaded, sep=",")
            except Exception:
                df = pd.read_csv(uploaded, sep="\t")
        elif uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload a CSV, TXT, or XLSX file.")
            df = None

        if df is not None:
            st.info("###### 🔍 **Claims Data Preview**")
            st.write(df.head())
            # Parse dates & basic cleaning
            df = parse_dates(df)

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        df = None

    # Ensure text column exist
    text_col = "Adjuster_Notes" if "Adjuster_Notes" in df.columns else None
    if text_col is None:
        st.error("Claims data must include 'Adjuster_Notes' column for text embeddings.")
        st.stop()

    # Default numeric & categorical columns we will use
    numeric_cols = []
    for c in ["Claim_Amount", "Customer_Age", "Premium_Amount", "days_to_submit", "policy_duration_days"]:
        if c in df.columns:
            numeric_cols.append(c)

    cat_cols = []
    for c in ["Policy_Type", "Claim_Type", "Incident_Type", "Claim_Status",
              "Customer_Gender", "Customer_Occupation", "Location"]:
        if c in df.columns:
            cat_cols.append(c)

    target_col = "Fraud_Flag" if "Fraud_Flag" in df.columns else None
    if target_col is None:
        st.warning("No 'Fraud_Flag' column found — the app will only predict using a model if one already exist or if you train one.")

    # Load embedder
    embedder = load_embedder()

    # Compute embeddings (with cache)
    with st.spinner("Computing / loading embeddings (cached)..."):
        texts = df[text_col].fillna("").astype(str).tolist()
        embeddings = get_embeddings(texts, embedder)
    st.success(f"Computed embeddings: shape {embeddings.shape}")

    # Build or load preprocessor
    if use_pretrained:
        xgb_clf, preproc = load_model_if_exists()
        if xgb_clf is not None:
            st.success("Loaded saved model and preprocessor.")
        else:
            preproc = build_preprocessor(df, numeric_cols, cat_cols)
            joblib.dump(preproc, PREPROC_FILE)
            st.info("Built and saved new preprocessor.")
            xgb_clf = None
    else:
        preproc = build_preprocessor(df, numeric_cols, cat_cols)
        joblib.dump(preproc, PREPROC_FILE)
        xgb_clf = None

    # If user clicked train or no model loaded and there is a target, train
    if run_training:
        if target_col is None:
            st.error("Cannot train: target column 'Fraud_Flag' missing.")
        else:
            X_struct = preproc.transform(df[numeric_cols + cat_cols])
            xgb_clf = train_and_save_model(X_struct, embeddings, df[target_col].astype(int).values)
            joblib.dump(preproc, PREPROC_FILE)

    # If model still None but file exists (maybe user didn't toggle)
    if xgb_clf is None:
        xgb_clf, preproc_loaded = load_model_if_exists()
        if xgb_clf is None:
            st.info("No model available yet. Train a model or enable pre-trained.")
    else:
        # Ensure preprocessor saved
        joblib.dump(preproc, PREPROC_FILE)

        # If we have a model, let user interact
        if xgb_clf is not None:
            explainer = make_shap_explainer(xgb_clf)

        # If you have a main dataframe "df" (the full claims dataset):
        if "Fraud" in df.columns:
            y = df["Fraud"]
            X = df.drop(columns=["Fraud"])
        else:
            X = df.copy()
            y = None  # optional if label not available

        # Use your preprocessor to transform X for model input
        X_prepared = preproc.transform(X)
        X_test = pd.DataFrame(X_prepared, columns=preproc.get_feature_names_out())

        st.info("###### 🔍 Single Claim Inspection and Analysis")
        claim_index = st.number_input("Row index to inspect (0 based)", min_value=0,
                                      max_value=len(df) - 1, value=0, step=1)

        # Trigger explanation when button is clicked
        if st.button("Inspect selected claim"):
            try:
                # Filter and display the selected row
                selected_row = df.loc[[claim_index]]

                # st.divider()
                st.markdown("###### 📄 Selected Claim Details")
                st.dataframe(selected_row)
            except Exception as e:
                st.error(f"An error occurred while explaining the claim: {e}")

                # Preprocess and explain the selected row
                X_struct_row = preproc.transform(df.loc[[claim_index], numeric_cols + cat_cols])
                emb_row = embeddings[claim_index]

                # Display explanation results
                explanation = explain_instance(xgb_clf, explainer, X_struct_row, emb_row, preproc, top_k=8)

                st.info("###### 📊 SHAP Data Driven Insights")
                st.write("Prediction result:", explanation["prediction"])
                st.write("Fraud probability:", f"{explanation['fraud_probability']:.2f}")
                st.write("**Top SHAP Feature Contributions:**")
                shap_df = pd.DataFrame(explanation["top_shap"])
                st.table(shap_df)

                # Natural language explanation
                insights = []
                for _, row in shap_df.iterrows():
                    feature, value = row["**Top Selected Features**"], row["**SHAP Values**"]
                    if value > 0:
                        insights.append(
                            f"- Positive SHAP values increase fraud score, boost model predictive power, and fraudulent claim detection.")
                    else:
                        insights.append(
                            f"- Negative SHAP values decrease fraud score, lower model predictive power, and sees claim as legitimate.")
                st.markdown("**SHAP Interpretation:**")
                st.write("\n".join(insights))
                st.write(explain_prediction_by_id(claim_index))

                st.divider()

                # Natural language summary (optional)
                if summarizer is not None:
                    # Build context string
                    top_factors = ", ".join(
                        [f"{f['**Top Selected Features**']} (impact: {f['**SHAP Values**']:.3f})" for f in explanation["top_shap"]])
                    context = (
                        f"Prediction: {explanation['prediction']}. "
                        f"Fraud probability: {explanation['fraud_probability']}. "
                        f"Top factors: {top_factors}."
                    )
                    with st.spinner("Generating natural language summary..."):
                        try:
                            summ = summarizer(
                                context,
                                max_length=80,
                                min_length=20,
                                do_sample=False
                            )[0]["summary_text"]
                            st.markdown("**Natural language summary:**")
                            st.write(summ)

                        except Exception as e:
                            st.warning("Summarizer failed: " + str(e))

                            st.info("###### 📄 Claim Batch Analysis")
                            # Selection controls
                            import uuid

                            # --- Helper to generate unique keys (avoids duplicate element IDs) ---
                            def unique_key(base: str) -> str:
                                return f"{base}_{uuid.uuid4().hex[:6]}"

                            mask = pd.Series(True, index=df.index)  # start with all True

                            with st.form(unique_key("claims_filters_form")):
                                st.write("ℹ️ Adjust filters, click **Apply Filters** to view result")

                                # --- 1️⃣ Date range filter ---
                                if "Claim_Submission_Date" in df.columns:
                                    # Convert to datetime safely
                                    df["Claim_Submission_Date"] = pd.to_datetime(df["Claim_Submission_Date"],
                                                                                 errors="coerce")

                                    min_date = df["Claim_Submission_Date"].min()
                                    max_date = df["Claim_Submission_Date"].max()

                                    sub_date_range = st.date_input(
                                        "**Submission date range:**",
                                        value=(
                                            min_date.date() if pd.notna(min_date) else None,
                                            max_date.date() if pd.notna(max_date) else None,
                                        ),
                                        key=unique_key("submission_date_filter")
                                    )

                                # --- 2️⃣ Location filter ---
                                if "Location" in df.columns:
                                    locs = ["All"] + sorted(df["Location"].dropna().unique().tolist())
                                    loc_choice = st.selectbox(
                                        "**Location filter:**",
                                        locs,
                                        index=0,
                                        key=unique_key("location_filter")
                                    )

                                # --- 3️⃣ Policy Type filter ---
                                if "Policy_Type" in df.columns:
                                    ptypes = ["All"] + sorted(df["Policy_Type"].dropna().unique().tolist())
                                    p_choice = st.selectbox(
                                        "**Policy type filter:**",
                                        ptypes,
                                        index=0,
                                        key=unique_key("policy_type_filter")
                                    )

                                    # --- 4️⃣ Apply button ---
                                    apply_filters = st.form_submit_button("**Apply Filters**")

                                    # --- Apply filters only after button click ---
                                    if apply_filters:
                                        if "Claim_Submission_Date" in df.columns and isinstance(sub_date_range, tuple):
                                            if sub_date_range[0] is not None:
                                                mask &= (df["Claim_Submission_Date"].dt.date >= sub_date_range[0])
                                            if len(sub_date_range) > 1 and sub_date_range[1] is not None:
                                                mask &= (df["Claim_Submission_Date"].dt.date <= sub_date_range[1])

                                    if "Location" in df.columns and loc_choice != "All":
                                        mask &= (df["Location"] == loc_choice)

                                    if "Policy_Type" in df.columns and p_choice != "All":
                                        mask &= (df["Policy_Type"] == p_choice)

                                    # Apply mask
                                    filtered_df = df[mask]

                                    st.success(f"✅ {filtered_df.shape[0]} claims matched filters.")
                                    st.dataframe(filtered_df.head())
                                    # else:
                                    # st.write(".")

                                    # Subset data
                                    df_sub = df[mask].reset_index(drop=True)
                                    st.info(f"**{len(df_sub)}** claim requests selected for indepth analysis.")

                                    st.divider()

                                    if "Run portfolio summary on filtered claims data":
                                        if df_sub.empty:
                                            st.warning("No claims selected.")
                                        else:
                                            if "Claim_Amount" not in df_sub.columns or "Fraud_Flag" not in df_sub.columns:
                                                st.error(
                                                    "Dataset must contain 'Claim_Amount' and 'Fraud_Flag' columns.")
                                            else:
                                                with st.spinner("Generating portfolio summary..."):
                                                    summary_text = run_portfolio_summary(df_sub, summarizer)
                                                st.markdown("###### 📄 Portfolio Summary")
                                                st.write(summary_text)

                                        # Explain each claim (This could be expensive — warning!)
                                        explanations = []
                                        X_struct_all = preproc.transform(df_sub[numeric_cols + cat_cols])

                                        # Compute embeddings subset from cached embeddings - already have them in same order
                                        # Because df_sub is a filtered view indexing matches original positions
                                        emb_sub = embeddings[df_sub.index]

                                        with st.spinner(
                                                "Computing explanations for each claim. This may take some time."):
                                            for i in range(len(df_sub)):
                                                expl = explain_instance(xgb_clf, explainer, X_struct_all[i], emb_sub[i],
                                                                        preproc,
                                                                        top_k=6)
                                                explanations.append(expl)

                                        agg = portfolio_aggregate(explanations)
                                        st.write("**Portfolio stats:**", {
                                            "total_claims": agg["total_claims"],
                                            "fraudulent_claims": agg["fraudulent_claims"],
                                            "legit_claims": agg["legit_claims"],
                                            "average_fraud_probability": agg["average_fraud_probability"]
                                        })
                                        st.info("###### 📄 Top contributing features (aggregated)")
                                        st.dataframe(agg["top_features"].head(20))

                                        # Fraud probability distribution
                                        probs = [e["fraud_probability"] for e in explanations]
                                        probs_df = pd.DataFrame({"fraud_probability": probs})
                                        fig = px.histogram(probs_df, x="fraud_probability", nbins=20,
                                                           title="Fraud probability distribution")
                                        st.plotly_chart(fig, use_container_width=True, key='bzx')

                                        # Counts by policy type
                                        if "Policy_Type" in df_sub.columns:
                                            # Compute predicted label counts
                                            labels = [e["prediction"] for e in explanations]
                                            df_sub2 = df_sub.copy()
                                            df_sub2["pred_label"] = labels
                                            fig2 = px.histogram(df_sub2, x="Policy_Type", color="pred_label",
                                                                barmode="group",
                                                                title="Predicted labels by policy type")
                                            st.plotly_chart(fig2, use_container_width=True, key='ytf')

                                        # Counts by location
                                        if "Location" in df_sub.columns:
                                            # Compute predicted label counts
                                            labels = [e["prediction"] for e in explanations]
                                            df_sub3 = df_sub.copy()
                                            df_sub3["pred_label"] = labels
                                            fig3 = px.histogram(df_sub3, x="Location", color="pred_label",
                                                                barmode="group", title="Predicted labels by location")
                                            st.plotly_chart(fig3, use_container_width=True, key='ied')

                                        # Counts by claim type
                                        if "Claim_Type" in df_sub.columns:
                                            # Compute predicted label counts
                                            labels = [e["prediction"] for e in explanations]
                                            df_sub4 = df_sub.copy()
                                            df_sub4["pred_label"] = labels
                                            fig4 = px.histogram(df_sub4, x="Claim_Type", color="pred_label",
                                                                barmode="group", title="Predicted labels by claim type")
                                            st.plotly_chart(fig4, use_container_width=True, key='qsd')

                                        # Counts by incident type
                                        if "Incident_Type" in df_sub.columns:
                                            # Compute predicted label counts
                                            labels = [e["prediction"] for e in explanations]
                                            df_sub5 = df_sub.copy()
                                            df_sub5["pred_label"] = labels
                                            fig5 = px.histogram(df_sub5, x="Incident_Type", color="pred_label",
                                                                barmode="group",
                                                                title="Predicted labels by incident type")
                                            st.plotly_chart(fig5, use_container_width=True, key='krd')

                                        # Counts by customer occupation
                                        if "Customer_Occupation" in df_sub.columns:
                                            # Compute predicted label counts
                                            labels = [e["prediction"] for e in explanations]
                                            df_sub6 = df_sub.copy()
                                            df_sub6["pred_label"] = labels
                                            fig6 = px.histogram(df_sub6, x="Customer_Occupation", color="pred_label",
                                                                barmode="group",
                                                                title="Predicted labels by customer occupation")
                                            st.plotly_chart(fig6, use_container_width=True, key='abh')

                                        # Natural language summary
                                        if summarizer is not None:
                                            context = ((f"Portfolio summary: {agg['total_claims']} claims,\n "
                                                       f"{agg['fraudulent']} flagged as Fraudulent, average fraud probability,\n "
                                                       f"{agg['avg_fraud_probability']: .2f}.Top features: ")
                                                       + ", \n".join(agg["top_features"]["feature"].head(10).tolist()))

                                        with st.spinner("Generating portfolio natural language summary..."):
                                            try:
                                                summ = summarizer(context, max_length=200, min_length=40,
                                                                  do_sample=False)[0]["summary_text"]
                                                st.markdown("##### 📄 Natural language portfolio summary")
                                                st.write(summ)
                                            except Exception as e:
                                                st.warning("Summarizer failed: " + str(e))
else:
    st.info("Upload claims data in the sidebar to get started.")
# st.info("Upload claims data in the sidebar to get started.")