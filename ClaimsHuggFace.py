import os
import time
from PIL import Image
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import plotly.express as px
from transformers import pipeline
from scipy.sparse import hstack, csr_matrix
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load environment variables from .env file
# load_dotenv()

# --- LOAD CLAIMS DATA ---
filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD_DETECTION\\SmartClaimsData.xlsx'
df = pd.read_excel(filepath)  # pd.read_excel (.xlsx)

df['Customer_Phone'] = (
    df['Customer_Phone']
    .astype(str)
    .str.strip()  # remove whitespace
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


'''@st.cache_resource
def load_summarizer(model_name2=SUMMARIZER_MODEL):
    try:
        return pipeline("summarization", model=model_name2)
    except Exception as e:
        st.warning("Could not load summarizer model locally (it may be large). Summaries will be disabled.")
    return None'''


@st.cache_resource
def load_summarizer():
    # explicitly load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARIZER_MODEL,
        torch_dtype=torch.float32,  # ensure CPU-friendly
        device_map="auto"  # auto device handling
    )

    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        #device=-1  # use CPU only; prevents GPU/meta tensor mismatch
    )
    return summarizer


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
    # Combine single row
    #if isinstance(X_struct_row, csr_matrix):
    #X_combined = np.hstack([X_struct_row, csr_matrix(emb_row.reshape(1, -1))])
    #else:
    #X_combined = np.hstack([X_struct_row, emb_row.reshape(1, -1)])

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
        "average_fraud_probability": float(avg_prob),
        "top_features": agg_df.head(20)
    }


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
image = Image.open('C:/Users/Springrose/Downloads/FRAUD_DETECTION/axa-logo.png')
st.image(image, width=80)

st.markdown('<p style="font-family: calibri; color:#000080; font-size: 38px;">AXA Mansard Insurance Plc</p>',
            unsafe_allow_html=True)
# st.markdown('## AXA Mansard Insurance Plc')
st.markdown("##### Life and Non-Life Insurance Claims Analyzer and Anomaly Detector")

image2 = Image.open('C:\\Users\\Springrose\\Downloads\\FRAUD_DETECTION\\insight-logo3.png')
st.image(image2, width=150)


def generate_content(tab_name):
    return f"{tab_name}"


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**📄 Single Claim Analysis**", "**📄 Portfolio-Level Insights**",
                                                    "**📄 Claim Batch Analysis**", "**📊 Claim Portfolio Breakdown**",
                                                    "**🛰️ OSINT Risk**", "**🕵️ Customer Search**",
                                                    "**📥 Export Reports**"])

#with st.sidebar:
#tab2, tab3, tab4 = st.tabs(["Portfolio-Level Insights", "Claim Batch Analysis", "📊 Portfolio Summary"])
#with st.sidebar:
#tab5, tab6, tab7 = st.tabs(["🛰️ OSINT Risk", "🔍 Customer Search", "📥 Export Reports"])

#with tab7:
#st.write(generate_content("📥 Export Reports"))

# Step 1: Add PDF Export Dependencies
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from io import BytesIO
import datetime
import os

LOGO_PATH = "C:/Users/Springrose/Downloads/FRAUD_DETECTION/axa-logo.png"


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
    st.info("⚠️ No risk summary available yet. Run **Portfolio Summary** first.")

#st.write("**Report special features:**")
#st.write("- Header with AXA logo & blue/red corporate colors")
#st.write("- Condensed portfolio summary")
#st.write("- Styled risk table with AXA blue header row")
#st.write("- Embedded charts from active Streamlit session")
#st.write("- Rich combined report summary")
#st.write("- Export CSV, Excel and PDF reports")

# st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")
# st.title("Fraud Detection Dashboard — Streamlit")

# st.divider()

with st.sidebar:
    st.info("**⚙️ Control Panel**")
    uploaded = st.file_uploader("**Upload Claims Data (Tabular)**", type=["csv", "txt", "xlsx"],
                                help="Claims data with columns like Claim_Amount, Adjuster_Notes, Fraud_Flag")

    use_pretrained = st.checkbox("Load saved preprocessor and model if available", value=True)
    enable_summarizer = st.checkbox("Enable natural language summaries (may load large model)", value=False)
    run_training = st.button("**Train model from uploaded data**")
    st.markdown("---")
    # st.markdown("**Notes:**\n- Embeddings cached to reduce repeated calls.\n-
    # Summarizer maybe large and slow to load.")

# load summarizer optionally
# summarizer = load_summarizer() if enable_summarizer else None
st.session_state.summarizer = load_summarizer()

# --- NEWLY ADDED ---
if "summarizer" not in st.session_state and load_summarizer:
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
        st.warning(
            "No 'Fraud_Flag' column found — the app will only predict using a model if one already exist or if you train one.")

    # Load embedder
    embedder = load_embedder()

    # Compute embeddings (with cache)
    with st.spinner("Computing / loading embeddings (cached)..."):
        texts = df[text_col].fillna("").astype(str).tolist()
        embeddings = get_embeddings(texts, embedder)
    st.success(f"Computed embeddings: Shape {embeddings.shape}")

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

        # st.divider()

        st.info("###### 🔍 Single Claim Inspection and Analysis")
        claim_index = st.number_input("Row index to inspect (0 based)", min_value=0,
                                      max_value=len(df) - 1, value=0, step=1)

        # Trigger explanation when button is clicked
        if st.button("Inspect and analyze selected claim"):
            # Filter and display the selected row
            selected_row = df.loc[[claim_index]]
            st.markdown("###### 📄 Selected Claim Details")
            st.dataframe(selected_row)
            #except Exception as e:
            #st.error(f"An error occurred while explaining the claim: {e}")

            # Preprocess and explain the selected row
            X_struct_row = preproc.transform(df.loc[[claim_index], numeric_cols + cat_cols])
            emb_row = embeddings[claim_index]

            # Display explanation results
            explanation = explain_instance(xgb_clf, explainer, X_struct_row, emb_row, preproc, top_k=8)

            #with tab7:
            #st.write(generate_content("📥 Export Reports"))
            st.info("###### 📊 SHAP Data Driven Insights")
            st.write("Prediction result:", explanation["prediction"])
            st.write("Fraud probability:", f"{explanation['fraud_probability']:.2f}")
            st.info("**Top SHAP Feature Contributions:**")
            shap_df = pd.DataFrame(explanation["top_shap"])
            st.table(shap_df)

            # Natural language explanation (IMPROVED)
            st.info("**SHAP Interpretation:**")
            interpretation_lines = []
            for _, row in shap_df.iterrows():
                feature, value = row["**Top Selected Features**"], row["**SHAP Values**"]
                impact = "increased" if value > 0 else "decreased"
                direction = "a risk factor" if value > 0 else "a mitigating factor"
                #customer = "delayed" if value >= 365 else "submitted"
                #duration = df.loc[[selected_row, "policy_duration_days"]]
                #days = df.loc[[selected_row, 'days_to_submit']]

                # Try to get the feature value for context (works well for one-hot and numerical, less for embeddings)
                try:
                    feature_name_base = feature.split('_')[0] if "cat__" in feature else feature
                    feature_value = selected_row.iloc[0][feature_name_base]

                    if "Adjuster_Notes" in feature:
                        value_text = "from the text analysis"
                    elif "cat__" in feature:
                        value_text = f"because the claim has feature: *{feature.split('__')[-1]}*"
                    else:
                        value_text = f"valued at **₦{feature_value}**"
                except:
                    value_text = ""

                line = (f"**{feature}** {value_text} ({direction}) with a SHAP value of "
                        f"**{value:.3f}**, significantly {impact} fraud probability.")
                interpretation_lines.append(line)
                st.write("\n\n".join(interpretation_lines))

            st.divider()

            st.info("###### 📄 Claim Batch Analysis")
            # Selection controls
            import uuid


            # --- Helper to generate unique keys (avoids duplicate element IDs) ---
            def unique_key(base: str) -> str:
                return f"{base}_{uuid.uuid4().hex[:6]}"


            # mask = pd.Series(True, index=df.index)  # start with all True
            mask = pd.Series([True] * len(df), index=df.index)

            with st.form(unique_key("claims_filters_form")):
                st.write("ℹ️ Adjust date filters, click **Apply Filters** to view result")

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

                mask = pd.Series([True] * len(df), index=df.index)

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
                else:
                    df_sub = df[mask].reset_index(drop=True)
                    st.info(f"**{len(df_sub)}** claim records detected.")

            if "Run portfolio summary on filtered claims data":
                if df_sub.empty:
                    st.warning("No claims selected.")
                else:
                    if "Claim_Amount" not in df_sub.columns or "Fraud_Flag" not in df_sub.columns:
                        st.error("Dataset must contain 'Claim_Amount' and 'Fraud_Flag' columns.")
                    else:
                        with st.spinner("**Generating portfolio summary...**"):
                            summary_text = run_portfolio_summary(df_sub, summarizer)
                        st.info("###### 📊 Claims Data Analytics")
                        #st.write(summary_text)

                        # Explain each claim (This could be expensive — warning!)
                        explanations = []
                        X_struct_all = preproc.transform(df_sub[numeric_cols + cat_cols])

                        # Compute embeddings subset from cached embeddings - already have them in same order
                        # Because df_sub is a filtered view indexing matches original positions
                        emb_sub = embeddings[df_sub.index]

                        with st.spinner("**Computing explanations for each claim. This may take some time.**"):
                            for i in range(len(df_sub)):
                                expl = explain_instance(xgb_clf, explainer, X_struct_all[i],
                                                        emb_sub[i], preproc, top_k=6)
                                explanations.append(expl)

                        agg = portfolio_aggregate(explanations)
                        st.write("**Portfolio Stats:**", {
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
                                                barmode="group", title="Predicted labels by policy type")
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
                                                barmode="group", title="Predicted labels by incident type")
                            st.plotly_chart(fig5, use_container_width=True, key='krd')

                        # Counts by customer occupation
                        if "Customer_Occupation" in df_sub.columns:
                            # Compute predicted label counts
                            labels = [e["prediction"] for e in explanations]
                            df_sub6 = df_sub.copy()
                            df_sub6["pred_label"] = labels
                            fig6 = px.histogram(df_sub6, x="Customer_Occupation", color="pred_label",
                                                barmode="group", title="Predicted labels by customer occupation")
                            st.plotly_chart(fig6, use_container_width=True, key='abh')

                            # Natural language portfolio summary
                            if summarizer is not None:
                                context = (f"{agg['**total_claims**']} claim requests were observed and "
                                           f"analysed by AI model. "
                                           f"**{agg['fraudulent_claims']}** of those were flagged "
                                           f"as potentially fraudulent. "
                                           f"Model predicted **{agg['average_fraud_probability']:.1%}**"
                                           f" as average fraud probability. "
                                           f"Top fraud detection features listed were " + ", ".join(
                                    agg["top_features"]["feature"].head(10).tolist()))
                                with st.spinner("**Generating portfolio summary...**"):
                                    try:
                                        summ = summarizer(
                                            context,
                                            max_length=150,
                                            min_length=40,
                                            do_sample=False
                                        )[0]["summary_text"]
                                        st.info("###### 📄 Claim Portfolio Summary")
                                        st.write(summ)
                                    except Exception as e:
                                        st.warning("Summarizer failed: " + str(e))
                        else:
                            st.warning("No successful explanations were generated.")

else:
    st.info("Upload claims data in the sidebar to get started. "
            "Claims data should contain at least 'Adjuster_Notes' and ideally numeric/categorical columns "
            "such as Claim_Amount, Premium_Amount, Customer_Age, Policy_Type, Claim_Type, Incident_Type, "
            "Location and Fraud_Flag.")

st.divider()

with tab4:
    st.write(generate_content(""))
    st.info("###### 📄 High Value Claim Detection")
    threshold = df['Claim_Amount'].quantile(0.80)
    high_claims = df[df['Claim_Amount'] > threshold]
    st.write(f"- Claim above 80th percentile: **₦{threshold:.2f}**")
    st.dataframe(high_claims)

    threshold = df['Claim_Amount'].quantile(0.80)
    high_claims = df[df['Claim_Amount'] > threshold]
    st.write(f"- Number of suspicious high-amount claims: **{len(high_claims)}**")

    st.info("###### 📄 Frequent Claimants")
    frequent_claims = df['Customer_Name'].value_counts()
    frequent_customers = frequent_claims[frequent_claims > 3]
    st.write(f"- Customers with greater than 3 claims: **{len(frequent_customers)}**")
    st.dataframe(frequent_customers)

    st.info("###### 📄 Simulated Claim Savings")
    df['capped_claim'] = np.where(df['Claim_Amount'] > 400000, 400000, df['Claim_Amount'])
    savings = df['Claim_Amount'].sum() - df['capped_claim'].sum()
    st.write(f"- Total Savings with Cap: **₦{savings:,.2f}**")

    st.info("###### 📄 Claim Cost Center Analysis")
    st.write(df['Policy_Type'].value_counts())
    # print('\n')
    corporate_claim_amount = df[df['Policy_Type'].isin(['Corporate'])]['Claim_Amount'].sum()
    st.write(f'- Corporate claims total amount: **₦{corporate_claim_amount:,.2f}**')
    family_claim_amount = df[df['Policy_Type'].isin(['Family'])]['Claim_Amount'].sum()
    st.write(f'- Family claims total amount: **₦{family_claim_amount:,.2f}**')
    individual_claim_amount = df[df['Policy_Type'].isin(['Individual'])]['Claim_Amount'].sum()
    st.write(f'- Individual claims total amount: **₦{individual_claim_amount:,.2f}**')

    st.info("###### 📄 Claim Type")
    st.write(df['Claim_Type'].value_counts())
    # print('\n')
    gadget_claim_amount = df[df['Claim_Type'].isin(['Gadget'])]['Claim_Amount'].sum()
    st.write(f'- Gadget claims total amount: **₦{gadget_claim_amount:,.2f}**')
    auto_claim_amount = df[df['Claim_Type'].isin(['Auto'])]['Claim_Amount'].sum()
    st.write(f'- Auto claims total amount: **₦{auto_claim_amount:,.2f}**')
    Fire_claim_amount = df[df['Claim_Type'].isin(['Fire'])]['Claim_Amount'].sum()
    st.write(f'- Fire claims total amount: **₦{Fire_claim_amount:,.2f}**')
    life_claim_amount = df[df['Claim_Type'].isin(['Life'])]['Claim_Amount'].sum()
    st.write(f'- Life claims total amount: **₦{life_claim_amount:,.2f}**')
    health_claim_amount = df[df['Claim_Type'].isin(['Health'])]['Claim_Amount'].sum()
    st.write(f'- Health claims total amount: **₦{health_claim_amount:,.2f}**')

    st.info("###### 📄 Incident Type")
    st.write(df['Incident_Type'].value_counts())
    # print('\n')
    death_claim_amount = df[df['Incident_Type'].isin(['Death'])]['Claim_Amount'].sum()
    st.write(f'- Death claims total amount: **₦{death_claim_amount:,.2f}**')
    theft_claim_amount = df[df['Incident_Type'].isin(['Theft'])]['Claim_Amount'].sum()
    st.write(f'- Theft claims total amount: **₦{theft_claim_amount:,.2f}**')
    fire_claim_amount = df[df['Incident_Type'].isin(['Fire'])]['Claim_Amount'].sum()
    st.write(f'- Fire claims total amount: **₦{fire_claim_amount:,.2f}**')
    accident_claim_amount = df[df['Incident_Type'].isin(['Accident'])]['Claim_Amount'].sum()
    st.write(f'- Accident claims total amount: **₦{accident_claim_amount:,.2f}**')
    illness_claim_amount = df[df['Incident_Type'].isin(['Illness'])]['Claim_Amount'].sum()
    st.write(f'- Illness claims total amount: **₦{illness_claim_amount:,.2f}**')

    st.info("###### 📄 Claim Status")
    st.write(df['Claim_Status'].value_counts())
    # print('\n')
    approved_claims_total_amount_paid = df[df['Claim_Status'].isin(['Approved'])]['Claim_Amount'].sum()
    st.write(f'- Approved claims total amount paid: **₦{approved_claims_total_amount_paid:,.2f}**')
    ave_appr_claims_total_amount_paid = df[df['Claim_Status'].isin(['Approved'])]['Claim_Amount'].sum() / 589
    st.write(f'- Average approved claims total amount paid: **₦{ave_appr_claims_total_amount_paid:,.2f}**')

    st.info("###### 📄 Claimants by Location")
    # Customer count by Location
    st.write(df['Location'].value_counts())

    st.info("###### 📄 Claim Amount by Location")
    # Claim amount by Location
    claim_amount_abuja = df[df['Location'].isin(['Abuja'])]['Claim_Amount'].sum()
    st.write(f'- Abuja total claim amount: **₦{claim_amount_abuja:,.2f}**')
    claim_amount_ibadan = df[df['Location'].isin(['Ibadan'])]['Claim_Amount'].sum()
    st.write(f'- Ibadan total claim amount: **₦{claim_amount_ibadan:,.2f}**')
    claim_amount_kano = df[df['Location'].isin(['Kano'])]['Claim_Amount'].sum()
    st.write(f'- Kano total claim amount: **₦{claim_amount_kano:,.2f}**')
    claim_amount_lagos = df[df['Location'].isin(['Lagos'])]['Claim_Amount'].sum()
    st.write(f'- Lagos total claim amount: **₦{claim_amount_lagos:,.2f}**')
    claim_amount_ph = df[df['Location'].isin(['Port Harcourt'])]['Claim_Amount'].sum()
    st.write(f'- Port Harcourt total claim amount: **₦{claim_amount_ph:,.2f}**')

    st.info("###### 📄 Claim Status Count")
    # Claim status count
    st.write(df['Claim_Status'].value_counts())
    # print('\n')

    st.info("###### 📄 Customer Gender")
    # Customer gender count
    st.write(df['Customer_Gender'].value_counts())
    # print('\n')

    st.info("###### 📄 Customer Occupation")
    # Customer occupation count
    st.write(df['Customer_Occupation'].value_counts())

    st.info("###### 📄 Minimum, Maximum and Average Claim Amount")
    # Minimum and maximum claim amount
    st.write(f"- Minimum claim amount: **₦{df['Claim_Amount'].min():,.2f}**")
    st.write(f"- Maximum claim amount: **₦{df['Claim_Amount'].max():,.2f}**")
    st.write(f"- Average claim amount: **₦{df['Claim_Amount'].mean():,.2f}**")

st.info(
    f"**Disclaimer:** Prediction is made by Machine Learning model and "
    f"might require further validation to ensure its accuracy.")

with tab6:
    st.write(generate_content(""))
    import os
    import pandas as pd
    import requests
    from difflib import get_close_matches
    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt
    import asyncio
    from typing import List, Dict, Any, Optional
    from dataclasses import dataclass
    from rapidfuzz import process, fuzz
    import pandas as pd
    # Tweepy for Twitter API v2 (X)
    import tweepy
    # Transformers for lightweight classification
    from transformers import pipeline
    # load env vars optionally
    from dotenv import load_dotenv

    load_dotenv()

    # ---------- CONFIG ----------
    TWITTER_BEARER = os.getenv("TWITTER_BEARER")  # set in .env or env
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # or another appropriate model
    WATCHLIST_CSV = "C:/Users/Springrose/Downloads/FRAUD_DETECTION/GlobalTerrorismDataset.csv"  # CSV with column 'name' (and optional aliases)
    GLOBAL_TERROR_TXT = "C:/Users/Springrose/Downloads/FRAUD_DETECTION/globalterrorism.txt"

    # ---------- DATACLASS FOR RESULTS ----------
    @dataclass
    class SocialProfileCheckResult:
        profiles_found: List[Dict[str, Any]]
        posts_sample: List[Dict[str, Any]]
        engagement: Dict[str, Any]
        watchlist_matches: List[Dict[str, Any]]
        extremist_flag: bool
        criminal_flag: bool
        notes: List[str]


    # ---------- HELPERS ----------
    def init_twitter_client(bearer_token: str) -> Optional[tweepy.Client]:
        if not bearer_token:
            return None
        try:
            client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
            return client
        except Exception:
            return None


    # Initialize classifier (zero-shot)
    def init_text_classifier(model_name: str = ZERO_SHOT_MODEL):
        try:
            classifier = pipeline("zero-shot-classification", model=model_name)
            return classifier
        except Exception as e:
            # In production log error and fall back to simpler keyword checks
            print("Failed to load HF model:", e)
            return None


    # --------- WATCHLIST LOADING & FUZZY MATCH ----------
    def load_watchlist(csv_path: str) -> pd.DataFrame:
        """
        Load watchlist CSV into DataFrame with normalized 'name' column.
        """
        try:
            df = pd.read_csv(csv_path, encoding='latin1')
        except Exception:
            # try other delimiters
            df = pd.read_csv(WATCHLIST_CSV, sep=None, engine="python", encoding='latin1')
        if "name" not in df.columns:
            raise ValueError("Watchlist CSV must contain a 'name' column.")
            # normalize
        df["name_norm"] = df["name"].astype(str).str.strip().str.lower()
        return df


    def fuzzy_watchlist_match(name: str, watchlist_df: pd.DataFrame, top_k: int = 10,
                              score_cutoff: int = 85):
        """
        Returns a list of watchlist entries that fuzzily match the input name.
        """
        if not name or watchlist_df.empty:
            return []
        choices = watchlist_df["name_norm"].tolist()
        matches = process.extract(name.lower().strip(), choices,
                                  scorer=fuzz.token_sort_ratio, limit=top_k)
        # matches: list of tuples (match, score, index)
        results = []
        for matched_name, score, idx in matches:
            if score >= score_cutoff:
                row = watchlist_df.iloc[idx].to_dict()
                results.append({
                    "watchlist_name": row.get("name"),
                    "normalized": matched_name,
                    "score": score,
                    "watchlist_row": row
                })
        return results


    # ---------- TWITTER / X FETCHING (PUBLIC) ----------
    def fetch_twitter_user_and_posts(client: tweepy.Client, username: Optional[str] = None,
                                     query_name: Optional[str] = None, max_posts: int = 10):
        """
        If username supplied, try to fetch that user's public tweets.
        Otherwise perform a recent search for the name.
        Returns dict: {'profile': {...} or None, 'posts': [...]}
        IMPORTANT: only public posts are fetched. Private accounts are not accessible.
        """
        if client is None:
            return {"profile": None, "posts": []}

            # If username provided
        try:
            if username:
                user = client.get_user(username=username, user_fields=["public_metrics", "created_at", "verified"])
                if user and user.data:
                    uid = user.data.id
                    profile = {
                        "username": user.data.username,
                        "name": user.data.name,
                        "id": uid,
                        "verified": user.data.verified,
                        "followers": user.data.public_metrics.get("followers_count"),
                        "following": user.data.public_metrics.get("following_count"),
                        "tweet_count": user.data.public_metrics.get("tweet_count")
                    }
                    tweets = []
                    # fetch recent tweets
                    resp = client.get_users_tweets(uid, max_results=min(100, max_posts),
                                                   tweet_fields=["created_at", "public_metrics", "text"])
                    if resp and resp.data:
                        for t in resp.data:
                            tweets.append({
                                "id": t.id,
                                "text": t.text,
                                "created_at": t.created_at.isoformat() if t.created_at else
                                None,
                                "retweets": t.public_metrics.get("retweet_count"),
                                "replies": t.public_metrics.get("reply_count"),
                                "likes": t.public_metrics.get("like_count"),
                                "quotes": t.public_metrics.get("quote_count")
                            })
                    return {"profile": profile, "posts": tweets}
                    # Otherwise search recent tweets mentioning the name
            if query_name:
                query = f'"{query_name}" -is:retweet lang:en'
                tweets = []
                resp = client.search_recent_tweets(query=query, max_results=min(100, max_posts),
                                                   tweet_fields=["created_at", "public_metrics", "text", "author_id"])
                if resp and resp.data:
                    # fetch author details separately if needed
                    for t in resp.data:
                        tweets.append({
                            "id": t.id,
                            "author_id": t.author_id,
                            "text": t.text,
                            "created_at": t.created_at.isoformat() if t.created_at else None,
                            "retweets": t.public_metrics.get("retweet_count"),
                            "replies": t.public_metrics.get("reply_count"),
                            "likes": t.public_metrics.get("like_count"),
                            "quotes": t.public_metrics.get("quote_count")
                        })
                return {"profile": None, "posts": tweets}
        except Exception as e:
            # gracefully return no results and log
            print("Twitter fetch failed:", e)
            return {"profile": None, "posts": []}
        return {"profile": None, "posts": []}


    # ---------- TEXT ANALYSIS ----------
    def analyze_posts_for_flags(posts: List[Dict[str, Any]], classifier, labels=None, threshold: float = 0.6):
        """
        Run zero-shot or keyword-based classification to flag extremist/criminal mentions.
        Returns: {'extremist': bool, 'criminal_activity': bool, 'detailed': [...]}
        """
        if not posts:
            return {"extremist": False, "criminal": False, "detailed": []}

        if labels is None:
            labels = ["terrorism", "extremist ideology", "criminal activity", "violence",
                      "none_of_the_above"]
        details = []
        extremist_flag = False
        criminal_flag = True

        for p in posts:
            text = p.get("text", "")
            item = {"id": p.get("id"), "text": text, "scores": {}, "flags": []}

            # If classifier available use it
            if classifier:
                try:
                    out = classifier(text, candidate_labels=labels)
                    # out: {'sequence':..., 'labels':[...], 'scores':[...]}
                    for lab, score in zip(out["labels"], out["scores"]):
                        item["scores"][lab] = float(score)
                    # decide flags
                    if item["scores"].get("terrorism", 0) >= threshold or item["scores"].get("extremist ideology",
                                                                                             0) >= threshold:
                        extremist_flag = True
                        item["flags"].append("extremist_indicator")
                    if item["scores"].get("criminal activity", 0) >= threshold:
                        criminal_flag = True
                        item["flags"].append("criminal_indicator")
                except Exception as e:

                    # fallback: simple keyword check
                    text_lower = text.lower()
                    if any(k in text_lower for k in ["bomb", "attack", "terror", "isis", "alqaeda"]):
                        extremist_flag = True
                        item["flags"].append("extremist_keyword")
                    if any(k in text_lower for k in ["steal", "fraud", "murder", "kill", "assault"]):
                        criminal_flag = True
                        item["flags"].append("criminal_keyword")
            else:
                # no classifier - use simple keyword heuristics
                text_lower = text.lower()
                if any(k in text_lower for k in ["bomb", "attack", "terror", "isis", "al qaeda"]):
                    extremist_flag = True
                    item["flags"].append("extremist_keyword")
                if any(k in text_lower for k in ["steal", "fraud", "murder", "kill",
                                                 "assault"]):
                    criminal_flag = True
                    item["flags"].append("criminal_keyword")
            details.append(item)
        return {"extremist": extremist_flag, "criminal": criminal_flag, "detailed": details}


    # ---------- MAIN CHECK FUNCTION ----------
    def check_customer_socials(customer_name: str,
                               customer_identifiers: Dict[str, str] = None,
                               twitter_bearer: Optional[str] = None,
                               watchlist_csv: Optional[str] = WATCHLIST_CSV,
                               max_posts: int = 50) -> Dict[str, Any]:
        """
        High-level function to check public social media, engagements, watchlists, and return
        a structured dict.
        Always returns a result dict (even if empty or no permissions).
        NOTE: This function searches public posts only. It does NOT access private accounts
        or private records.
        """
        result = {
            "customer_name": customer_name,
            "profiles": [],
            "posts_sample": [],
            "engagement": {},
            "watchlist_matches": [],
            "extremist_affiliation": False,
            "criminal_records_flag": True,
            "notes": []
        }

        # 1) Initialize services
        twitter_client = init_twitter_client(twitter_bearer)
        classifier = init_text_classifier()

        # 2) Load watchlist
        watchlist_df = None
        if WATCHLIST_CSV and os.path.exists(WATCHLIST_CSV):
            try:
                watchlist_df = load_watchlist(WATCHLIST_CSV)
            except Exception as e:
                result["notes"].append(f"Watchlist load failed: {e}")

        # 3) Fuzzy match name against watchlist
        if watchlist_df is not None:
            matches = fuzzy_watchlist_match(customer_name, watchlist_df, top_k=5,
                                            score_cutoff=85)
            result["watchlist_matches"] = matches
            if matches:
                result["notes"].append("Potential watchlist name matches found (use official verification).")

        # 4) Try to fetch twitter by username if identifier provided, else search name
        posts_all = []
        if customer_identifiers and customer_identifiers.get("twitter"):
            info = fetch_twitter_user_and_posts(twitter_client,
                                                username=customer_identifiers.get("twitter"), max_posts=max_posts)
            if info["profile"]:
                result["profiles"].append(info["profile"])
            posts_all.extend(info["posts"])
        else:
            info = fetch_twitter_user_and_posts(twitter_client, username=None,
                                                query_name=customer_name, max_posts=max_posts)
            posts_all.extend(info["posts"])

        # 5) Engagement aggregation
        if posts_all:
            total_likes = sum(p.get("likes", 0) or 0 for p in posts_all)
            total_retweets = sum(p.get("retweets", 0) or 0 for p in posts_all)
            result["engagement"] = {
                "num_posts": len(posts_all),
                "total_likes": int(total_likes),
                "total_retweets": int(total_retweets),
                "avg_likes": float(total_likes) / len(posts_all),
                "avg_retweets": float(total_retweets) / len(posts_all)
            }
            # sample posts
            result["posts_sample"] = posts_all[:10]
        else:
            result["notes"].append("No public posts found via Twitter API search.")

        # 6) Analyze posts for extremist / criminal signals
        analysis = analyze_posts_for_flags(result["posts_sample"], classifier)
        result["extremist_affiliation"] = analysis["extremist"]
        result["criminal_records_flag"] = analysis["criminal"]
        result["posts_analysis"] = analysis["detailed"]

        # 7) Criminal records check: **do not** try to scrape govt records.
        # Use an approved provider. Here we return None placeholder with guidance.
        result["criminal_records_source"] = None
        result["notes"].append("Criminal records checks require licensed providers (not performed here).")

        # Always return the result
        return result


    # How to use it (quick example)
    res = check_customer_socials(
        customer_name="John Doe",
        customer_identifiers={"twitter": "johndoe"},
        twitter_bearer=os.getenv("TWITTER_BEARER"),
        watchlist_csv=WATCHLIST_CSV,
        max_posts=5
    )
    # optional
    print(res["extremist_affiliation"], res["watchlist_matches"])

    # Integration Snippet for Streamlit
    #from social_checker import check_customer_socials  # put helper code in social_checker.py

    # Sidebar Input
    with st.sidebar:
        st.info("#### 🕵️ Social Search & Watchlist Check")
        customer_name = st.text_input("**Customer Name** (Enter exact customer name to run social search)")
        twitter_handle = st.text_input("**Twitter Username (optional)**")
        run_social_check = st.button("**Run Social Checks**")

    # Main Panel Results
    if run_social_check and customer_name:
        st.info("###### 🛰️ Social Search and Watchlist Checks")
        with st.spinner("**Running checks...**"):
            result = check_customer_socials(
                customer_name=customer_name,
                customer_identifiers={"twitter": twitter_handle} if twitter_handle else {},
                twitter_bearer=os.getenv("TWITTER_BEARER"),  # set in your .env
                watchlist_csv=WATCHLIST_CSV,
                max_posts=5
            )

        st.success("Search and Checks completed")

        # Display key flags
        st.write("**Customer Name:**", result["customer_name"])
        st.write("**Extremist Affiliation Detected?**", result["extremist_affiliation"])
        st.write("**Criminal Activity Flagged?**", result["criminal_records_flag"])

        # Watchlist
        if result["watchlist_matches"]:
            st.info("###### 🚨 Watchlist Matches")
            st.table(result["watchlist_matches"])
        else:
            st.info("No close matches found in watchlist.")
        st.write(f'**Search result returned {len(result["watchlist_matches"])} name matches**')

        # Engagement summary
        if result["engagement"]:
            st.info("####### 📊 Engagement Metrics")
            st.json(result["engagement"])

        # Posts sample
        if result["posts_sample"]:
            st.info("###### 📝 Recent Posts (Sample)")
            for p in result["posts_sample"]:
                st.write(f"- {p['text']} ({p['created_at']}) | Likes: {p['likes']} RTs: {p['retweets']}")

        # Analysis
        if "posts_analysis" in result:
            st.info("###### 🔎 Post-Level Analysis")
            st.json(result["posts_analysis"])

        # Notes
        if result["notes"]:
            st.info("###### ⚠️ Notes")
            for n in result["notes"]:
                st.warning(n)

    # Step 1: Extend Portfolio Aggregation (Portfolio summary so you can view aggregate OSINT risk across customers, side
    # by-side with your fraud portfolio metrics.)

    def portfolio_social_summary(df, max_customers=50):
        """
        Run OSINT screening across portfolio and return aggregate stats.
        """
        global flagged_df
        results = []
        for i, row in df.head(max_customers).iterrows():
            name = row.get("Customer_Name", None)
            if not name:
                continue
            try:
                check = check_customer_socials(
                    customer_name=name,
                    customer_identifiers={},
                    twitter_bearer=os.getenv("TWITTER_BEARER"),
                    watchlist_csv=WATCHLIST_CSV,
                    max_posts=10
                )
                results.append(check)
            except Exception as e:
                results.append({"customer_name": name, "error": str(e)})
            return results

        # Step 2: Portfolio Summary Section in Streamlit
        # In your Portfolio Summary tab/button handler: ##################################
    if run_social_check and customer_name:
        st.info("###### 📊 Search Summary")
        with st.spinner("**Running OSINT checks across portfolio...**"):
            social_results = portfolio_social_summary(df, max_customers=30)

        # Count risk flags
        extremist_count = sum(r.get("extremist_affiliation", False) for r in social_results)
        criminal_count = sum(r.get("criminal_records_flag", False) for r in social_results)
        watchlist_hits = sum(bool(r.get("watchlist_matches")) for r in social_results)

        st.write(f"**Extremist Affiliation Flags:** {extremist_count}")
        st.write(f"**Criminal Record Flags:** {criminal_count}")
        st.write(f"**Watchlist Matches:** {len(result["watchlist_matches"])}")

    import os
    import pandas as pd
    from difflib import get_close_matches
    # -----------------------
    # Config paths
    # -----------------------
    WATCHLIST_CSV = "C:/Users/Springrose/Downloads/FRAUD_DETECTION/GlobalTerrorismDataset.csv"  # CSV with column 'name' (and optional aliases)
    GLOBAL_TERROR_TXT = "C:/Users/Springrose/Downloads/FRAUD_DETECTION/globalterrorism.txt"

    # -----------------------
    # Core check function
    # -----------------------
    def check_watchlist(watchlist_path: str, terror_list_path: str = None):
        """
        Checks if the customer name appears in the local watchlist or global terror text.
        Returns a structured dictionary with notes, matches, and flag status.
        """
        result = {
            "notes": [],
            "matches": [],
            "flagged": False
        }

        # Validate name
        if not customer_name or not isinstance(customer_name, str):
            result["notes"].append("Invalid or missing customer name.")
            return result

        # 1️⃣ Check local CSV watchlist
        if watchlist_path and os.path.exists(WATCHLIST_CSV):
            try:
                watch_df = pd.read_csv(WATCHLIST_CSV, encoding='latin1')
                if "name" not in watch_df.columns:
                    result["notes"].append("Watchlist missing 'name' column.")
                else:
                    names = watch_df["name"].astype(str).tolist()
                    matches = get_close_matches(customer_name, names, cutoff=0.8)

                    if matches:
                        result["matches"].extend(matches)
                        result["flagged"] = True
                        result["notes"].append(f"🚨 **Name matched in {os.path.basename(WATCHLIST_CSV)}**")
            except Exception as e:
                result["notes"].append(f"Watchlist check failed: {e}")
        else:
            result["notes"].append("Watchlist file not found")

        # 2️⃣ Check optional secondary list (like global terror)
        if terror_list_path and os.path.exists(GLOBAL_TERROR_TXT):
            try:
                with open(GLOBAL_TERROR_TXT, "r", encoding="latin1") as f:
                    text_names = [line.strip() for line in f if line.strip()]
                matches = get_close_matches(customer_name, text_names, cutoff=0.8)

                if matches:
                    result["matches"].extend(matches)
                    result["flagged"] = True
                    result["notes"].append(f"🚨 **Name matched in {os.path.basename(GLOBAL_TERROR_TXT)}**")
            except Exception as e:
                result["notes"].append(f"Terror list check failed: {e}")
        else:
            result["notes"].append("Global terror file not found")

        return result

    # -----------------------
    # Streamlit UI Example
    # -----------------------
    #st.info("###### 🕵️ OSINT & Watchlist Screening")
    #customer_name = st.text_input("Enter customer name for screening:")
    #if st.button("**Run Screening**"):

    if run_social_check:
        with st.spinner("**Checking global watchlists...**"):
            res = check_watchlist(WATCHLIST_CSV, GLOBAL_TERROR_TXT)

        #st.write("###### 🔎 Screening Results")
        st.write("**Name Flagged:**", res["flagged"])
        if res["matches"]:
            st.success(f"️️️🕵️ **Matched names:** {', '.join(res['matches'])}")
        if res["notes"]:
            st.info(" | ".join(res["notes"]))