import os
import time
from PIL import Image
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import plotly.express as px
from transformers import pipeline
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier

# --- FILE PATHS & CONFIG ---
# NOTE: Removed hardcoded data loading (df = pd.read_excel) and hardcoded image paths.
# The app now relies entirely on the st.file_uploader for data.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers alias
SUMMARIZER_MODEL = "facebook/bart-large-cnn"  # optional, heavy
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
EMBED_CACHE = os.path.join(MODEL_DIR, "embed_cache.pkl")
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_clf.joblib")
PREPROC_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")

@st.cache_resource
def load_embedder(model_name=EMBED_MODEL_NAME):
    """Loads the Sentence Transformer model for text embeddings."""
    return SentenceTransformer(model_name)

@st.cache_resource
def load_summarizer(model_name2=SUMMARIZER_MODEL):
    """Loads the BART summarization pipeline."""
    try:
        return pipeline("summarization", model=model_name2)
    except Exception:
        # Fail silently if the model is too large or fails to load
        return None

# ---------------------------
# DATA PREPROCESSING
# ---------------------------
def parse_dates(df: pd.DataFrame):
    """Parse dates and engineer time-based features."""
    date_cols = ["Incident_Date", "Claim_Submission_Date", "Policy_Start_Date", "Policy_End_Date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Engineered features
    df["days_to_submit"] = 0
    df["policy_duration_days"] = 0
    if {"Incident_Date", "Claim_Submission_Date"}.issubset(df.columns):
        df["days_to_submit"] = (df["Claim_Submission_Date"] - df["Incident_Date"]).dt.days.fillna(0).clip(lower=0)
    if {"Policy_Start_Date", "Policy_End_Date"}.issubset(df.columns):
        df["policy_duration_days"] = (df["Policy_End_Date"] - df["Policy_Start_Date"]).dt.days.fillna(0).clip(lower=0)
    return df

# ---------------------------
# EMBEDDINGS (CACHED)
# ---------------------------
def get_embeddings(texts: List[str], embedder: SentenceTransformer):
    """Computes/loads embeddings with file-based cache."""
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
    """Builds the column transformer for structural features."""
    num_transform = StandardScaler()
    # Use drop='first' on OneHotEncoder for non-linear models like XGBoost to reduce multicollinearity
    cat_transform = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preproc = ColumnTransformer([
        ("num", num_transform, numeric_cols),
        ("cat", cat_transform, cat_cols)
    ], remainder='drop', verbose_feature_names_out=True)

    # Drop rows with NaN in critical features before fitting preprocessor
    df_clean = df.dropna(subset=numeric_cols + cat_cols)
    preproc.fit(df_clean[numeric_cols + cat_cols])
    return preproc

# ---------------------------
# TRAIN / LOAD MODEL
# ---------------------------
def train_and_save_model(X_struct, embeddings, y):
    """Trains the XGBoost model and evaluates it."""
    # Ensure X_struct is dense for hstack if it's small (sparse_output=False in OHE helps)
    if isinstance(X_struct, csr_matrix):
        X_final = hstack([X_struct, csr_matrix(embeddings)]).toarray()
    else:
        X_final = np.hstack([X_struct, embeddings])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2,
                                                        stratify=y if len(np.unique(y)) > 1 else None,
                                                        random_state=42)
    st.info("Training XGBoost Model. This may take a few minutes for large data.")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False  # Suppress future warning
    )

    xgb_clf.fit(X_train, y_train)

    # --- EVALUATE ---
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]

    # Only calculate AUC if both classes are present in test set
    if len(np.unique(y_test)) > 1:
        st.write(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    st.write("Classification report:")
    st.text(classification_report(y_test, y_pred))
    st.write("Confusion matrix:")
    st.text(str(confusion_matrix(y_test, y_pred)))
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")

    joblib.dump(xgb_clf, MODEL_FILE)
    st.success("Model trained and saved.")
    return xgb_clf

def load_model_if_exists():
    """Loads saved model and preprocessor."""
    if os.path.exists(MODEL_FILE) and os.path.exists(PREPROC_FILE):
        try:
            xgb_clf = joblib.load(MODEL_FILE)
            preproc = joblib.load(PREPROC_FILE)
            return xgb_clf, preproc
        except Exception:
            return None, None
    return None, None

# ---------------------------
# SHAP EXPLAINERS (CRITICAL CORRECTION)
# ---------------------------
@st.cache_resource
def make_shap_explainer(_xgb_clf):
    """Initializes the SHAP Tree Explainer."""
    return shap.TreeExplainer(_xgb_clf)

def explain_instance(xgb_clf, explainer, X_struct_row, emb_row, preproc, top_k=6):
    """
    Generates SHAP values for a single instance.
    CRITICAL FIX: Uses shap_values()[1] for the positive class (Fraud).
    """
    # Ensure structural features are dense and 2D for hstack
    if isinstance(X_struct_row, csr_matrix):
        X_struct_row = X_struct_row.toarray()
    X_struct_row = X_struct_row.reshape(1, -1)
    emb_row = emb_row.reshape(1, -1)

    X_combined = np.hstack([X_struct_row, emb_row])

    proba = float(xgb_clf.predict_proba(X_combined)[0, 1])
    pred = int(proba > 0.5)

    # 💥 CORRECTION 1: Use index [1] to get SHAP values for the FRAUD class
    shap_vals = explainer.shap_values(X_combined)[0]  # [1] for Class 1, [0] for the single instance

    # 💥 CORRECTION 2: Robust feature name generation
    try:
        struct_names = list(preproc.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        num_names = preproc.transformers_[0][2]
        cat_encoder = preproc.transformers_[1][1]
        cat_names = list(cat_encoder.get_feature_names_out(preproc.transformers_[1][2]))
        struct_names = list(num_names) + cat_names

    # The number of embedding features depends on the model, typically 384
    embed_dim = emb_row.shape[-1]
    embed_names = [f"Adjuster_Notes_Embed_{i}" for i in range(embed_dim)]
    all_feature_names = struct_names + embed_names

    # Handle cases where feature name list length doesn't match shap_vals length
    if len(all_feature_names) != len(shap_vals):
        st.error(
            f"Feature name count ({len(all_feature_names)}) does not match SHAP value count ({len(shap_vals)}). Cannot display SHAP table.")
        return {
            "prediction": "**ERROR**", "fraud_probability": proba, "top_shap": [],
            "all_feature_names": all_feature_names, "shap_values": shap_vals
        }

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
    """Aggregates SHAP explanations across multiple claims."""
    total = len(explanations)
    fraud_count = sum(1 for e in explanations if e["prediction"] == "**Potentially fraudulent claim**")
    avg_prob = np.mean([e["fraud_probability"] for e in explanations]) if total > 0 else 0.0

    # Aggregate top features frequency and average impact
    flat = {}
    for e in explanations:
        for feat in e["top_shap"]:
            name = feat["**Top Selected Features**"]
            value = feat["**SHAP Values**"]
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

def run_portfolio_summary(df, summarizer=None):
    """Provides a basic quantitative summary."""
    stats = {
        "num_claims": len(df),
        "avg_claim_amount": df["Claim_Amount"].mean(),
        "fraud_rate": df["Fraud_Flag"].mean() if "Fraud_Flag" in df.columns else np.nan,
    }

    fraud_rate_text = f"Estimated fraud rate: **{stats['fraud_rate']:.1%}**." if pd.notna(
        stats['fraud_rate']) else "Target (Fraud_Flag) not available."

    base_summary = (
        f"Portfolio has **{stats['num_claims']}** claim requests. "
        f"Average claim amount: **₦{stats['avg_claim_amount']:.2f}**. "
        f"{fraud_rate_text}"
    )
    st.info(base_summary)

    if summarizer:
        try:
            # Simple summarization for the base stats
            response = summarizer(
                base_summary,
                max_length=60,
                min_length=20,
                do_sample=False
            )
            return response[0]["summary_text"]
        except Exception:
            return f"Fallback summary: {base_summary}"
    return base_summary

# ---------------------------
# STREAMLIT UI - MAIN APP
# ---------------------------

# Use a generic image placeholder since the original path is not portable
st.markdown('<h1 style="color:#000080; font-size: 38px;">AXA Mansard Claims Analyzer</h1>', unsafe_allow_html=True)
st.markdown("##### Life and Non-Life Insurance Claims Analyzer and Anomaly Detector")
st.image("https://placehold.co/150x80/000080/ffffff?text=AXA+Logo", width=80)
st.image("https://placehold.co/150x50/3498db/ffffff?text=Insight+Logo", width=150)
st.divider()

# Initial state for DataFrame
df = pd.DataFrame()

with st.sidebar:
    st.header("**Control Panel**")
    uploaded = st.file_uploader("**Upload Claims Data (Tabular)**", type=["csv", "txt", "xlsx"],
                                help="Claims data with columns like Claim_Amount, Adjuster_Notes, Fraud_Flag")

    use_pretrained = st.checkbox("Load saved preprocessor and model if available", value=True)
    enable_summarizer = st.checkbox("Enable natural language summaries (may load large model)", value=False)
    run_training = st.button("**Train model from uploaded data**")
    st.markdown("---")
    st.markdown("**Notes:**\n- Embeddings cached to reduce repeated calls.\n- Model files are saved locally.")
    st.markdown("---")

summarizer = None
if enable_summarizer:
    # Load summarizer with progress spinner
    with st.spinner("Loading summarization model..."):
        summarizer = load_summarizer()
        if summarizer is None:
            st.warning("Summarizer could not be loaded. Please check model availability.")

# --- DATA LOAD AND PRE-PROCESS ---
if uploaded is not None:
    try:
        # Load logic
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        elif uploaded.name.endswith(".txt"):
            try:
                df = pd.read_csv(uploaded, sep=",")
            except Exception:
                df = pd.read_csv(uploaded, sep="\t")
        elif uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            st.error("Unsupported file type.")
            df = pd.DataFrame()

        if not df.empty:
            st.info("###### 🔍 **Claims Data Preview**")
            st.write(df.head())

            # Basic cleaning (removed Customer_Phone hardcode)
            if 'Customer_Phone' in df.columns:
                df['Customer_Phone'] = df['Customer_Phone'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

            # Parse dates & basic cleaning
            df = parse_dates(df)

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        df = pd.DataFrame()

    if not df.empty:
        # Check required columns
        text_col = "Adjuster_Notes" if "Adjuster_Notes" in df.columns else None
        if text_col is None:
            st.error("Claims data must include 'Adjuster_Notes' column for text embeddings.")
            st.stop()

        # Define features
        numeric_cols = [c for c in
                        ["Claim_Amount", "Customer_Age", "Premium_Amount", "days_to_submit", "policy_duration_days"] if
                        c in df.columns]
        cat_cols = [c for c in ["Policy_Type", "Claim_Type", "Incident_Type", "Claim_Status", "Customer_Gender",
                                "Customer_Occupation", "Location"] if c in df.columns]
        target_col = "Fraud_Flag" if "Fraud_Flag" in df.columns else None

        if target_col is None:
            st.warning("No 'Fraud_Flag' column found. Only prediction is possible, training is disabled.")

        # Load embedder and compute embeddings
        embedder = load_embedder()
        with st.spinner("Computing / loading embeddings (cached)..."):
            # Filter NaN text column before getting embeddings
            texts = df[text_col].fillna("").astype(str).tolist()
            embeddings = get_embeddings(texts, embedder)
        st.success(f"Computed embeddings: Shape {embeddings.shape}")

        # Build or load preprocessor
        xgb_clf, preproc = None, None
        if use_pretrained:
            xgb_clf, preproc = load_model_if_exists()
            if xgb_clf is not None:
                st.success("Loaded saved model and preprocessor.")
            else:
                preproc = build_preprocessor(df, numeric_cols, cat_cols)
                joblib.dump(preproc, PREPROC_FILE)
                st.info("Built and saved new preprocessor.")
        else:
            preproc = build_preprocessor(df, numeric_cols, cat_cols)
            joblib.dump(preproc, PREPROC_FILE)
            st.info("Built and saved new preprocessor.")

        # If user clicked train and target is available
        if run_training and target_col is not None:
            X_struct = preproc.transform(df[numeric_cols + cat_cols])
            # Ensure target is integers
            y = df[target_col].astype(int).values
            xgb_clf = train_and_save_model(X_struct, embeddings, y)

        # If model is available, enable interaction
        if xgb_clf is not None:
            explainer = make_shap_explainer(xgb_clf)
            st.divider()

            ######################################################################

            st.info("###### 🔍 Single Claim Inspection and Analysis")
            claim_index = st.number_input("Row index to inspect (0 based)", min_value=0,
                                          max_value=len(df) - 1, value=0, step=1)

            # Trigger explanation when button is clicked
            if st.button("Inspect and analyze selected claim"):
                try:
                    # Filter and display the selected row
                    selected_row = df.loc[[claim_index]]

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

                # Natural language explanation (IMPROVED)
                st.markdown("**SHAP Interpretation:**")
                interpretation_lines = []
                for _, row in shap_df.iterrows():
                    feature, value = row["**Top Selected Features**"], row["**SHAP Values**"]
                    impact = "increased" if value > 0 else "decreased"
                    direction = "a risk factor" if value > 0 else "a mitigating factor"

                    # Try to get the feature value for context (works well for one-hot and numerical, less for embeddings)
                    try:
                        feature_name_base = feature.split('_')[0] if "cat__" in feature else feature
                        feature_value = selected_row.iloc[0][feature_name_base]

                        if "Adjuster_Notes" in feature:
                            value_text = "from the text analysis"
                        elif "cat__" in feature:
                            value_text = f"because the claim has feature: *{feature.split('__')[-1]}*"
                        else:
                            value_text = f"with a value of **{feature_value}**"
                    except:
                        value_text = ""

                    line = (f"The feature **{feature}** {value_text} ({direction}) by a SHAP value of "
                            f"**{value:.3f}**, significantly {impact} fraud probability.")
                    interpretation_lines.append(line)

                st.write("\n\n".join(interpretation_lines))

                st.divider()

                # Natural language summary (optional)
                if summarizer is not None:
                    # Build context string
                    top_factors = ", ".join(
                        [f"{f['**Top Selected Features**']} (impact: {f['**SHAP Values**']:.3f})" for f in
                         explanation["top_shap"]]
                    )
                    context = (
                        f"Prediction: {explanation['prediction']}. "
                        f"Fraud probability: {explanation['fraud_probability']}. "
                        f"Top factors: {top_factors}. Claim details: {selected_row.to_dict('records')[0]}"
                    )
                    with st.spinner("Generating natural language summary..."):
                        try:
                            summ = summarizer[0]["summary_text"]
                            st.markdown("**Natural language summary:**")
                            st.write(summ)

                        except Exception as e:
                            st.warning("Summarizer failed: " + str(e))

                        except Exception as e:
                            st.error(f"An error occurred while inspecting the claim: {e}")

                            # ----------------------------------------------------------------------
                            # CLAIM BATCH ANALYSIS
                            # ----------------------------------------------------------------------
                            st.info("###### 📄 Claim Batch Analysis")
                            import uuid

                            def unique_key(base: str) -> str:
                                return f"{base}_{uuid.uuid4().hex[:6]}"

                            # Filtering logic is kept inside the form to re-run only on form submit
                            with st.form(unique_key("claims_filters_form")):
                                st.write("ℹ️ Adjust filters, click **Apply Filters** to view result")

                                mask = pd.Series(True, index=df.index)

                                # --- Date range filter ---
                                sub_date_range = None
                                if "Claim_Submission_Date" in df.columns and pd.api.types.is_datetime64_any_dtype(
                                        df["Claim_Submission_Date"]):
                                    min_date = df["Claim_Submission_Date"].min()
                                    max_date = df["Claim_Submission_Date"].max()

                                    if pd.notna(min_date) and pd.notna(max_date):
                                        sub_date_range = st.date_input(
                                            "**Submission date range:**",
                                            value=(min_date.date(), max_date.date()),
                                            key=unique_key("submission_date_filter")
                                        )

                                # --- Location filter ---
                                loc_choice = "All"
                                if "Location" in df.columns:
                                    locs = ["All"] + sorted(df["Location"].dropna().unique().tolist())
                                    loc_choice = st.selectbox("**Location filter:**", locs, index=0, key=unique_key("location_filter"))

                                # --- Policy Type filter ---
                                p_choice = "All"
                                if "Policy_Type" in df.columns:
                                    ptypes = ["All"] + sorted(df["Policy_Type"].dropna().unique().tolist())
                                    p_choice = st.selectbox("**Policy type filter:**", ptypes, index=0,
                                                            key=unique_key("policy_type_filter"))

                                # --- Apply button ---
                                apply_filters = st.form_submit_button("**Apply Filters**")

                            # --- Apply filters after button click ---
                            if apply_filters:
                                if sub_date_range and isinstance(sub_date_range, tuple) and len(sub_date_range) == 2:
                                    start_date = pd.to_datetime(sub_date_range[0])
                                    end_date = pd.to_datetime(sub_date_range[1])
                                    mask &= (df["Claim_Submission_Date"].dt.date >= start_date.date()) & (
                                            df["Claim_Submission_Date"].dt.date <= end_date.date())

                                if "Location" in df.columns and loc_choice != "All":
                                    mask &= (df["Location"] == loc_choice)

                                if "Policy_Type" in df.columns and p_choice != "All":
                                    mask &= (df["Policy_Type"] == p_choice)

                                filtered_df = df[mask].reset_index(drop=False)  # Keep original index for embedding lookup
                                st.success(f"✅ {filtered_df.shape[0]} claims matched filters.")
                                st.dataframe(filtered_df.head())

                                # Store filtered DF in session state to use outside the form
                                st.session_state['filtered_df'] = filtered_df

                            # --- Continue analysis outside the form if a filtered set exists ---
                            if 'filtered_df' in st.session_state:
                                df_sub = st.session_state['filtered_df']

                                st.divider()
                                st.info(f"**{len(df_sub)}** claim requests selected for in-depth analysis.")

                                if st.button("Run Portfolio Explanations"):
                                    if df_sub.empty:
                                        st.warning("No claims selected.")
                                    else:
                                        if "Claim_Amount" not in df_sub.columns or target_col is None:
                                            st.error(
                                                "Dataset must contain 'Claim_Amount' and 'Fraud_Flag' columns for a meaningful summary.")
                                        else:
                                            with st.spinner("Generating portfolio summary..."):
                                                run_portfolio_summary(df_sub, summarizer)

                                            # Explain each claim (Expensive operation)
                                            explanations = []

                                            # Transform structural features for the filtered set
                                            X_struct_all = preproc.transform(df_sub[numeric_cols + cat_cols])

                                            # Subset embeddings using the original index from the filtered dataframe
                                            original_indices = df_sub['index'].values
                                            emb_sub = embeddings[original_indices]

                                            with st.spinner(
                                                    f"Computing SHAP explanations for all {len(df_sub)} claims. This may take some time."):
                                                for i in range(len(df_sub)):
                                                    try:
                                                        expl = explain_instance(xgb_clf, explainer, X_struct_all[i],
                                                                                emb_sub[i], preproc, top_k=6)
                                                        explanations.append(expl)
                                                    except Exception as e:
                                                        st.error(f"Error explaining claim index {original_indices[i]}: {e}")

                                            if explanations:
                                                agg = portfolio_aggregate(explanations)
                                                st.write("**Portfolio stats:**", {
                                                    "total_claims": agg["total_claims"],
                                                    "fraudulent_claims": agg["fraudulent_claims"],
                                                    "legit_claims": agg["legit_claims"],
                                                    "average_fraud_probability": f"{agg['avg_fraud_probability']:.4f}"
                                                })
                                                st.info("###### 📄 Top contributing features (aggregated)")
                                                st.dataframe(agg["top_features"].head(20))

                                                # Visualization (kept original logic)
                                                probs = [e["fraud_probability"] for e in explanations]
                                                probs_df = pd.DataFrame({"Fraud Probability": probs})
                                                fig = px.histogram(probs_df, x="Fraud Probability", nbins=20,
                                                                   title="Fraud Probability Distribution Across Portfolio")
                                                st.plotly_chart(fig, use_container_width=True)

                                                # Visualization by Policy Type
                                                if "Policy_Type" in df_sub.columns:
                                                    labels = [e["prediction"] for e in explanations]
                                                    df_sub2 = df_sub.copy()
                                                    df_sub2["Predicted Label"] = labels
                                                    fig2 = px.histogram(df_sub2, x="Policy_Type", color="Predicted Label",
                                                                        barmode="group", title="Predicted Labels by Policy Type")
                                                    st.plotly_chart(fig2, use_container_width=True)

                                                # Natural language portfolio summary
                                                if summarizer is not None:
                                                    context = (f"Portfolio summary: {agg['total_claims']} claims, "
                                                               f"{agg['fraudulent_claims']} flagged as Fraudulent, "
                                                               f"average fraud probability {agg['avg_fraud_probability']:.2f}. "
                                                               f"Top features: " + ", ".join(
                                                        agg["top_features"]["feature"].head(10).tolist()))
                                                    with st.spinner("Generating portfolio natural language summary..."):
                                                        try:
                                                            summ = summarizer(
                                                                context,
                                                                max_length=150,
                                                                min_length=40,
                                                                do_sample=False
                                                            )[0]["summary_text"]
                                                            st.markdown("###### 📄 Natural language portfolio summary")
                                                            st.write(summ)
                                                        except Exception as e:
                                                            st.warning("Summarizer failed: " + str(e))
                                            else:
                                                st.warning("No successful explanations were generated.")
                        else:
                            st.info("No model available yet. Train a model or enable pre-trained loading.")
    else:
        st.info("Upload claims data in the sidebar to get started. "
                "Claims data should contain at least 'Adjuster_Notes' and ideally numeric/categorical columns "
                "such as Claim_Amount, Premium_Amount, Customer_Age, Policy_Type, Claim_Type, Incident_Type, "
                "Location and Fraud_Flag.")

st.divider()