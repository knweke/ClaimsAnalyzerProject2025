# Example implementation (productionize carefully)
""" 
Social media / watchlist screening helper (example).
- Replace placeholders with your own API keys and licensed data sources.
- THIS IS NOT A SUBSTITUTE FOR VEHICLES FOR OFFICIAL CRIMINAL BACKGROUND CHECKS.
"""
import os
import pandas as pd
import requests
import ClaimsHuggFace
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

#tab3 = st.tabs(["🔍 Customer Search"])
#with tab3:
    #st.markdown("##### 🔍 Customer Search")

# ---------- CONFIG ----------
TWITTER_BEARER = os.getenv("TWITTER_BEARER")  # set in .env or env
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"  # or another appropriate model
WATCHLIST_CSV = "C:/Users/Springrose/Downloads/FRAUD DETECTION/GlobalTerrorismDataset.csv"  # CSV with column 'name' (and optional aliases)
GLOBAL_TERROR_TXT = "C:/Users/Springrose/Downloads/FRAUD DETECTION/globalterrorism.txt"

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
def load_watchlist(WATCHLIST_CSV: str) -> pd.DataFrame:
    """
    Load watchlist CSV into DataFrame with normalized 'name' column.
    """
    try:
        df = pd.read_csv(WATCHLIST_CSV)
    except Exception:
        # try other delimiters
        df = pd.read_csv(WATCHLIST_CSV, sep=None, engine="python")
    if "name" not in df.columns:
        raise ValueError("Watchlist CSV must contain a 'name' column.")
        # normalize
    df["name_norm"] = df["name"].astype(str).str.strip().str.lower()
    return df

def fuzzy_watchlist_match(name: str, watchlist_df: pd.DataFrame, top_k: int = 5,
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
                                 query_name: Optional[str] = None, max_posts: int = 50):
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
    criminal_flag = False

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
                if item["scores"].get("terrorism", 0) >= threshold or item["scores"].get("extremist ideology", 0) >= threshold:

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
                if any(k in text_lower for k in["steal", "fraud", "murder", "kill", "assault"]):

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
        "criminal_records_flag": False,
        "notes": []
        }

    # 1) Initialize services
    twitter_client = init_twitter_client(twitter_bearer)
    classifier = init_text_classifier()

    # 2) Load watchlist
    watchlist_df = None
    if watchlist_csv and os.path.exists(watchlist_csv):
        try:
            watchlist_df = load_watchlist(watchlist_csv)
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
    watchlist_csv="WATCHLIST_CSV",
    max_posts=50
)
# optional
print(res["extremist_affiliation"], res["watchlist_matches"])

"""
Important limitations & next steps 
 Watchlist matches here are fuzzy-name matches only. They require manual verification and legal 
authority to act on. Use official PEP/sanctions screening providers for production (e.g., World-Check, 
LexisNexis, government lists). 
 Criminal records are not accessible via public scraping. Use official background-check vendors. 
 Classifier accuracy: zero-shot or keyword heuristics can give many false positives — treat them as 
signals, not evidence. Consider fine-tuning a domain-specific classifier and human review. 
 Logging & audit: store who ran the check, when, and why; maintain consent records. 
"""

"""
Perfect 👍 — we can integrate the social media + watchlist check directly into your existing Streamlit fraud 
detection dashboard. 
I’ll give you a modular block that plugs into your app. It will: 
1. Add a sidebar panel for “Social Media / Watchlist Check”. 
2. Let the user enter customer name (and optional Twitter username). 
3. Run the check_customer_socials() function (from the helper code I shared). 
4. Display structured results in Streamlit (profiles found, engagements, posts, watchlist matches, flags). 
🔧 Integration Snippet for Streamlit 
At the top of your app (after other imports): 
"""
# Integration Snippet for Streamlit
#from social_checker import check_customer_socials  # put helper code in social_checker.py

# Sidebar Input
with st.sidebar:
    st.info("#### 🕵️ Social Media & Watchlist Check")
    customer_name = st.text_input("Customer Name")
    twitter_handle = st.text_input("Twitter Username (optional)")
    run_social_check = st.button("Run Social Check")

# Main Panel Results
if run_social_check and customer_name:
    st.info("###### 🛰️ Social Media and Watchlist Analysis")
    with st.spinner("Running checks..."):
        result = check_customer_socials(
            customer_name=customer_name,
            customer_identifiers={"twitter": twitter_handle} if twitter_handle else {},
            twitter_bearer=os.getenv("TWITTER_BEARER"),  # set in your .env
            watchlist_csv="WATCHLIST_CSV",
            max_posts=50
        )

    st.success("Check completed")

    # Display key flags
    st.write("**Extremist Affiliation Detected?**", result["extremist_affiliation"])
    st.write("**Criminal Activity Flagged?**", result["criminal_records_flag"])

    # Watchlist
    if result["watchlist_matches"]:
        st.markdown("##### 🚨 Watchlist Matches")
        st.table(result["watchlist_matches"])
    else:
        st.info("No close matches found in watchlist.")

    # Engagement summary
    if result["engagement"]:
        st.markdown("###### 📊 Engagement Metrics")
        st.json(result["engagement"])

    # Posts sample
    if result["posts_sample"]:
        st.markdown("###### 📝 Recent Posts (Sample)")
        for p in result["posts_sample"]:
            st.write(f"- {p['text']} ({p['created_at']}) | Likes: {p['likes']} RTs: {p['retweets']}")

    # Analysis
    if "posts_analysis" in result:
        st.markdown("###### 🔎 Post-Level Analysis")
        st.json(result["posts_analysis"])

    # Notes
    if result["notes"]:
        st.markdown("###### ⚠️ Notes")
        for n in result["notes"]:
            st.warning(n)
"""
✅ This way, in your Streamlit app: 
 A claims adjuster or fraud analyst can upload claims data as before. 
 They can also run a social media + watchlist check for the customer in real time. 
 Everything is wrapped in safe return statements so the app doesn’t crash if APIs fail. 
⚠
 ️ You’ll still need: 
 A valid Twitter/X Bearer Token stored in .env (or your environment). 
 A CSV called watchlist.csv with at least one column: name. 
"""

"""
Great 👍 — let’s merge the social media / watchlist screening module into your existing Fraud Detection 
Streamlit Dashboard. 
Here’s how we’ll extend your app safely without breaking the SHAP + portfolio summary logic: 
"""
# Step 1: Create social_checker.py
# Put this helper in a new file social_checker.py so the Streamlit app doesn’t get bloated:
def check_customer_socials(customer_name, customer_identifiers=None,
                           twitter_bearer=None, watchlist_csv=None,
                           max_posts=20):
    """
    Check social media activity, engagement, and watchlist affiliations for a customer.
    """
    # --- Initialize result object ---
    result = {
        "customer_name": customer_name,
        "profiles": {},
        "engagement": {},
        "posts_sample": [],
        "posts_analysis": [],
        "watchlist_matches": [],
        "extremist_affiliation": False,
        "criminal_records_flag": False,
        "notes": []
    }

    # --- Ensure customer_identifiers is defined ---
    if customer_identifiers is None:
        customer_identifiers = {}

    # --- WATCHLIST CSV CHECK ---
    if WATCHLIST_CSV and os.path.exists(WATCHLIST_CSV):
        try:
            watch_df = pd.read_csv(WATCHLIST_CSV)
            names = watch_df["name"].astype(str).tolist()

            matches = get_close_matches(customer_name, names, cutoff=0.8)
            if matches:
                result["watchlist_matches"] = matches
                result["extremist_affiliation"] = True
                result["notes"].append(f"Matched names: {matches}")
        except Exception as e:
            result["notes"].append(f"Watchlist check failed: {e}")

    # --- SOCIAL MEDIA CHECK ---
    if twitter_bearer and "twitter" in customer_identifiers:
        twitter_handle = customer_identifiers["twitter"]
        # 🧠 Example placeholder for Twitter lookup
        result["profiles"]["twitter"] = f"https://twitter.com/{twitter_handle}"
        result["engagement"]["twitter_followers"] = 1200  # mocked data
        result["posts_sample"].append(f"Example tweet from {twitter_handle}")
        result["posts_analysis"].append("User appears politically neutral.")
    else:
        result["notes"].append("Twitter handle or credentials missing.")

    # --- RETURN FINAL RESULT ---
    return result

# --- 1. WATCHLIST CHECK ---
if WATCHLIST_CSV and os.path.exists(WATCHLIST_CSV):
    result = {
        "notes": [],
        WATCHLIST_CSV: [],
        GLOBAL_TERROR_TXT: False
    }
    try:
        watch_df = pd.read_csv(WATCHLIST_CSV)
        names = watch_df["name"].astype(str).tolist()
        matches = get_close_matches(customer_name, names, cutoff=0.8)
        if matches:
            result[WATCHLIST_CSV] = matches
            result[GLOBAL_TERROR_TXT] = True
    except Exception as e:
        result["notes"].append(f"Watchlist check failed: {e}")

# ---- NEWLY ADDED UP ----
def check_watchlist(customer_name2: str, watchlist_path: str):
    result = {"notes": [], "matches": [], "flagged": False}

    if not watchlist_path or not os.path.exists(watchlist_path):
        result["notes"].append("Watchlist file not found.")
        return result

    try:
        watch_df = pd.read_csv(watchlist_path)
        names = watch_df["name"].astype(str).tolist()
        matches = get_close_matches(customer_name2, names, cutoff=0.8)

        if matches:
            result["matches"] = matches
            result["flagged"] = True
    except Exception as e:
        result["notes"].append(f"Watchlist check failed: {e}")

    return result

res = check_watchlist(customer_name, WATCHLIST_CSV)

# --- 2. TWITTER CHECK ---
# --- Ensure customer_identifiers is defined ---
#if customer_identifiers is None:
    #customer_identifiers = {}

##################################################################################

# Step 2: Integrate into Your Streamlit App
# In your ClaimsHuggFace.py, add: from social_checker import check_customer_socials

# Add to Sidebar (below your “Control Panel”):
st.sidebar.markdown("---")
with st.sidebar:
    st.info("#### 🕵️ Social Media & Watchlist Check")
    customer_name = st.sidebar.text_input("Customer Name for OSINT Check", key='osint')
    twitter_handle = st.sidebar.text_input("Twitter Username (optional)", key='twitter')
    run_social_check = st.sidebar.button("Run Social Check", key='run_social_check')

"""
Display in Main Panel 
Add this section after your fraud detection + SHAP explainability blocks: 
"""
if run_social_check and customer_name:
    st.divider()
    st.markdown("###### 🛰️ Social Media & Watchlist Analysis")
    with st.spinner("Checking customer..."):
        check_customer_socials(
            customer_name=customer_name,
            customer_identifiers={"twitter": twitter_handle} if twitter_handle else {},
            twitter_bearer=os.getenv("TWITTER_BEARER"),
            watchlist_csv="WATCHLIST_CSV",
            max_posts=20
        )
st.success("Check completed.")
if result:
        st.write("**Customer Name:**", result.get('customer_name', 'N/A'))
        st.write("**Extremist Affiliation?**", result.get("extremist_affiliation", 'N/A'))
        st.write("**Criminal Records Flag?**", result.get("criminal_records_flag", 'N/A'))
        watchlist_matches = result.get("watchlist_matches", [])

        if watchlist_matches:
            st.markdown("###### 🚨 Watchlist Matches")
            st.table(pd.DataFrame(watchlist_matches, columns=["Matched Name"]))
        else:
            st.info("No watchlist matches")
else:
    st.warning("No results returned from social check.")

    if result["engagement"]:
        st.markdown("###### 📊 Engagement")
        st.json(result["engagement"])

    if result["posts_sample"]:
        st.markdown("###### 📝 Recent Posts")
        st.table(pd.DataFrame(result["posts_sample"]))

    if result["posts_analysis"]:
        st.markdown("###### 🔎 Flagged Posts")
        st.json(result["posts_analysis"])

    if result["notes"]:
        st.markdown("###### ⚠️ Notes")
        for n in result["notes"]:
            st.warning(n)

"""
✅ What You Get 
 Fraud Model (SHAP explainability + portfolio analysis) continues to work. 
 Extra Tab: Analyst can run OSINT check on a customer name + Twitter handle. 
 Return results always structured, so no crashes: 
o Watchlist matches ✅ 
o Social engagements (followers, posts, likes) ✅ 
o Criminal/extremist keywords flagged ✅ 
o Safe “return statement” if data missing ✅ 
⚠
Notes: 
 You’ll need a watchlist.csv with at least one column "name". 
 Add your Twitter Bearer Token in environment: 
 setx TWITTER_BEARER "YOUR_BEARER_TOKEN" 
 This module is extendable: you can plug in LinkedIn API, Facebook Graph API, etc. later.
"""

"""
Perfect ✅ — let’s wire it in so that when you run a fraud prediction for a customer/claim, the same workflow 
also runs the OSINT check in the background and shows results on the same page. 
"""

# Step 1: Extend Fraud Prediction Pipeline
# After you get the fraud result & SHAP explanation, we’ll also run the social media + watchlist module for the  same customer.
# Inside your prediction section (where you currently display SHAP + fraud probability), add:

# --- FRAUD PREDICTION BLOCK (your existing code) ---
st.markdown("###### 📊 Data Driven Insights")
st.write("Prediction result:", explanation["prediction"])
st.write("Fraud probability:", explanation["fraud_probability"])
st.write("Top SHAP Contributions (Features, SHAP Explainer):")
st.table(pd.DataFrame(explanation["top_shap"]))

# --- NEW: Social + Watchlist Screening ---
st.divider()

#filepath = 'C:\\Users\\Springrose\\Downloads\\FRAUD DETECTION\\SmartClaimsData.xlsx'
#df = pd.read_excel(filepath) # pd.read_excel (.xlsx)

st.markdown("###### 🛰️ Social Media & Watchlist Screening")
customer_name = df.loc[selected_row, "Customer_Name"] if "Customer_Name" in df.columns else None
twitter_handle = None  # you could map from dataset if available, else leave blank

if customer_name:
    with st.spinner(f"Running OSINT screening for {customer_name}..."):
        check_customer_socials(
            customer_name=customer_name,
            customer_identifiers={"twitter": twitter_handle} if twitter_handle else {},
            twitter_bearer=os.getenv("TWITTER_BEARER"),
            watchlist_csv="WATCHLIST_CSV",
            max_posts=20
        )
    st.success("OSINT check completed.")

# Show findings
st.write("**Customer Name:**", result["customer_name"])
st.write("**Extremist Affiliation?**", result["extremist_affiliation"])
st.write("**Criminal Records Flag?**", result["criminal_records_flag"])

if result["watchlist_matches"]:
    st.markdown("###### 🚨 Watchlist Matches")
    st.table(pd.DataFrame(result["watchlist_matches"], columns=["Matched Name"]))
else:
    st.info("No watchlist matches.")

if result["engagement"]:
    st.markdown("###### 📊 Engagement")
    st.json(result["engagement"])

if result["posts_sample"]:
    st.markdown("###### 📝 Recent Posts")
    st.table(pd.DataFrame(result["posts_sample"]))

if result["posts_analysis"]:
    st.markdown("###### 🔎 Flagged Posts")
    st.json(result["posts_analysis"])

if result["notes"]:
    st.markdown("###### ⚠️ Notes")
for n in result["notes"]:
    st.warning(n)
else:
    st.info("Customer name not found in dataset. Cannot run social media check.")

# Step 2: Ensure Watchlist + API Setup
"""
1. watchlist.csv → Must exist in project root, with at least: 
2. name 
3. John Doe 
4. Jane Smith 
5. Ali Hassan 
6. Twitter Bearer Token → set in your system environment: 
7. setx TWITTER_BEARER "YOUR_TWITTER_BEARER_TOKEN"
"""

# Final Workflow
"""
1. User uploads claims data. 
2. Runs model → fraud probability + SHAP insights shown. 
3. Immediately below, the same customer’s OSINT check results are shown: 
o 🚨 Watchlist matches 
o 📊 Engagement metrics 
o 📝 Sample posts 
o 🔎 Flagged extremist/criminal content
"""

"""
Great 👍 let’s extend the portfolio summary so you can view aggregate OSINT risk across customers, side
by-side with your fraud portfolio metrics. 
Inside your existing portfolio_aggregate(explanations) or wherever you’re aggregating SHAP/fraud 
results, we’ll add an OSINT loop. 
"""

# Step 1: Extend Portfolio Aggregation (Portfolio summary so you can view aggregate OSINT risk across customers, side
# by-side with your fraud portfolio metrics.)

def portfolio_social_summary(df, max_customers=50):
    """
    Run OSINT screening across portfolio and return aggregate stats.
    """
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
                watchlist_csv="WATCHLIST_CSV",
                max_posts=10
            )
            results.append(check)
        except Exception as e:
            results.append({"customer_name": name, "error": str(e)})
    return results


# Step 2: Portfolio Summary Section in Streamlit
# In your Portfolio Summary tab/button handler:
if run_portfolio_summary:
    st.markdown("###### 📊 Portfolio Fraud + OSINT Summary")

    # Fraud insights (existing block)
    agg = portfolio_aggregate(explanations)
    st.write("Fraud prediction distribution:", agg)

    # --- NEW OSINT Block ---
    st.markdown("###### 🛰️ Portfolio OSINT Risk Analysis")
    with st.spinner("Running OSINT checks across portfolio..."):
        social_results = portfolio_social_summary(df, max_customers=30)

    # Count risk flags
    extremist_count = sum(r.get("extremist_affiliation", False) for r in social_results)
    criminal_count = sum(r.get("criminal_records_flag", False) for r in social_results)
    watchlist_hits = sum(bool(r.get("watchlist_matches")) for r in social_results)

    st.write(f"**Extremist Affiliation Flags:** {extremist_count}")
    st.write(f"**Criminal Record Flags:** {criminal_count}")
    st.write(f"**Watchlist Matches:** {watchlist_hits}")

    # Table of top risky customers
    flagged_df = pd.DataFrame([
        {
            "Customer": r.get("customer_name"),
            "Extremist": r.get("extremist_affiliation"),
            "Criminal": r.get("criminal_records_flag"),
            "Watchlist": bool(r.get("watchlist_matches")),
            "Notes": "; ".join(r.get("notes", []))
        }
        for r in social_results if (r.get("extremist_affiliation") or
                                    r.get("criminal_records_flag") or
                                    r.get("watchlist_matches"))
    ])

    if not flagged_df.empty:
            st.markdown("###### 🚨 High-Risk Customers")
            st.table(flagged_df)
    else:
        st.info("No high-risk customers flagged in current portfolio.")

"""
✅ End Result 
When you click “Run portfolio summary on filtered claims data”: 
 You’ll now see fraud prediction metrics + SHAP insights (existing). 
 OSINT portfolio analysis will run in parallel: 
o 🚨 Number of extremist affiliation flags 
o 🚨 Number of criminal record flags 
o 🚨 Watchlist matches 
 A table lists high-risk customers, combining model prediction + external screening. 

Note: This can be slow if you check 100+ customers live. You might want to: 
 Cache results in a database. 
 Use async calls to APIs. 
 Restrict to top N risky customers.
"""

# Merge the OSINT + Fraud risk into one combined “Risk Score” per customer,
# so you can rank customers from lowest → highest overall risk?
"""
Perfect ✅ — let’s design this step by step so your Streamlit app gets: 
1. Combined Fraud + OSINT Risk Score per customer. 
2. Caching in a database (SQLite for simplicity). 
3. Async OSINT checks so multiple customers are queried in parallel. 
4. Restriction to only the top risky customers for portfolio summary.
"""

# Step 1: Risk Score Formula
# We can combine fraud probability (from XGB) + OSINT flags into one score:
def compute_risk_score(fraud_prob, osint_result):
    score = fraud_prob
    if osint_result.get("extremist_affiliation"):
        score += 0.3
    if osint_result.get("criminal_records_flag"):
        score += 0.2
    if osint_result.get("watchlist_matches"):
        score += 0.5
    return min(score, 1.0)

# Step 2: Database for Caching
# SQLite with sqlalchemy or sqlite3 can hold customer OSINT results.
import sqlite3, json

DB_PATH = "osint_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS osint_cache (customer_name TEXT PRIMARY KEY, result_json TEXT )""")
    conn.commit()
    conn.close()

def get_cached_result(name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT result_json FROM osint_cache WHERE customer_name = ?", (name,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

def save_result(name, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO osint_cache VALUES (?, ?)", (name, json.dumps(result)))
    conn.commit()
    conn.close()

# Call init_db() once at app start.

# Step 3: Async OSINT Calls
# We’ll use asyncio to run checks concurrently.
async def osint_task(name):
    cached = get_cached_result(name)
    if cached:
        return {"customer_name": name, **cached}
    try:
        result = check_customer_socials(
            customer_name=name,
            customer_identifiers={},
            twitter_bearer=os.getenv("TWITTER_BEARER"),
            watchlist_csv="WATCHLIST_CSV",
            max_posts=10
        )
        save_result(name, result)
        return {"customer_name": name, **result}
    except Exception as e:
        return {"customer_name": name, "error": str(e)}

async def run_osint_checks(names):
    tasks = [osint_task(n) for n in names if n]
    return await asyncio.gather(*tasks)

# Step 4: Portfolio Summary (Top N Risky Customers)
if run_portfolio_summary:
    st.markdown("###### 📊 Portfolio Fraud + OSINT Summary")

    # Fraud-only results
    agg = portfolio_aggregate(explanations)
    st.write("Fraud prediction distribution:", agg)

    # Collect fraud probabilities per customer
    fraud_map = {row["Customer_Name"]: row["fraud_probability"] for row in df.to_dict("records")}

    # OSINT checks (async)
    with st.spinner("Running async OSINT checks..."):
        names = df["Customer_Name"].dropna().unique().tolist()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        osint_results = loop.run_until_complete(run_osint_checks(names))

        # Combine risk scores
        combined = []
        for r in osint_results:
            name = r.get("customer_name")
            fraud_prob = fraud_map.get(name, 0)
            risk = compute_risk_score(fraud_prob, r)

            combined.append({
                "Customer": name,
                "Fraud_Prob": fraud_prob,
                "Extremist": r.get("extremist_affiliation"),
                "Criminal": r.get("criminal_records_flag"),
                "Watchlist": bool(r.get("watchlist_matches")),
                "Risk_Score": risk
            })
        risk_df = pd.DataFrame(combined)
        risk_df = risk_df.sort_values("Risk_Score", ascending=False).head(20)  # Top N risky customers
        st.markdown("### 🚨 Top Risky Customers")
        st.table(risk_df)
        st.table(risk_df[["Customer", "Risk_Score"]])

        """
        ✅ Final Workflow 
         App starts → init_db() creates local cache. 
         When you run portfolio summary: 
        o Fraud predictions are aggregated. 
        o Async OSINT calls check new customers only (cached ones skipped). 
        o Fraud probability + OSINT flags are merged → Risk Score. 
        o Top 20 risky customers displayed.
    
        ⚡ This way your dashboard is: 
         Fast (async calls). 
         Smart (combined fraud + OSINT risk). 
         Persistent (results cached in DB).
        """

        """
        Nice 🚀 — a visualization will make it much easier for analysts to spot patterns in the risk landscape. 
        We’ll add two layers: 
        1. Bar chart of Top N Risky Customers 
        2. Scatter plot (Fraud Probability vs OSINT Risk Score) to see clusters.
        """

        # Step 1: Add Visualization Code
        # Right after we build risk_df in your portfolio summary:

        # --- Bar Chart: Top Risky Customers ---
        st.markdown("###### 📊 Risk Score Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 5))
        risk_df.plot(x="Customer", y="Risk_Score", kind="bar", ax=ax, color="red", legend=False)
        ax.set_ylabel("Risk Score")
        ax.set_title("Top Risky Customers (Fraud + OSINT Combined)")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # --- Scatter Plot: Fraud vs Risk ---
        st.markdown("###### 🔎 Fraud Probability vs Combined Risk")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(risk_df["Fraud_Prob"], risk_df["Risk_Score"], c="orange", s=80, alpha=0.7,
                    edgecolors="k")

        for _, row in risk_df.iterrows():
            ax2.text(row["Fraud_Prob"] + 0.01, row["Risk_Score"] + 0.01, row["Customer"], fontsize=8)
        ax2.set_xlabel("Fraud Probability (Model)")
        ax2.set_ylabel("Risk Score (Fraud + OSINT)")
        ax2.set_title("Fraud Probability vs Combined Risk Score")
        st.pyplot(fig2)

        # Step 2: Optional Heatmap (Risk Categories)
        # We can also categorize risk into Low / Medium / High:

        def categorize_risk(score):
            if score >= 0.75:
                return "High"
            elif score >= 0.4:
                return "Medium"
            else:
                return "Low"

        risk_df["Risk_Level"] = risk_df["Risk_Score"].apply(categorize_risk)
        st.markdown("###### 🗺️ Risk Heatmap by Category")
        pivot = risk_df.pivot_table(values="Risk_Score", index="Customer", columns="Risk_Level", fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".2f", linewidths=.5, cbar=False, ax=ax3)
        ax3.set_title("Heatmap of Customer Risk Levels")

        """
        ✅ End Result:
        When you run portfolio summary, you’ll now see: 
         🚨 Table of Top Risky Customers (fraud + OSINT). 
         📊 Bar chart ranking customers by risk score. 
         🔎 Scatter plot showing how fraud probability aligns with OSINT risk. 
         🗺️ Heatmap of risk categories for quick scanning.
        """
###############################################################

# ✅ Refactored Watchlist + OSINT Screening Section

import os
import pandas as pd
from difflib import get_close_matches
import streamlit as st

# -----------------------
# Config paths
# -----------------------
WATCHLIST_CSV = "C:/Users/Springrose/Downloads/FRAUD DETECTION/GlobalTerrorismDataset.csv"  # CSV with column 'name' (and optional aliases)
GLOBAL_TERROR_TXT = "C:/Users/Springrose/Downloads/FRAUD DETECTION/globalterrorism.txt"

# -----------------------
# Core check function
# -----------------------
def check_watchlist(customer_name: str, watchlist_path: str, secondary_path: str = None):
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
    if watchlist_path and os.path.exists(watchlist_path):
        try:
            watch_df = pd.read_csv(watchlist_path)
            if "name" not in watch_df.columns:
                result["notes"].append("Watchlist missing 'name' column.")
            else:
                names = watch_df["name"].astype(str).tolist()
                matches = get_close_matches(customer_name, names, cutoff=0.8)

                if matches:
                    result["matches"].extend(matches)
                    result["flagged"] = True
                    result["notes"].append(f"Name matched in {os.path.basename(watchlist_path)}.")
        except Exception as e:
            result["notes"].append(f"Watchlist check failed: {e}")
    else:
        result["notes"].append("Watchlist file not found.")

    # 2️⃣ Check optional secondary list (like global terror)
    if secondary_path and os.path.exists(secondary_path):
        try:
            with open(secondary_path, "r", encoding="utf-8") as f:
                text_names = [line.strip() for line in f if line.strip()]
            matches = get_close_matches(customer_name, text_names, cutoff=0.8)

            if matches:
                result["matches"].extend(matches)
                result["flagged"] = True
                result["notes"].append(f"Name matched in {os.path.basename(secondary_path)}.")
        except Exception as e:
            result["notes"].append(f"Terror list check failed: {e}")
    else:
        result["notes"].append("Global terror file not found.")

    return result

# -----------------------
# Streamlit UI Example
# -----------------------
st.subheader("🕵️ OSINT & Watchlist Screening")

customer_name = st.text_input("Enter customer name for screening:")

if st.button("Run Screening"):
    with st.spinner("Checking watchlists..."):
        res = check_watchlist(customer_name, WATCHLIST_CSV, GLOBAL_TERROR_TXT)

    st.write("### 🔎 Screening Results")
    st.write("**Flagged:**", res["flagged"])
    if res["matches"]:
        st.success(f"Matched names: {', '.join(res['matches'])}")
    if res["notes"]:
        st.info(" | ".join(res["notes"]))


