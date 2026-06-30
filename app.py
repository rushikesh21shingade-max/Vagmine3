"""
VAGMINÉ — Luxury Beauty Personalization
Survey Analytics & Predictive Insights Dashboard
(single-file build: all data processing & ML utilities are included directly
below to avoid any import / folder-structure issues on Streamlit Cloud)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

# =============================================================================
# DATA PROCESSING (formerly utils/data_processing.py)
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_THIS_DIR, "data", "VAGMINE_Synthetic_Survey_Data.xlsx")

AGE_ORDER = ["18–24", "25–30", "31–35", "36–40"]
RESEARCH_ORDER = ["<10 minutes", "10–30 minutes", "30–60 minutes", ">1 hour"]

SUSTAINABILITY_COLS = ["CrueltyFree", "Vegan", "Ethical", "Refillable", "Recyclable", "PaySustainable"]
MEMBERSHIP_PERK_COLS = ["VIPLaunches", "VIPRewards", "RoutineUpdates", "Consultations",
                         "EarlyAccess", "SubscriptionSavings"]
DNA_SHARE_COLS = ["ShareSkinType", "ShareSkinTone", "ShareLifestyle", "ShareClimate",
                   "ShareGoals", "ShareBudget", "ShareIngredients"]
DIGITAL_FEATURE_COLS = ["AIAdvisor", "VirtualConsult", "SkinAnalysis", "RoutineTracking",
                         "Dashboard", "ShadeMatching"]
LAUNCH_COLS = ["LaunchCleanser", "LaunchSerum", "LaunchMoisturizer", "LaunchSunscreen",
               "LaunchFoundation", "LaunchLipstick"]
PRIORITY_COLS = ["PriorityPerformance", "PriorityPersonalization", "PriorityIngredients",
                  "PriorityPackaging", "PriorityPrice"]

SEGMENT_FEATURES = [
    "Income", "Confidence", "BeautyDNAInterest", "TrustPersonalization",
    "PurchaseIntent", "Loyalty", "PayPremium", "PaySustainable",
    "Membership", "LuxuryPackaging",
]

SEGMENT_NAMES = {
    0: "Aspirational Explorers",
    1: "Tech-Savvy Luxury Devotees",
    2: "Value-Conscious Pragmatists",
    3: "Sustainability-Driven Loyalists",
}


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived / composite features used throughout the dashboard."""
    df = df.copy()

    # Ordered categorical helpers
    df["AgeOrder"] = df["Age"].map({a: i for i, a in enumerate(AGE_ORDER)})
    df["ResearchTimeOrder"] = df["ResearchTime"].map({a: i for i, a in enumerate(RESEARCH_ORDER)})

    # Composite indices (mean of related Likert / binary items)
    df["SustainabilityIndex"] = df[SUSTAINABILITY_COLS].mean(axis=1)
    df["MembershipPerkIndex"] = df[MEMBERSHIP_PERK_COLS].mean(axis=1)
    df["DataSharingIndex"] = df[DNA_SHARE_COLS].mean(axis=1)
    df["DigitalFeatureIndex"] = df[DIGITAL_FEATURE_COLS].mean(axis=1)
    df["LaunchInterestIndex"] = df[LAUNCH_COLS].mean(axis=1)
    df["PricePriorityShare"] = df["PriorityPrice"] / df[PRIORITY_COLS].sum(axis=1)

    # Binary / labelled targets for diagnostics & ML
    df["HighPurchaseIntent"] = (df["PurchaseIntent"] >= 4).astype(int)
    df["HighBeautyDNAInterest"] = (df["BeautyDNAInterest"] >= 4).astype(int)
    df["HighLoyalty"] = (df["Loyalty"] >= 4).astype(int)
    df["PromoterNPS"] = (df["NPS"] >= 9).astype(int)

    df["PurchaseIntentLabel"] = df["PurchaseIntent"].map({3: "Moderate (3)", 4: "High (4)", 5: "Very High (5)"})
    df["MembershipLabel"] = df["Membership"].map({0: "Not Interested", 1: "Interested"})
    df["PayPremiumLabel"] = df["PayPremium"].map({0: "Not Willing", 1: "Willing"})
    df["BeautyDNALabel"] = df["BeautyDNAInterest"].map({3: "Moderate (3)", 4: "High (4)", 5: "Very High (5)"})
    df["IncomeLabel"] = df["Income"].map({1: "Q1 - Lowest", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 - Highest"})

    return df


def add_segments(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """KMeans-based behavioural customer segmentation."""
    df = df.copy()
    X = df[SEGMENT_FEATURES].astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = km.fit_predict(Xs)

    # Rank clusters by a composite "value score" (income, intent, premium pricing, membership)
    centers = pd.DataFrame(km.cluster_centers_, columns=SEGMENT_FEATURES)
    value_score = centers[["Income", "PurchaseIntent", "PayPremium", "Membership"]].mean(axis=1)
    ranked = value_score.sort_values(ascending=False).index.tolist()

    top_cluster = ranked[0]           # highest value -> Tech-Savvy Luxury Devotees
    bottom_cluster = ranked[-1]       # lowest value -> Value-Conscious Pragmatists
    middle_clusters = ranked[1:-1]    # disambiguate by sustainability orientation

    middle_sorted = sorted(middle_clusters, key=lambda c: centers.loc[c, "PaySustainable"], reverse=True)
    sustain_cluster = middle_sorted[0] if middle_sorted else None
    aspirational_cluster = middle_sorted[1] if len(middle_sorted) > 1 else None

    remap = {top_cluster: 1}                      # Tech-Savvy Luxury Devotees
    if sustain_cluster is not None:
        remap[sustain_cluster] = 3                # Sustainability-Driven Loyalists
    if aspirational_cluster is not None:
        remap[aspirational_cluster] = 0           # Aspirational Explorers
    remap[bottom_cluster] = 2                     # Value-Conscious Pragmatists

    df["SegmentID"] = [remap[c] for c in raw_labels]
    df["Segment"] = df["SegmentID"].map(SEGMENT_NAMES)
    return df


def get_full_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = load_raw_data(path)
    df = engineer_features(df)
    df = add_segments(df)
    return df


def segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Income", "BeautyDNAInterest", "PurchaseIntent", "Loyalty",
            "PayPremium", "PaySustainable", "Membership", "SustainabilityIndex", "NPS"]
    prof = df.groupby("Segment")[cols].mean().round(2)
    prof["Count"] = df["Segment"].value_counts()
    return prof.sort_values("Count", ascending=False)

# =============================================================================
# MACHINE LEARNING (formerly utils/ml_models.py)
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)

CATEGORICAL_FEATURES = ["Country", "Occupation", "ResearchTime", "TrustSource", "Bundle", "Segment"]
NUMERIC_FEATURES = [
    "AgeOrder", "Income", "Confidence", "Overwhelmed", "BeautyDNAInterest",
    "TrustPersonalization", "ImproveRecs", "SustainabilityIndex", "MembershipPerkIndex",
    "DataSharingIndex", "DigitalFeatureIndex", "LaunchInterestIndex", "PricePriorityShare",
    "Loyalty", "NPS", "LuxuryPackaging",
]

TARGET_OPTIONS = {
    "High Purchase Intention (PurchaseIntent >= 4)": "HighPurchaseIntent",
    "Membership Interest": "Membership",
    "Premium Price Willingness": "PayPremium",
    "High Beauty DNA™ Interest (>= 4)": "HighBeautyDNAInterest",
    "High Loyalty (>= 4)": "HighLoyalty",
}


def _feature_target_split(df: pd.DataFrame, target_col: str):
    drop_cols = set(["HighPurchaseIntent", "Membership", "PayPremium",
                      "HighBeautyDNAInterest", "HighLoyalty", "PurchaseIntent",
                      "BeautyDNAInterest", "Loyalty"])
    num_feats = [c for c in NUMERIC_FEATURES if c not in drop_cols]
    cat_feats = [c for c in CATEGORICAL_FEATURES]
    X = df[num_feats + cat_feats].copy()
    y = df[target_col].astype(int)
    return X, y, num_feats, cat_feats


def build_preprocessor(num_feats, cat_feats):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )


MODEL_ZOO = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
}


def train_and_evaluate(df: pd.DataFrame, target_col: str, test_size: float = 0.25, random_state: int = 42):
    """Train all four classifiers on the chosen target and return a results dict."""
    X, y, num_feats, cat_feats = _feature_target_split(df, target_col)
    preproc = build_preprocessor(num_feats, cat_feats)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    results = {}
    for name, base_model in MODEL_ZOO.items():
        pipe = Pipeline([("prep", preproc), ("model", base_model)])
        pipe.fit(X_train, y_train)

        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            test_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            test_proba = test_pred.astype(float)

        fpr, tpr, _ = roc_curve(y_test, test_proba)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, test_pred)

        results[name] = {
            "pipeline": pipe,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, zero_division=0),
            "recall": recall_score(y_test, test_pred, zero_division=0),
            "f1": f1_score(y_test, test_pred, zero_division=0),
            "roc_auc": roc_auc,
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "y_test": y_test,
            "test_proba": test_proba,
        }

    # Feature importance (tree-based models) for diagnostic narrative
    try:
        rf_pipe = results["Random Forest"]["pipeline"]
        ohe_names = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_feats)
        all_names = list(num_feats) + list(ohe_names)
        importances = rf_pipe.named_steps["model"].feature_importances_
        fi = pd.Series(importances, index=all_names).sort_values(ascending=False)
        results["_feature_importance"] = fi
    except Exception:
        results["_feature_importance"] = pd.Series(dtype=float)

    results["_class_balance"] = y.value_counts(normalize=True).round(3).to_dict()
    return results


def metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        if name.startswith("_"):
            continue
        rows.append({
            "Model": name,
            "Train Accuracy": r["train_accuracy"],
            "Test Accuracy": r["test_accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "AUC": r["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

# =============================================================================
# STREAMLIT APP
# =============================================================================
"""
VAGMINÉ — Luxury Beauty Personalization
Survey Analytics & Predictive Insights Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="VAGMINÉ | Beauty DNA Analytics",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOLD = "#A8763E"
DARK = "#2B2420"
CREAM = "#F8F3EE"
PALETTE = ["#A8763E", "#5C4742", "#D9A86C", "#7A8B7A", "#C2876B", "#3F4B44"]

st.markdown(f"""
<style>
    .main {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {DARK}; font-family: 'Georgia', serif; letter-spacing: 0.5px; }}
    .vag-kpi {{
        background: linear-gradient(135deg, {CREAM} 0%, #FFFFFF 100%);
        border: 1px solid #EAE0D5;
        border-left: 4px solid {GOLD};
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }}
    .vag-kpi h2 {{ margin: 0; color: {GOLD}; font-size: 2rem; }}
    .vag-kpi p {{ margin: 0; color: {DARK}; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;}}
    .vag-banner {{
        background: linear-gradient(120deg, {DARK} 0%, #4A3B33 100%);
        color: #F8F3EE; padding: 28px 32px; border-radius: 12px; margin-bottom: 18px;
    }}
    .vag-banner h1 {{ color: #F8F3EE; margin-bottom: 4px; }}
    .vag-banner p {{ color: #D9C9BB; margin: 0; font-size: 1.0rem; }}
    .insight-box {{
        background-color: {CREAM}; border-radius: 10px; padding: 14px 18px;
        border-left: 4px solid {GOLD}; margin: 8px 0;
    }}
    [data-testid="stMetricValue"] {{ color: {GOLD}; }}
</style>
""", unsafe_allow_html=True)

px.defaults.color_discrete_sequence = PALETTE


@st.cache_data(show_spinner="Loading & engineering survey data...")
def load_data():
    return get_full_dataset()


@st.cache_resource(show_spinner="Training classification models...")
def run_ml(df, target_col):
    return train_and_evaluate(df, target_col)


df = load_data()

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
st.sidebar.markdown("## ✨ VAGMINÉ")
st.sidebar.caption("Beauty DNA™ Survey Intelligence Suite")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Executive Overview", "📊 Descriptive Analytics", "🔍 Diagnostic Analytics",
     "🧬 Customer Segmentation", "🤖 Predictive Modelling", "💡 Insights & Recommendations"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Respondents")
country_f = st.sidebar.multiselect("Country", sorted(df["Country"].unique()), default=list(df["Country"].unique()))
age_f = st.sidebar.multiselect("Age Group", sorted(df["Age"].unique()), default=list(df["Age"].unique()))
income_f = st.sidebar.multiselect("Income Quintile", sorted(df["IncomeLabel"].unique()), default=list(df["IncomeLabel"].unique()))

fdf = df[df["Country"].isin(country_f) & df["Age"].isin(age_f) & df["IncomeLabel"].isin(income_f)]
st.sidebar.markdown(f"**Filtered respondents:** {len(fdf)} / {len(df)}")
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic survey data · For research & dissertation use")

if len(fdf) < 20:
    st.warning("Very few respondents match the current filters — insights below may be unstable.")

# ----------------------------------------------------------------------------
# PAGE 1 — EXECUTIVE OVERVIEW
# ----------------------------------------------------------------------------
if page == "🏠 Executive Overview":
    st.markdown("""
    <div class="vag-banner">
        <h1>VAGMINÉ Beauty DNA™ Analytics</h1>
        <p>Validating the luxury beauty personalization business model through customer survey intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, f"{len(fdf)}", "Survey Respondents"),
        (k2, f"{fdf['HighPurchaseIntent'].mean()*100:.1f}%", "High Purchase Intent"),
        (k3, f"{fdf['HighBeautyDNAInterest'].mean()*100:.1f}%", "Beauty DNA™ Interest (High)"),
        (k4, f"{fdf['Membership'].mean()*100:.1f}%", "Membership Interest"),
        (k5, f"{fdf['PayPremium'].mean()*100:.1f}%", "Willing to Pay Premium"),
    ]
    for col, val, label in kpis:
        col.markdown(f'<div class="vag-kpi"><h2>{val}</h2><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("####")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        seg_counts = fdf["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig = px.bar(seg_counts, x="Count", y="Segment", orientation="h", text="Count",
                     title="Customer Segments by Size", color="Segment")
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Respondents")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        nps_avg = fdf["NPS"].mean()
        promoters = (fdf["NPS"] >= 9).mean() * 100
        detractors = (fdf["NPS"] <= 6).mean() * 100
        nps_score = promoters - detractors
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=nps_score,
            title={"text": "Estimated NPS"},
            gauge={"axis": {"range": [-100, 100]},
                   "bar": {"color": GOLD},
                   "steps": [{"range": [-100, 0], "color": "#F2E2DA"},
                             {"range": [0, 50], "color": "#EFE2C8"},
                             {"range": [50, 100], "color": "#DCE8D2"}]},
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Why This Matters")
    st.markdown(f"""
    <div class="insight-box">
    Across <b>{len(fdf)}</b> surveyed customers, <b>{fdf['HighPurchaseIntent'].mean()*100:.0f}%</b> express high purchase intent
    toward VAGMINÉ's personalized offering, and <b>{fdf['HighBeautyDNAInterest'].mean()*100:.0f}%</b> show strong interest in the
    Beauty DNA™ personalization concept — early validation that personalization-led luxury beauty resonates with this audience.
    Use the navigation panel to explore descriptive, diagnostic, segmentation, and predictive views in depth.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE 2 — DESCRIPTIVE ANALYTICS
# ----------------------------------------------------------------------------
elif page == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    st.caption("Profiling respondents and core survey variables")

    tabs = st.tabs(["Demographics", "Purchase & DNA Interest", "Membership & Pricing",
                     "Sustainability", "Product Preferences", "Cross-Tabulations"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.pie(fdf, names="Age", title="Age Distribution", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(fdf, names="Country", title="Country Distribution", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        with c3:
            fig = px.pie(fdf, names="Occupation", title="Occupation Distribution", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(fdf, x="IncomeLabel", color="Occupation", barmode="group",
                            title="Income Quintile by Occupation",
                            category_orders={"IncomeLabel": ["Q1 - Lowest", "Q2", "Q3", "Q4", "Q5 - Highest"]})
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(fdf, x="PurchaseIntentLabel", color="PurchaseIntentLabel",
                                title="Purchase Intention Distribution",
                                category_orders={"PurchaseIntentLabel": ["Moderate (3)", "High (4)", "Very High (5)"]})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(fdf, x="BeautyDNALabel", color="BeautyDNALabel",
                                title="Beauty DNA™ Interest Distribution",
                                category_orders={"BeautyDNALabel": ["Moderate (3)", "High (4)", "Very High (5)"]})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        share_means = fdf[["ShareSkinType", "ShareSkinTone", "ShareLifestyle", "ShareClimate",
                            "ShareGoals", "ShareBudget", "ShareIngredients"]].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=share_means.values, y=share_means.index, orientation="h",
                     title="Willingness to Share Data for Personalization (%)", text=share_means.round(1))
        fig.update_layout(yaxis_title="", xaxis_title="% Willing to Share")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(fdf, names="MembershipLabel", title="Membership Interest", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(fdf, names="PayPremiumLabel", title="Premium Price Willingness", hole=0.45)
            st.plotly_chart(fig, use_container_width=True)
        perk_means = fdf[["VIPLaunches", "VIPRewards", "RoutineUpdates", "Consultations",
                           "EarlyAccess", "SubscriptionSavings"]].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=perk_means.values, y=perk_means.index, orientation="h",
                     title="Most Appealing Membership Perks (%)", text=perk_means.round(1))
        fig.update_layout(yaxis_title="", xaxis_title="% Interested")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        c1, c2 = st.columns(2)
        sustain_means = fdf[["CrueltyFree", "Vegan", "Ethical", "Refillable", "Recyclable", "PaySustainable"]].mean().sort_values(ascending=False) * 100
        with c1:
            fig = px.bar(x=sustain_means.values, y=sustain_means.index, orientation="h",
                         title="Sustainability Preferences (%)", text=sustain_means.round(1))
            fig.update_layout(yaxis_title="", xaxis_title="% Important / Willing")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(fdf, x="LuxuryPackaging", title="Importance of Luxury Packaging",
                                color="LuxuryPackaging")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(fdf, names="Bundle", title="Preferred Routine Bundle")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(fdf, names="Feature", title="Most Desired App Feature")
            st.plotly_chart(fig, use_container_width=True)
        launch_means = fdf[["LaunchCleanser", "LaunchSerum", "LaunchMoisturizer",
                             "LaunchSunscreen", "LaunchFoundation", "LaunchLipstick"]].mean().sort_values(ascending=False) * 100
        fig = px.bar(x=launch_means.values, y=launch_means.index, orientation="h",
                     title="Product Launch Priority (%)", text=launch_means.round(1))
        fig.update_layout(yaxis_title="", xaxis_title="% Prioritising")
        st.plotly_chart(fig, use_container_width=True)
        priority_means = fdf[PRIORITY_COLS].mean().sort_values(ascending=False)
        fig = px.bar(x=priority_means.values, y=priority_means.index, orientation="h",
                     title="Average Priority Ranking (1-5) of Decision Factors")
        fig.update_layout(yaxis_title="", xaxis_title="Average Score")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.markdown("##### Build a Cross-Tabulation")
        cat_vars = {
            "Purchase Intention": "PurchaseIntentLabel", "Beauty DNA™ Interest": "BeautyDNALabel",
            "Membership Interest": "MembershipLabel", "Premium Pricing": "PayPremiumLabel",
            "Customer Segment": "Segment", "Age Group": "Age", "Country": "Country",
            "Income Quintile": "IncomeLabel", "Preferred Bundle": "Bundle",
        }
        c1, c2 = st.columns(2)
        row_choice = c1.selectbox("Row variable", list(cat_vars.keys()), index=4)
        col_choice = c2.selectbox("Column variable", list(cat_vars.keys()), index=0)
        rowv, colv = cat_vars[row_choice], cat_vars[col_choice]

        ctab = pd.crosstab(fdf[rowv], fdf[colv])
        ctab_pct = pd.crosstab(fdf[rowv], fdf[colv], normalize="index") * 100

        st.markdown(f"**{row_choice} × {col_choice} — Counts**")
        st.dataframe(ctab, use_container_width=True)
        fig = px.imshow(ctab_pct.round(1), text_auto=True, color_continuous_scale="Oranges",
                         title=f"{row_choice} × {col_choice} (Row %)", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE 3 — DIAGNOSTIC ANALYTICS
# ----------------------------------------------------------------------------
elif page == "🔍 Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")
    st.caption("What drives Purchase Intention, Beauty DNA™ adoption, Membership, Loyalty & Premium Pricing?")

    outcome_map = {
        "Purchase Intention": "PurchaseIntent", "Beauty DNA™ Interest": "BeautyDNAInterest",
        "Membership Interest": "Membership", "Loyalty": "Loyalty", "Premium Price Willingness": "PayPremium",
    }
    driver_map = {
        "Age Group": "Age", "Income Quintile": "IncomeLabel", "Country": "Country",
        "Occupation": "Occupation", "Customer Segment": "Segment", "Sustainability Index": "SustainabilityIndex",
        "Research Time": "ResearchTime", "Trust Source": "TrustSource",
    }

    c1, c2 = st.columns(2)
    outcome_choice = c1.selectbox("Outcome Variable", list(outcome_map.keys()))
    driver_choice = c2.selectbox("Candidate Driver", list(driver_map.keys()))
    outcome_col, driver_col = outcome_map[outcome_choice], driver_map[driver_choice]

    if pd.api.types.is_numeric_dtype(fdf[driver_col]) and fdf[driver_col].nunique() > 6:
        corr = fdf[[outcome_col, driver_col]].corr().iloc[0, 1]
        fig = px.scatter(fdf, x=driver_col, y=outcome_col, trendline="ols",
                          title=f"{outcome_choice} vs {driver_choice} (r = {corr:.2f})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        means = fdf.groupby(driver_col)[outcome_col].mean().sort_values(ascending=False)
        overall = fdf[outcome_col].mean()
        fig = px.bar(x=means.index.astype(str), y=means.values,
                     title=f"Average {outcome_choice} by {driver_choice}",
                     labels={"x": driver_choice, "y": f"Average {outcome_choice}"})
        fig.add_hline(y=overall, line_dash="dash", line_color=GOLD,
                       annotation_text="Overall average")
        st.plotly_chart(fig, use_container_width=True)
        spread = means.max() - means.min()
        st.markdown(f"""
        <div class="insight-box">
        The gap between the highest- and lowest-scoring {driver_choice.lower()} group is
        <b>{spread:.2f} points</b> on {outcome_choice.lower()}. The
        <b>{means.idxmax()}</b> group shows the strongest association with {outcome_choice.lower()},
        while <b>{means.idxmin()}</b> shows the weakest — a useful lens for targeted messaging.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Correlation Heatmap — Numeric Behavioural Drivers")
    corr_cols = ["Income", "Confidence", "BeautyDNAInterest", "TrustPersonalization",
                 "SustainabilityIndex", "MembershipPerkIndex", "DataSharingIndex",
                 "DigitalFeatureIndex", "LuxuryPackaging", "PurchaseIntent", "Loyalty",
                 "PayPremium", "Membership", "NPS"]
    corr_matrix = fdf[corr_cols].corr().round(2)
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                     title="Correlation Matrix of Key Behavioural & Outcome Variables", aspect="auto")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    top_corr = corr_matrix["PurchaseIntent"].drop("PurchaseIntent").abs().sort_values(ascending=False).head(3)
    st.markdown(f"""
    <div class="insight-box">
    The variables most strongly associated with <b>Purchase Intention</b> are
    <b>{', '.join(top_corr.index)}</b> (|r| = {', '.join(top_corr.round(2).astype(str))}).
    This suggests product, pricing and CRM strategy should prioritise levers tied to these drivers.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE 4 — CUSTOMER SEGMENTATION
# ----------------------------------------------------------------------------
elif page == "🧬 Customer Segmentation":
    st.title("🧬 Customer Segmentation")
    st.caption("Behavioural clustering (K-Means) across income, personalization interest, loyalty & pricing signals")

    prof = segment_profile(fdf)
    st.markdown("##### Segment Profile (Average Scores)")
    st.dataframe(prof.style.background_gradient(cmap="Oranges", subset=[c for c in prof.columns if c != "Count"]),
                 use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(fdf, x="Income", y="PurchaseIntent", color="Segment", size="PayPremium",
                          hover_data=["Age", "Country", "Membership"],
                          title="Segments: Income vs Purchase Intent")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        radar_cols = ["Income", "BeautyDNAInterest", "PurchaseIntent", "Loyalty", "PayPremium", "SustainabilityIndex"]
        radar_df = fdf.groupby("Segment")[radar_cols].mean()
        radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        fig = go.Figure()
        for seg in radar_norm.index:
            fig.add_trace(go.Scatterpolar(r=radar_norm.loc[seg].values, theta=radar_cols,
                                           fill="toself", name=seg))
        fig.update_layout(title="Normalised Segment Profiles (Radar)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Segment Composition by Country & Income")
    fig = px.sunburst(fdf, path=["Segment", "Country", "IncomeLabel"], title="Segment → Country → Income Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Narratives")
    descriptions = {
        "Aspirational Explorers": "Curious, value-aware customers who are intrigued by Beauty DNA™ but need reassurance on price and trust before committing.",
        "Tech-Savvy Luxury Devotees": "High-income, high-intent customers enthusiastic about AI personalization, digital features and premium pricing — VAGMINÉ's primary launch segment.",
        "Value-Conscious Pragmatists": "Price-sensitive customers who like the concept but require clear ROI, bundles, and entry-level offers.",
        "Sustainability-Driven Loyalists": "Customers strongly motivated by ethical/sustainable credentials, loyal once values are demonstrated, often advocate via word of mouth.",
    }
    for seg in prof.index:
        st.markdown(f"""
        <div class="insight-box"><b>{seg}</b> ({int(prof.loc[seg, 'Count'])} respondents) — {descriptions.get(seg, '')}</div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE 5 — PREDICTIVE MODELLING
# ----------------------------------------------------------------------------
elif page == "🤖 Predictive Modelling":
    st.title("🤖 Predictive Modelling")
    st.caption("KNN · Decision Tree · Random Forest · Gradient Boosting — classifier comparison")

    target_label = st.selectbox("Choose business outcome to predict", list(TARGET_OPTIONS.keys()))
    target_col = TARGET_OPTIONS[target_label]

    with st.spinner("Training models..."):
        results = run_ml(df, target_col)

    mt = metrics_table(results)
    st.markdown("### Model Performance Comparison")
    st.dataframe(mt.style.background_gradient(cmap="Oranges", subset=["Test Accuracy", "Precision", "Recall", "F1-Score", "AUC"]),
                 use_container_width=True)

    best_model_name = mt.iloc[0]["Model"]
    st.success(f"🏆 Best performing model for **{target_label}**: **{best_model_name}** "
               f"(Test Accuracy {mt.iloc[0]['Test Accuracy']:.2%}, AUC {mt.iloc[0]['AUC']:.2f})")

    c1, c2 = st.columns(2)
    with c1:
        metric_long = mt.melt(id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy"],
                               var_name="Split", value_name="Accuracy")
        fig = px.bar(metric_long, x="Model", y="Accuracy", color="Split", barmode="group",
                     title="Training vs Testing Accuracy (Stability Check)")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        metric_long2 = mt.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-Score"],
                                var_name="Metric", value_name="Score")
        fig = px.bar(metric_long2, x="Model", y="Score", color="Metric", barmode="group",
                     title="Precision · Recall · F1-Score by Model")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### ROC Curves")
    fig = go.Figure()
    for name, r in results.items():
        if name.startswith("_"):
            continue
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines",
                                  name=f"{name} (AUC={r['roc_auc']:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Confusion Matrices")
    cols = st.columns(4)
    for i, (name, r) in enumerate([(k, v) for k, v in results.items() if not k.startswith("_")]):
        with cols[i % 4]:
            cm = r["confusion_matrix"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Oranges",
                             labels=dict(x="Predicted", y="Actual"),
                             x=["Negative", "Positive"], y=["Negative", "Positive"],
                             title=name)
            fig.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)

    fi = results.get("_feature_importance")
    if fi is not None and len(fi) > 0:
        st.markdown("### Top Feature Importances (Random Forest)")
        fig = px.bar(x=fi.head(12).values[::-1], y=fi.head(12).index[::-1], orientation="h",
                     title="Most Influential Features")
        fig.update_layout(yaxis_title="", xaxis_title="Importance")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    Across all four algorithms, model performance for predicting <b>{target_label}</b> indicates
    {"strong and stable" if mt.iloc[0]['Test Accuracy'] - mt.iloc[0]['Train Accuracy'] > -0.15 else "some overfitting in"}
    generalisation. The <b>{best_model_name}</b> model offers the best balance of accuracy, precision and recall,
    making it the recommended engine for a production propensity-scoring model that prioritises high-value leads
    for VAGMINÉ's sales and CRM teams.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE 6 — INSIGHTS & RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif page == "💡 Insights & Recommendations":
    st.title("💡 Insights & Strategic Recommendations")

    st.markdown("### Key Findings")
    findings = [
        f"**{fdf['HighPurchaseIntent'].mean()*100:.0f}%** of respondents report high purchase intent, validating demand for a personalized luxury beauty proposition.",
        f"**{fdf['HighBeautyDNAInterest'].mean()*100:.0f}%** show strong interest in the Beauty DNA™ concept, with respondents most willing to share skin type, skin tone and goals data — but more hesitant on budget and lifestyle data.",
        f"**{fdf['Membership'].mean()*100:.0f}%** are interested in a membership programme, with early access and VIP rewards as the most appealing perks.",
        f"**{fdf['PayPremium'].mean()*100:.0f}%** are willing to pay a premium for personalization, suggesting room for tiered, value-based pricing rather than mass-market pricing.",
        f"Sustainability credentials (cruelty-free, vegan, refillable) resonate strongly and should be embedded into the brand promise, not treated as a niche add-on.",
        "Customer segmentation reveals a clear high-value core (Tech-Savvy Luxury Devotees) alongside price-sensitive and sustainability-driven segments — a one-size-fits-all GTM strategy would under-serve at least two of the four segments.",
    ]
    for f in findings:
        st.markdown(f"- {f}")

    st.markdown("### Strategic Recommendations")
    rec_cols = st.columns(2)
    with rec_cols[0]:
        st.markdown("""
        **Product Development**
        - Prioritise launching cleanser/serum personalization first (highest launch interest), then expand to color cosmetics.
        - Build the Beauty DNA™ quiz around low-friction data (skin type/tone/goals) before requesting sensitive inputs (budget).
        - Invest in AI-driven shade matching and skin analysis — the most desired digital features.

        **Pricing Strategy**
        - Introduce a tiered model: an accessible entry tier for Value-Conscious Pragmatists and a premium Beauty DNA™ + concierge tier for Tech-Savvy Devotees.
        - Frame premium pricing around personalization depth and sustainability credentials, both shown to correlate with willingness to pay.
        """)
    with rec_cols[1]:
        st.markdown("""
        **Customer Segmentation & Marketing**
        - Lead acquisition marketing with the Tech-Savvy Luxury Devotees segment; use loyalty/VIP messaging to convert Sustainability-Driven Loyalists.
        - Use the propensity model (best classifier) to score leads and prioritise high-likelihood converters for sales outreach.

        **Future Growth**
        - Pilot a referral-driven membership programme leveraging early access and VIP rewards.
        - Track NPS and loyalty over time as personalization accuracy improves, to validate retention assumptions in the business case.
        """)

    st.markdown("---")
    st.caption("Prepared using synthetic survey data for the VAGMINÉ business model validation study. "
               "All figures should be replicated with live customer data prior to investment or scaling decisions.")
