import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
import pickle
import os

st.set_page_config(page_title="TB Treatment Outcome Predictor", layout="wide")

# -------------------------------
# Load model + feature list
# -------------------------------
# @st.cache_resource
# def load_model():
#     model = xgb.XGBClassifier()
#     model = pickle.load('xgboost_full_model_weights.pkl')
#     # model.load_model("xgboost_full_model_weights.json")
#     return model

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "xgboost_full_model_weights.pkl")
    model = joblib.load(model_path)
    return model


@st.cache_resource
def load_feature_list():
    try:
        return joblib.load("feature_list.pkl")   # column ordering used during training
    except:
        return None

model = load_model()
feature_list = load_feature_list()

# --------------------------------
# Title
# --------------------------------
st.title("🫁 TB Treatment Outcome Prediction Dashboard")
st.markdown("Upload patient-level data to obtain predicted treatment outcome and model explanations.")
st.markdown(
    """
**Citation:**  
Wang, L., Campino, S., Clark, T.G. and Phelan, J.E. (2025a). A multi-stage machine learning framework for stepwise prediction of tuberculosis treatment outcomes: Integrating gradient boosted decision trees and feature-level analysis for clinical decision support. Research Square, (Preprint). doi:https://doi.org/10.21203/rs.3.rs-7558046/v1.

**Github repo:**  
https://github.com/linfeng-wang/TBpt
"""
)
# --------------------------------
# Tabs
# --------------------------------
tab_upload, tab_shap, tab_waterfall = st.tabs(["📁 Upload & Predict", "📊 Model SHAP Summary", "🔍 Sample Waterfall Explanation"])


# ======================================================================
# TAB 1: UPLOAD & PREDICT
# ======================================================================
with tab_upload:

    uploaded = st.file_uploader("**Upload CSV or Excel file**", type=["csv", "xlsx"])

    st.markdown(
        """

    ---

    _Example input file_

    You can download an example input file below to run a test prediction.  
    This file also serves as a reference for the required data format. If you follow
    the same structure and simply replace the values with your own data, the dashboard
    will generate predictions for your dataset. \n
    _(Exact feature encoding and and leveling can be found in the publication)_
    """
    )

    # Download example input file
    # with open("example_input.csv", "rb") as f:
    #     st.download_button(
    #         label="⬇️ Download Example Input File",
    #         data=f,
    #         file_name="example_input.csv",
    #         mime="text/csv"
    #     )

    example_path = os.path.join(os.path.dirname(__file__), "example_input.csv")

    with open(example_path, "rb") as f:
        st.download_button(
            label="⬇️ Download Example Input File",
            data=f,
            file_name="example_input.csv",
            mime="text/csv"
        )
    
    if uploaded is not None:
        # Read file
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Ensure features align with training
        if feature_list is not None:
            missing = [c for c in feature_list if c not in df.columns]

            # Create missing columns filled with NaN (safe for XGBoost)
            if missing:
                st.warning(f"The following features were missing and have been added as NaN: {missing}")
                for col in missing:
                    df[col] = np.nan

            # Drop extra columns, keep only required set
            df = df.reindex(columns=feature_list)

        st.success("Data successfully validated. Ready to predict.")

        # -------------------------------
        # Predict
        # -------------------------------
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        result_df = pd.DataFrame({
            "Predicted Outcome": ["Cured" if p == 1 else "Failed" for p in preds],
            "Confidence Score": probs
        })

        st.subheader("🔮 Predictions")
        st.dataframe(result_df)

        # -------------------------------
        # Summary Statistics
        # -------------------------------
        st.subheader("📈 Summary Statistics")
        st.write(f"**Predicted cured:** { (preds==1).mean()*100:.2f}%")
        st.write(f"**Predicted failed:** { (preds==0).mean()*100:.2f}%")
        st.write(f"**Mean confidence:** { probs.mean():.3f}")

        # Histogram
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(probs, bins=20, color='skyblue')
        ax_hist.set_title("Confidence Score Distribution")
        st.pyplot(fig_hist)

        # -------------------------------
        # Download Predictions
        # -------------------------------
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv,
            file_name="treatment_outcomes_predictions.csv",
            mime="text/csv"
        )



# ======================================================================
# TAB 2: SHAP SUMMARY
# ======================================================================
with tab_shap:

    if uploaded is None:
        st.info("Upload a file first in the 'Upload & Predict' tab.")
    else:
        st.subheader("📊 SHAP Feature Importance Summary")

        # Limit rows for SHAP to prevent slowdowns
        MAX_SHAP = 200
        if len(df) > MAX_SHAP:
            st.warning(f"Dataset is large. Only first {MAX_SHAP} rows used for SHAP visualization.")
            df_shap = df.head(MAX_SHAP)
        else:
            df_shap = df

        # -------------------------------
        # Compute SHAP values (XGBoost-safe)
        # -------------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_shap)

        # -------------------------------
        # Bar plot (global importance)
        # -------------------------------
        st.write("**Mean Absolute SHAP Values (Global Importance)**")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, df_shap, plot_type="bar", show=False)
        st.pyplot(fig1)
        plt.clf()

        # -------------------------------
        # Beeswarm plot
        # -------------------------------
        st.write("**SHAP Beeswarm Plot (Full Feature Contributions)**")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, df_shap, show=False)
        st.pyplot(fig2)
        plt.clf()



# ======================================================================
# TAB 3: SHAP WATERFALL
# ======================================================================
# ======================================================================
# TAB 3: SHAP WATERFALL
# ======================================================================
with tab_waterfall:

    if uploaded is None:
        st.info("Upload a file first in the 'Upload & Predict' tab.")
    else:
        st.subheader("🔍 SHAP Waterfall Explanation")

        # Limit rows for SHAP
        MAX_SHAP = 200
        df_shap = df.head(MAX_SHAP)

        # -------------------------------
        # Compute SHAP values (XGBoost-safe)
        # -------------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_shap)
        base_value = explainer.expected_value

        # Choose index for detailed explanation
        example_idx = st.number_input(
            "Choose a sample index for a waterfall explanation:",
            min_value=0,
            max_value=len(df_shap) - 1,
            value=0
        )

        st.write(f"**SHAP Waterfall Plot for Sample {example_idx}**")

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[example_idx],
                base_values=base_value,
                data=df_shap.iloc[example_idx],
                feature_names=df_shap.columns
            ),
            show=False
        )
        st.pyplot(fig3)
        plt.clf()

