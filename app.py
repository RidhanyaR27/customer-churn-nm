import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load models and expected features
rf_model = joblib.load("model_rf.pkl")
xgb_model = joblib.load("model_xgb.pkl")
expected_features = joblib.load("model_features.pkl")

# Streamlit page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")
st.title("📊 Customer Churn Analysis & Prediction App")

# ===== File Upload Section (center) =====
st.markdown("### 📂 Upload Dataset for Analysis")
uploaded_file = st.file_uploader("Upload a cleaned CSV file with customer data", type="csv")

# ===== Model Selection in Sidebar =====
st.sidebar.header("🔍 Choose Model")
model_choice = st.sidebar.radio("Select model:", ("Random Forest", "XGBoost"))
model = rf_model if model_choice == "Random Forest" else xgb_model

# ===== Feature Settings =====
meaningful_features = [
    "age", "weekly_hours", "average_session_length",
    "skip_rate", "notifications_clicked", "account_months", "engagement_score"
]

# ===== Uploaded Dataset Handling =====
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    if "churned" not in df.columns:
        st.warning("⚠️ Please include a 'churned' column in your dataset for analysis.")
    else:
        # Only use important features for analysis
        available_features = [col for col in meaningful_features if col in df.columns]

        # Churn distribution
        st.markdown("### 📊 Churn Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="churned", data=df, ax=ax1)
        ax1.set_title("Churn Count (0 = Stay, 1 = Churn)")
        st.pyplot(fig1)

        # === Improved Bar Graph Section ===
        st.markdown("### 📊 How Key Features Differ Between Churned and Non-Churned Users")
        for col in available_features[:4]:  # Show only top 4
            try:
                avg_values = df.groupby("churned")[col].mean().reset_index()
                avg_values["churned"] = avg_values["churned"].map({0: "Stayed", 1: "Churned"})

                fig, ax = plt.subplots(figsize=(6, 3.5))
                sns.barplot(data=avg_values, x="churned", y=col, palette="Set2", ax=ax)
                ax.set_title(f"Average {col} by Churn Status")
                ax.set_xlabel("Churn Status")
                ax.set_ylabel(f"Average {col}")
                for i in range(len(avg_values)):
                    ax.text(i, avg_values[col][i], f"{avg_values[col][i]:.2f}", ha='center', va='bottom')

                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not plot {col}: {str(e)}")

        # === Correlation Heatmap ===
        st.markdown("### 🔥 Correlation Heatmap (Numeric Features Only)")
        numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["churned"], errors="ignore")
        if not numeric_df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
            st.pyplot(fig2)
        else:
            st.warning("No numeric features available for heatmap.")

        # === Insights & Recommendations ===
        st.markdown("### 💡 Key Insights & Recommendations")
        churn_rate = df["churned"].mean()
        st.markdown(f"**Overall Churn Rate:** `{churn_rate:.2%}`")

        insights = []
        if "notifications_clicked" in df.columns:
            nc = df.groupby("churned")["notifications_clicked"].mean()
            if nc[1] < nc[0]:
                insights.append("🔔 Customers who click more notifications churn less. Personalize notification content.")

        if "engagement_score" in df.columns:
            es = df.groupby("churned")["engagement_score"].mean()
            if es[1] < es[0]:
                insights.append("🎧 High engagement score reduces churn. Encourage playlist curation and listening time.")

        if "account_months" in df.columns:
            churn_new = df[df["account_months"] < 3]["churned"].mean()
            if churn_new > churn_rate:
                insights.append("🆕 New users churn more. Improve onboarding and early touchpoints.")

        if insights:
            for tip in insights:
                st.info(tip)
        else:
            st.info("No clear churn patterns detected. Try adding more behavioral features.")

# ===== Sidebar: Single Prediction UI =====
st.sidebar.markdown("## ✍️ Predict a Single Customer")

def get_user_input():
    age = st.sidebar.slider("Age", 18, 70, 35)
    weekly_hours = st.sidebar.slider("Weekly Listening Hours", 0, 40, 10)
    session_length = st.sidebar.slider("Average Session Length", 5, 60, 20)
    skip_rate = st.sidebar.slider("Song Skip Rate (%)", 0, 100, 20)
    notifications_clicked = st.sidebar.slider("Notifications Clicked", 0, 50, 5)
    account_months = st.sidebar.slider("Account Age (Months)", 1, 60, 12)
    engagement_score = st.sidebar.slider("Engagement Score", 0.0, 1.0, 0.5)

    return pd.DataFrame({
        'age': [age],
        'weekly_hours': [weekly_hours],
        'average_session_length': [session_length],
        'skip_rate': [skip_rate],
        'notifications_clicked': [notifications_clicked],
        'account_months': [account_months],
        'engagement_score': [engagement_score]
    })

user_input_df = get_user_input()
user_input_df = user_input_df.reindex(columns=expected_features, fill_value=0)

if st.sidebar.button("🔮 Predict Churn"):
    pred = model.predict(user_input_df)[0]
    prob = model.predict_proba(user_input_df)[0][1]
    if pred == 1:
        st.error(f"⚠️ Prediction: Customer likely to churn. (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Prediction: Customer likely to stay. (Probability: {1 - prob:.2%})")

# ===== Bulk Prediction =====
# ===== Bulk Prediction =====
if uploaded_file:
    st.markdown("### 📥 Bulk Churn Prediction")
    try:
        input_df = df.reindex(columns=expected_features, fill_value=0)
        df["churn_prediction"] = model.predict(input_df)
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction File", csv, "churn_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")


