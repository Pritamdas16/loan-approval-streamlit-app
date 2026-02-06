import streamlit as st
import pickle
import pandas as pd
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}
.block-container {
    max-width: 820px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #e5e7eb;
}
p {
    color: #9ca3af;
}
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: #020617 !important;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    padding: 10px;
}
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #4f46e5);
    color: white;
    border-radius: 14px;
    padding: 0.8rem;
    font-size: 16px;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #1d4ed8, #4338ca);
}
.result-approve {
    background: linear-gradient(90deg, #064e3b, #022c22);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}
.result-reject {
    background: linear-gradient(90deg, #7f1d1d, #450a0a);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD ARTIFACTS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "loan_model.pkl"), "rb"))
ohe = pickle.load(open(os.path.join(BASE_DIR, "ohe.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
model_columns = pickle.load(open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb"))

cat_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

# ==================== HEADER ====================
st.markdown("<h1 style='text-align:center;'>🏦 Loan Approval Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered loan eligibility assessment</p>", unsafe_allow_html=True)
st.markdown("---")

st.info("👉 Fill in the details and click **Check Loan Eligibility** to see the result.")

# ==================== TABS ====================
tab1, tab2 = st.tabs(["👤 Applicant Details", "💰 Financial Details"])

# ---------- Applicant Details ----------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        employment_status = st.selectbox(
            "Employment Status", ["Salaried", "Self-employed", "Unemployed"]
        )
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Home", "Car", "Education", "Personal"]
        )
        property_area = st.selectbox(
            "Property Area", ["Urban", "Semiurban", "Rural"]
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        employer_category = st.selectbox(
            "Employer Category", ["Government", "MNC", "Private", "Unemployed"]
        )

# ---------- Financial Details ----------
with tab2:
    col3, col4 = st.columns(2)

    with col3:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0.0)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0)
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900
        )

    with col4:
        age = st.number_input("Age", min_value=18)
        dependents = st.number_input("Dependents", min_value=0)
        loan_term = st.number_input("Loan Term (months)", min_value=1)

    with st.expander("🔧 Advanced Financial Details (Optional)"):
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)", min_value=0.0
        )
        savings = st.number_input("Savings (₹)", min_value=0.0)
        collateral_value = st.number_input(
            "Collateral Value (₹)", min_value=0.0
        )
        existing_loans = st.number_input(
            "Existing Loans", min_value=0
        )

# ==================== PREDICTION ====================
st.markdown("---")

if st.button("🔍 Check Loan Eligibility", use_container_width=True):

    total_income = applicant_income + coapplicant_income
    dti_ratio = loan_amount / total_income if total_income > 0 else 0

    input_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existing_loans,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "DTI_Ratio_sq": dti_ratio ** 2,
        "Credit_Score_sq": credit_score ** 2,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer_category
    }

    input_df = pd.DataFrame([input_data])

    encoded = ohe.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded, columns=ohe.get_feature_names_out(cat_cols)
    )

    final_df = pd.concat(
        [input_df.drop(columns=cat_cols).reset_index(drop=True),
         encoded_df.reset_index(drop=True)],
        axis=1
    )

    final_df = final_df.reindex(columns=model_columns, fill_value=0)
    final_scaled = scaler.transform(final_df)

    prediction = model.predict(final_scaled)[0]
    probability = model.predict_proba(final_scaled)[0][1] * 100
    display_conf = min(99.0, probability if prediction == 1 else 100 - probability)

    st.progress(display_conf / 100)

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-approve">
                <h2>✅ Loan Approved</h2>
                <p style="font-size:18px;">Confidence Score: <b>{display_conf:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-reject">
                <h2>❌ Loan Rejected</h2>
                <p style="font-size:18px;">Confidence Score: <b>{display_conf:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==================== RISK EXPLANATION ====================
    st.markdown("### 🔎 Risk Analysis")

    reasons = []
    if credit_score < 600:
        reasons.append("Low credit score")
    if dti_ratio > 0.4:
        reasons.append("High debt-to-income ratio")
    if loan_amount > applicant_income * 10 and applicant_income > 0:
        reasons.append("Loan amount is high compared to income")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("• Overall financial profile appears stable")

# ==================== FOOTER ====================
st.markdown(
    "<hr><p style='text-align:center; font-size:12px;'>Built by Pritam Das · Machine Learning Project</p>",
    unsafe_allow_html=True
)
