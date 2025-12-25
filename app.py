import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ======================================================
# PAGE CONFIG (MUST BE FIRST)
# ======================================================
st.set_page_config(
    page_title="Employee Attrition Risk",
    layout="centered"
)

# ======================================================
# SIDEBAR – CONTROL PANEL
# ======================================================
st.sidebar.markdown(
    """
    <div style="padding:10px 0;">
        <h2 style="color:#F97316; margin-bottom:5px;">Control Panel</h2>
        <p style="color:#A1A1AA; font-size:14px;">
            Navigate & filter employee data
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Dashboards", "Risk Table"]
)

# ======================================================
# APP HEADER
# ======================================================
st.markdown(
    """
    <div style="padding:20px 0;">
        <h1 style="color:#F97316; margin-bottom:0;">
            Employee Attrition Risk System
        </h1>
        <p style="color:#A1A1AA; font-size:16px;">
            Predict • Prioritize • Prevent employee attrition
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "This tool helps HR teams proactively identify employees at risk of attrition "
    "using historical data and workplace factors."
)

# ======================================================
# FILE UPLOAD
# ======================================================
uploaded_file = st.file_uploader(
    "Upload Employee Data (CSV or Excel)",
    type=["csv", "xlsx"]
)

# ======================================================
# LOAD DATA
# ======================================================
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file, sep="\t")
    return pd.read_excel(file)

if uploaded_file:
    df = load_data(uploaded_file)
    st.success("File uploaded successfully.")
else:
    df = pd.read_csv("data/employee_attrition.csv", sep="\t")
    st.info("Using demo dataset.")

# ======================================================
# REQUIRED COLUMNS CHECK
# ======================================================
required_columns = {
    'Age', 'Department', 'JobRole', 'MonthlyIncome',
    'OverTime', 'JobSatisfaction', 'Attrition'
}

missing_cols = required_columns - set(df.columns)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# ======================================================
# SIDEBAR FILTERS
# ======================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

selected_department = st.sidebar.multiselect(
    "Department",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

selected_overtime = st.sidebar.multiselect(
    "OverTime",
    options=df['OverTime'].unique(),
    default=df['OverTime'].unique()
)

selected_jobrole = st.sidebar.multiselect(
    "Job Role",
    options=df['JobRole'].unique(),
    default=df['JobRole'].unique()
)

df = df[
    (df['Department'].isin(selected_department)) &
    (df['OverTime'].isin(selected_overtime)) &
    (df['JobRole'].isin(selected_jobrole))
]

# ======================================================
# MODEL + RISK CALCULATION
# ======================================================
cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_model['Attrition'] = df_model['Attrition'].map({'No': 0, 'Yes': 1})

X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

def risk_bucket(p):
    if p >= 0.6:
        return "High"
    elif p >= 0.3:
        return "Medium"
    return "Low"

results = X_test.copy()
results['Attrition Probability'] = y_prob
results['Risk Level'] = results['Attrition Probability'].apply(risk_bucket)

risk_summary = results['Risk Level'].value_counts()

# ======================================================
# OVERVIEW PAGE
# ======================================================
if page == "Overview":
    st.header("Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 📊 Risk Snapshot")
    col1, col2, col3 = st.columns(3)

    col1.metric("🔥 High Risk", int(risk_summary.get("High", 0)))
    col2.metric("⚠️ Medium Risk", int(risk_summary.get("Medium", 0)))
    col3.metric("✅ Low Risk", int(risk_summary.get("Low", 0)))

    st.markdown("---")
    st.subheader("Recommended HR Actions")
    st.markdown(
        """
        - **High Risk** → Immediate HR intervention  
        - **Medium Risk** → Manager check-in  
        - **Low Risk** → Monitor periodically  
        """
    )

# ======================================================
# DASHBOARDS PAGE
# ======================================================
elif page == "Dashboards":
    st.header("Attrition Insights")
    st.caption("All charts reflect the selected filters")
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Risk Distribution")
    st.bar_chart(risk_summary)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Rate by Department")
        st.bar_chart(
            df_model.groupby('Department')['Attrition']
            .mean()
            .sort_values(ascending=False)
        )

    with col2:
        st.subheader("Attrition by Overtime")
        st.bar_chart(
            df_model.groupby('OverTime')['Attrition'].mean()
        )

# ======================================================
# RISK TABLE PAGE
# ======================================================
elif page == "Risk Table":
    st.header("Employee Risk Table")
    st.markdown("<br>", unsafe_allow_html=True)

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "High", "Medium", "Low"]
    )

    if risk_filter == "All":
        st.dataframe(results, use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            results[results['Risk Level'] == risk_filter],
            use_container_width=True,
            hide_index=True
        )
