# Employee Attrition Risk System 🔍

A production-ready Machine Learning web application that predicts employee attrition risk and helps HR teams take proactive, data-driven decisions.

🔗 **Live Application:**  
https://employee-attrition-risk-app.streamlit.app/

---

## 📌 Problem Statement
Employee attrition is expensive and often unpredictable.  
Organizations need a way to **identify employees at risk early** so they can take preventive actions instead of reacting after resignations happen.

---

## 💡 Solution
This application uses Machine Learning to:
- Predict employee attrition probability
- Categorize employees into **High / Medium / Low risk**
- Provide **interactive dashboards and filters** for non-technical users (HR, managers)


## 🚀 Key Features
- Upload your own employee dataset (CSV / Excel)
- Built-in demo dataset for quick testing
- ML-based attrition risk prediction
- Risk categorization using probability thresholds
- Interactive dashboards:
  - Risk distribution
  - Department-wise attrition rate
  - Overtime vs attrition
- Filters by:
  - Department
  - Job Role
  - Overtime
- Clean, professional UI
- Fully deployed on Streamlit Cloud


## 🧠 Machine Learning Approach
- **Model:** Random Forest Classifier  
- Handles class imbalance using balanced weights  
- Uses probability-based risk scoring instead of hard labels  
- Focuses on business usability rather than just accuracy  

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Git & GitHub
- Streamlit Cloud (Deployment)



## 📂 Project Structure
employee-attrition-risk-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── employee_attrition.csv
│
├── notebooks/
│ └── exploratory_analysis.ipynb
│
├── outputs/
│ ├── plots/
│ └── predictions.csv
│
└── .streamlit/
└── config.toml


## 📈 Use Cases
- HR Analytics teams
- Business Analysts
- Data Science / ML portfolios
- Demonstration of end-to-end ML deployment


## 👩‍💻 Author
**Anushka Jamkar**  
Aspiring Data Analyst / Machine Learning Engineer  

