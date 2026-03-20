# IBM Employee Attrition Prediction

A complete end-to-end data science project that identifies the key 
drivers of employee attrition and predicts which employees are at 
highest risk of leaving the organisation.

## Live App

👉 [Employee Attrition Risk Predictor](https://ibm-employee-attrition.streamlit.app)

The app supports:
- **Single employee assessment** — enter an employee profile and 
  get an instant attrition risk score
- **Bulk CSV assessment** — upload your entire workforce and 
  download a risk scored results file

---

## Business Problem

Employee attrition costs organisations between 1.5 and 2 times an 
employee's annual salary in replacement costs. This project 
identifies who is likely to leave and why — enabling targeted 
retention strategies before it is too late.

---

## Key Findings

- **Overtime is the strongest attrition driver** — employees working 
  overtime are twice as likely to leave (Odds Ratio 2.00)
- **Frequent business travel** increases attrition odds by 73%
- **Sales Representatives** have the highest attrition rate of any 
  role at 39.8% — more than double the overall rate of 16.1%
- **Low tenure employees are at highest risk** — median years at 
  company for leavers is 3 versus 6 for stayers
- **Career stagnation drives attrition** — every additional year 
  without a promotion increases attrition odds by 52%
- **Senior roles are protective** — Research Directors show only 
  2.5% attrition versus the 16.1% overall rate
- **Employee engagement matters** — a composite engagement score 
  combining satisfaction dimensions reduces attrition odds by 26%

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.81 |
| Recall (Attrition = Yes) | 0.66 |
| Precision (Attrition = Yes) | 0.38 |
| F1 Score (Attrition = Yes) | 0.48 |
| Overall Accuracy | 0.77 |

The model catches 66% of actual leavers — 31 out of 47 in the 
test set. A naive baseline that always predicts No achieves 83.9% 
accuracy but catches zero leavers. Our model trades some overall 
accuracy for meaningful minority class detection.

---

## Project Structure
```
ibm-employee-attrition/
│
├── app.py                    ← Streamlit web application
├── model.pkl                 ← Trained logistic regression model
├── scaler.pkl                ← Fitted StandardScaler
├── columns.pkl               ← Feature column names
├── requirements.txt          ← Python dependencies
└── README.md                 ← Project documentation
```

---

## Methodology

### Data
- Dataset: IBM HR Analytics Employee Attrition and Performance
- 1,470 employee records across 35 features
- Target variable: Attrition (Yes/No) — 16.1% positive class

### Exploratory Data Analysis
- Seven data quality checks — missing values, duplicates, 
  constants, impossible values, cardinality, data types, 
  identifier columns
- Univariate analysis — distributions, skewness, concentration
- Bivariate analysis — attrition rates by every feature
- Correlation analysis — multicollinearity detection
- Feature engineering — three composite scores

### Feature Engineering
Three composite features were created from correlated clusters:

| Feature | Components |
|---|---|
| EngagementScore | JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance |
| SeniorityScore | Age, TotalWorkingYears, JobLevel, MonthlyIncome |
| TenureScore | YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager |

### Modeling
- Algorithm: Logistic Regression with L2 regularization (C=0.1)
- Class imbalance handled via class_weight='balanced'
- Validation: Stratified 5-fold cross validation
- Cross validation ROC-AUC: 0.8322 (std: 0.0266)

### Ethical Considerations
Gender and MaritalStatus are protected characteristics under 
employment law in most jurisdictions. Their inclusion in an HR 
predictive model raises ethical and legal considerations around 
algorithmic discrimination. Both features are included in this 
academic project for analytical completeness but their use in 
a production HR system would require a fairness audit and legal 
review before deployment.

---

## Business Recommendations

Based on model findings the following actions are recommended:

1. **Review overtime policies** — the strongest controllable 
   attrition driver. Identify departments with highest overtime 
   concentration and investigate resourcing levels
2. **Implement structured promotion review cycles** — career 
   stagnation is a significant predictor. Regular promotion 
   reviews reduce the risk of employees feeling stuck
3. **Develop targeted retention programmes for Sales and HR** — 
   both functions show consistently elevated attrition rates 
   across multiple dimensions
4. **Invest in manager relationship quality** — tenure with 
   current manager is a protective factor. Leadership stability 
   and manager effectiveness programmes reduce attrition risk
5. **Monitor engagement scores quarterly** — the composite 
   EngagementScore can serve as an early warning system for 
   attrition risk across the workforce

---

## Tech Stack

- Python 3
- pandas — data manipulation
- numpy — numerical computation
- matplotlib and seaborn — data visualisation
- scikit-learn — modeling and evaluation
- Streamlit — web application deployment
- joblib — model serialisation

---

## How to Run Locally
```bash
git clone https://github.com/MthulisiMlilo-TVXM/IBM-Employee-Attrition
cd IBM-Employee-Attrition
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Mthulisi Mlilo  
[GitHub](https://github.com/MthulisiMlilo-TVXM)# IBM-Employee-Attrition
Employee attrition prediction using logistic regression 
