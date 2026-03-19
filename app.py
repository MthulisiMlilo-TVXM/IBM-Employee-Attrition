# ── Imports ───────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ── Load model assets ─────────────────────────────────────────────────
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👤",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────────
st.title("Employee Attrition Risk Predictor")
st.markdown("""
This tool predicts the likelihood of an employee leaving the 
organisation. Use the **Single Employee** tab for individual 
assessments or the **Bulk Assessment** tab to upload a CSV file.
""")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Employee", "Bulk CSV Assessment"])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE EMPLOYEE
# ══════════════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Employee Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        age = st.slider("Age", 18, 60, 35)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox(
            "Marital Status", ["Divorced", "Married", "Single"])
        distance = st.slider("Distance From Home (km)", 1, 30, 5)
        num_companies = st.slider(
            "Number of Companies Worked", 0, 9, 1)

    with col2:
        st.markdown("**Job Details**")
        department = st.selectbox(
            "Department",
            ["Human Resources", "Research & Development", "Sales"])
        job_role = st.selectbox("Job Role", [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"])
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        business_travel = st.selectbox(
            "Business Travel",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        job_involvement = st.slider(
            "Job Involvement (1-4)", 1, 4, 3)
        job_satisfaction = st.slider(
            "Job Satisfaction (1-4)", 1, 4, 3)

    with col3:
        st.markdown("**Compensation and Tenure**")
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=500)
        stock_option = st.slider(
            "Stock Option Level (0-3)", 0, 3, 1)
        percent_hike = st.slider(
            "Percent Salary Hike", 11, 25, 14)
        total_working_years = st.slider(
            "Total Working Years", 0, 40, 10)
        years_at_company = st.slider(
            "Years at Company", 0, 40, 5)
        years_current_role = st.slider(
            "Years in Current Role", 0, 18, 3)
        years_since_promotion = st.slider(
            "Years Since Last Promotion", 0, 15, 1)
        years_with_manager = st.slider(
            "Years With Current Manager", 0, 17, 3)

    st.divider()

    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Education**")
        education = st.slider("Education Level (1-5)", 1, 5, 3)
        education_field = st.selectbox("Education Field", [
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree"])
        training_times = st.slider(
            "Training Times Last Year", 0, 6, 2)

    with col5:
        st.markdown("**Satisfaction Ratings**")
        environment_satisfaction = st.slider(
            "Environment Satisfaction (1-4)", 1, 4, 3)
        relationship_satisfaction = st.slider(
            "Relationship Satisfaction (1-4)", 1, 4, 3)
        work_life_balance = st.slider(
            "Work Life Balance (1-4)", 1, 4, 3)
        daily_rate = st.slider("Daily Rate", 100, 1500, 800)
        hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate = st.slider(
            "Monthly Rate", 2000, 27000, 14000)
        performance_rating = st.selectbox(
            "Performance Rating", [3, 4])

    st.divider()

    if st.button("Predict Attrition Risk", type="primary"):

        input_data = {
            'Age': age,
            'DailyRate': daily_rate,
            'DistanceFromHome': distance,
            'Education': education,
            'EnvironmentSatisfaction': environment_satisfaction,
            'HourlyRate': hourly_rate,
            'JobInvolvement': job_involvement,
            'JobSatisfaction': job_satisfaction,
            'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies,
            'OverTime': 1 if overtime == 'Yes' else 0,
            'PercentSalaryHike': percent_hike,
            'PerformanceRating': performance_rating,
            'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option,
            'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times,
            'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_current_role,
            'YearsSinceLastPromotion': years_since_promotion,
            'YearsWithCurrManager': years_with_manager,
        }

        input_data['EngagementScore'] = np.mean([
            job_satisfaction,
            environment_satisfaction,
            relationship_satisfaction,
            work_life_balance]) / 4

        input_data['SeniorityScore'] = np.mean([
            (age - 18) / (60 - 18),
            total_working_years / 40,
            (monthly_income - 1000) / (20000 - 1000)])

        input_data['TenureScore'] = np.mean([
            years_at_company / 40,
            years_current_role / 18,
            years_with_manager / 17])

        input_df = pd.DataFrame([input_data])

        input_df['BusinessTravel_Travel_Frequently'] = (
            1 if business_travel == 'Travel_Frequently' else 0)
        input_df['BusinessTravel_Travel_Rarely'] = (
            1 if business_travel == 'Travel_Rarely' else 0)
        input_df['Department_Research & Development'] = (
            1 if department == 'Research & Development' else 0)
        input_df['Department_Sales'] = (
            1 if department == 'Sales' else 0)

        for field in [
            'Life Sciences', 'Marketing',
            'Medical', 'Other', 'Technical Degree']:
            input_df[f'EducationField_{field}'] = (
                1 if education_field == field else 0)

        input_df['Gender_Male'] = (1 if gender == 'Male' else 0)

        for role in [
            'Human Resources',
            'Laboratory Technician',
            'Manager',
            'Manufacturing Director',
            'Research Director',
            'Research Scientist',
            'Sales Executive',
            'Sales Representative']:
            input_df[f'JobRole_{role}'] = (
                1 if job_role == role else 0)

        input_df['MaritalStatus_Married'] = (
            1 if marital_status == 'Married' else 0)
        input_df['MaritalStatus_Single'] = (
            1 if marital_status == 'Single' else 0)

        input_df = input_df.reindex(columns=columns, fill_value=0)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Prediction Result")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prediction == 1:
                st.error("HIGH ATTRITION RISK")
            else:
                st.success("LOW ATTRITION RISK")
            st.metric(
                label="Attrition Probability",
                value=f"{probability:.1%}")

        with col_r2:
            st.markdown("**Risk Level**")
            if probability < 0.3:
                risk_label = "Low Risk"
            elif probability < 0.6:
                risk_label = "Medium Risk"
            else:
                risk_label = "High Risk"
            st.progress(float(probability), text=risk_label)

        st.markdown("---")
        st.markdown("**Key Risk Factors Identified:**")

        risk_notes = []

        if overtime == 'Yes':
            risk_notes.append(
                "Working overtime — strongest attrition predictor")
        if business_travel == 'Travel_Frequently':
            risk_notes.append(
                "Frequent business travel — elevated attrition risk")
        if marital_status == 'Single':
            risk_notes.append(
                "Single — fewer personal ties, higher mobility")
        if years_since_promotion > 3:
            risk_notes.append(
                f"{years_since_promotion} years since last promotion "
                f"— career stagnation risk")
        if job_satisfaction <= 2:
            risk_notes.append(
                f"Low job satisfaction ({job_satisfaction}/4)")
        if monthly_income < 3500:
            risk_notes.append(
                f"Below average monthly income (${monthly_income:,})")
        if num_companies > 3:
            risk_notes.append(
                f"Worked at {num_companies} previous companies "
                f"— history of job changes")
        if environment_satisfaction <= 2:
            risk_notes.append(
                f"Low environment satisfaction "
                f"({environment_satisfaction}/4)")
        if work_life_balance <= 2:
            risk_notes.append(
                f"Poor work life balance ({work_life_balance}/4)")

        if risk_notes:
            for note in risk_notes:
                st.warning(f"• {note}")
        else:
            st.info("No major risk factors identified.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — BULK CSV ASSESSMENT
# ══════════════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Bulk Employee Assessment")
    st.markdown("""
    Upload a CSV file containing employee profiles. The app will 
    generate attrition risk scores for every employee and produce 
    a downloadable results file.
    """)

    # ── Download template ─────────────────────────────────────────────
    st.markdown("**Step 1 — Download the CSV template**")
    st.markdown(
        "Use this template to ensure your data is formatted correctly.")

    template_cols = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField',
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
        'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]

    template_df = pd.DataFrame(columns=template_cols)

    # Add one example row
    template_df.loc[0] = [
        35, 'Travel_Rarely', 800, 'Research & Development',
        5, 3, 'Life Sciences', 3, 'Male', 65,
        3, 'Research Scientist', 3,
        'Married', 5000, 14000,
        2, 'No', 14,
        3, 3,
        1, 10,
        2, 3,
        5, 3,
        1, 3
    ]

    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv_template,
        file_name="employee_template.csv",
        mime="text/csv")

    st.divider()

    # ── Upload CSV ────────────────────────────────────────────────────
    st.markdown("**Step 2 — Upload your completed CSV file**")
    uploaded_file = st.file_uploader(
        "Upload employee data CSV",
        type="csv")

    if uploaded_file is not None:

        # Load uploaded data
        upload_df = pd.read_csv(uploaded_file)

        st.success(
            f"File uploaded successfully — "
            f"{len(upload_df)} employees found.")
        st.dataframe(upload_df.head())

        if st.button("Run Bulk Assessment", type="primary"):

            results = []

            for idx, row in upload_df.iterrows():

                # ── Build input for each row ──────────────────────────
                input_data = {
                    'Age': row.get('Age', 35),
                    'DailyRate': row.get('DailyRate', 800),
                    'DistanceFromHome': row.get(
                        'DistanceFromHome', 5),
                    'Education': row.get('Education', 3),
                    'EnvironmentSatisfaction': row.get(
                        'EnvironmentSatisfaction', 3),
                    'HourlyRate': row.get('HourlyRate', 65),
                    'JobInvolvement': row.get('JobInvolvement', 3),
                    'JobSatisfaction': row.get('JobSatisfaction', 3),
                    'MonthlyIncome': row.get('MonthlyIncome', 5000),
                    'MonthlyRate': row.get('MonthlyRate', 14000),
                    'NumCompaniesWorked': row.get(
                        'NumCompaniesWorked', 2),
                    'OverTime': 1 if row.get(
                        'OverTime', 'No') == 'Yes' else 0,
                    'PercentSalaryHike': row.get(
                        'PercentSalaryHike', 14),
                    'PerformanceRating': row.get(
                        'PerformanceRating', 3),
                    'RelationshipSatisfaction': row.get(
                        'RelationshipSatisfaction', 3),
                    'StockOptionLevel': row.get(
                        'StockOptionLevel', 1),
                    'TotalWorkingYears': row.get(
                        'TotalWorkingYears', 10),
                    'TrainingTimesLastYear': row.get(
                        'TrainingTimesLastYear', 2),
                    'WorkLifeBalance': row.get(
                        'WorkLifeBalance', 3),
                    'YearsAtCompany': row.get('YearsAtCompany', 5),
                    'YearsInCurrentRole': row.get(
                        'YearsInCurrentRole', 3),
                    'YearsSinceLastPromotion': row.get(
                        'YearsSinceLastPromotion', 1),
                    'YearsWithCurrManager': row.get(
                        'YearsWithCurrManager', 3),
                }

                # Composite features
                js = input_data['JobSatisfaction']
                es = input_data['EnvironmentSatisfaction']
                rs = input_data['RelationshipSatisfaction']
                wl = input_data['WorkLifeBalance']
                ag = input_data['Age']
                tw = input_data['TotalWorkingYears']
                mi = input_data['MonthlyIncome']
                ya = input_data['YearsAtCompany']
                yr = input_data['YearsInCurrentRole']
                yw = input_data['YearsWithCurrManager']

                input_data['EngagementScore'] = (
                    np.mean([js, es, rs, wl]) / 4)
                input_data['SeniorityScore'] = np.mean([
                    (ag - 18) / (60 - 18),
                    tw / 40,
                    (mi - 1000) / (20000 - 1000)])
                input_data['TenureScore'] = np.mean([
                    ya / 40, yr / 18, yw / 17])

                input_df = pd.DataFrame([input_data])

                bt = row.get('BusinessTravel', 'Non-Travel')
                input_df['BusinessTravel_Travel_Frequently'] = (
                    1 if bt == 'Travel_Frequently' else 0)
                input_df['BusinessTravel_Travel_Rarely'] = (
                    1 if bt == 'Travel_Rarely' else 0)

                dept = row.get('Department', 'Sales')
                input_df['Department_Research & Development'] = (
                    1 if dept == 'Research & Development' else 0)
                input_df['Department_Sales'] = (
                    1 if dept == 'Sales' else 0)

                ef = row.get('EducationField', 'Other')
                for field in [
                    'Life Sciences', 'Marketing',
                    'Medical', 'Other', 'Technical Degree']:
                    input_df[f'EducationField_{field}'] = (
                        1 if ef == field else 0)

                gen = row.get('Gender', 'Male')
                input_df['Gender_Male'] = (
                    1 if gen == 'Male' else 0)

                jr = row.get('JobRole', 'Research Scientist')
                for role in [
                    'Human Resources',
                    'Laboratory Technician',
                    'Manager',
                    'Manufacturing Director',
                    'Research Director',
                    'Research Scientist',
                    'Sales Executive',
                    'Sales Representative']:
                    input_df[f'JobRole_{role}'] = (
                        1 if jr == role else 0)

                ms = row.get('MaritalStatus', 'Married')
                input_df['MaritalStatus_Married'] = (
                    1 if ms == 'Married' else 0)
                input_df['MaritalStatus_Single'] = (
                    1 if ms == 'Single' else 0)

                input_df = input_df.reindex(
                    columns=columns, fill_value=0)
                input_scaled = scaler.transform(input_df)

                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(
                    input_scaled)[0][1]

                if probability < 0.3:
                    risk_level = "Low"
                elif probability < 0.6:
                    risk_level = "Medium"
                else:
                    risk_level = "High"

                results.append({
                    'Employee Index': idx + 1,
                    'Attrition Prediction': (
                        'Yes' if prediction == 1 else 'No'),
                    'Attrition Probability %': round(
                        probability * 100, 1),
                    'Risk Level': risk_level
                })

            # ── Build results dataframe ───────────────────────────────
            results_df = pd.DataFrame(results)

            # Add original columns back for context
            final_df = pd.concat(
                [upload_df.reset_index(drop=True),
                 results_df.drop(
                     columns=['Employee Index'])],
                axis=1)

            st.success(
                f"Assessment complete — "
                f"{len(final_df)} employees assessed.")

            # ── Summary stats ─────────────────────────────────────────
            st.markdown("**Assessment Summary:**")

            s1, s2, s3, s4 = st.columns(4)

            high_risk = (
                results_df['Risk Level'] == 'High').sum()
            medium_risk = (
                results_df['Risk Level'] == 'Medium').sum()
            low_risk = (
                results_df['Risk Level'] == 'Low').sum()
            avg_prob = results_df[
                'Attrition Probability %'].mean()

            s1.metric("Total Employees", len(final_df))
            s2.metric("High Risk", high_risk)
            s3.metric("Medium Risk", medium_risk)
            s4.metric("Avg Attrition Probability",
                      f"{avg_prob:.1f}%")

            # ── Results table ─────────────────────────────────────────
            st.markdown("**Full Results:**")
            st.dataframe(
                final_df.sort_values(
                    'Attrition Probability %',
                    ascending=False),
                use_container_width=True)

            # ── Download results ──────────────────────────────────────
            csv_results = final_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv_results,
                file_name="attrition_assessment_results.csv",
                mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Logistic Regression  |  "
    "Dataset: IBM HR Analytics  |  "
    "ROC-AUC: 0.81")