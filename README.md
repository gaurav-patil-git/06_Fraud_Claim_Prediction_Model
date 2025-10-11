# 🤖 Fraud Insurance Claim Prediction| BFSI and Fintech

### Develop automobile insurance claim assessment ML model to predict the probability of fraud application based on historic data and help Surveyor prepare claim report.

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#model-preview">Model Preview</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools-technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning-preparation">Data Cleaning & Preparation</a>
- <a href="#model-development">Model Development</a>
- <a href="#streamlit-app">Streamlit App</a>
- <a href="#author-contact">Author & Contact</a>

<h2><a class="anchor" id="overview"></a>📝 Overview</h2>

This project aims to develop a machine learning product to help Motor Insurance Surveyor to assess applicants and identify potential fraud claim. This project streamlines claim approval process, reduces risk of financial loses and enables data driven decision.
- Develop a classification model
- Ensure high value on model recall
- Deploy a most viable product (MVP) using Streamlit application.


<h2><a class="anchor" id="model-preview"></a>🔗 Model Preview</h2>


<h2><a class="anchor" id="credits"></a>🪪 Credits</h2>

NaN

<h2><a class="anchor" id="dataset"></a>📊 Dataset</h2>

`.xlsx` files located in `/data/raw` folder


#### Motor Insurance Data:
| Feature                       | Description                                                                 | Data Type |
|-------------------------------|-----------------------------------------------------------------------------|-----------|
| months_as_customer            | Duration (in months) the customer has been with the insurance company       | Integer   |
| age                           | Age of the insured person                                                    | Integer   |
| policy_number                 | Unique identifier for the insurance policy                                   | String    |
| policy_bind_date              | Date when the policy was issued                                              | Date      |
| policy_state                  | State in which the policy is issued                                          | String    |
| policy_split                  | Type of policy split (e.g., split, non-split)                               | String    |
| policy_deductable             | Deductible amount for the policy                                            | Float     |
| policy_annual_premium         | Annual premium amount for the policy                                        | Float     |
| umbrella_limit                | Umbrella policy coverage limit                                              | Float     |
| insured_zip                   | Zip code of the insured person's residence                                   | String    |
| insured_sex                   | Gender of the insured person                                                | String    |
| insured_education_level       | Highest education level of the insured                                       | String    |
| insured_occupation            | Occupation of the insured person                                             | String    |
| insured_hobbies               | Hobbies or recreational activities of the insured                            | String    |
| insured_relationship          | Relationship status of the insured                                           | String    |
| capital-gains                  | Capital gains of the insured person                                          | Float     |
| capital-loss                   | Capital losses of the insured person                                         | Float     |
| incident_date                  | Date when the incident/accident occurred                                     | Date      |
| incident_type                  | Type of incident (e.g., collision, theft, fire)                              | String    |
| collision_type                 | Type of collision if applicable (e.g., rear-end, side-swipe)                 | String    |
| incident_severity              | Severity of the incident (e.g., minor, major, total loss)                    | String    |
| authorities_contacted          | Authorities contacted during the incident                                    | String    |
| incident_state                 | State in which the incident occurred                                         | String    |
| incident_city                  | City in which the incident occurred                                          | String    |
| incident_location              | Specific location of the incident                                            | String    |
| incident_hour_of_the_day       | Hour of the day when the incident occurred (0-23)                            | Integer   |
| number_of_vehicles_involved    | Number of vehicles involved in the incident                                   | Integer   |
| property_damage                | Indicator if property damage occurred                                        | String    |
| bodily_injuries                | Number of bodily injuries reported                                           | Integer   |
| witnesses                      | Number of witnesses present at the incident                                   | Integer   |
| police_report_available        | Indicator if a police report is available                                    | String    |
| total_claim_amount             | Total amount claimed from the insurance                                      | Float     |
| injury_claim                   | Amount claimed for injuries                                                  | Float     |
| property_claim                 | Amount claimed for property damage                                           | Float     |
| vehicle_claim                  | Amount claimed for vehicle damage                                            | Float     |
| auto_make                      | Make/brand of the insured vehicle                                           | String    |
| auto_model                     | Model of the insured vehicle                                                | String    |
| auto_year                      | Manufacturing year of the insured vehicle                                    | Integer   |
| fraud_reported                 | Indicator if the claim was reported as fraudulent                            | String    |


<h2><a class="anchor" id="tools-technologies"></a>🛠️ Tools & Technologies</h2>

|Task                  | Libraries Used                      |
|----------------------|-------------------------------------|
| Data Preprocessing   | Pandas                              |
| Data Visualization   | Matplotlib, Seaborn                 |
| Feature Engineering  | Pandas, Statsmodels, Scikit-learn   |
| Model Training       | Scikit-learn, XGBoost               |
| Model Fine Tuning    | Scikit-learn                        |
| UI Frontend          | Streamlit                           |
| Code Assistance      | ChatGPT, GitHub Copilot             |


<h2><a class="anchor" id="project-structure"></a>📁 Project Structure</h2>

```
02_Credit_Risk_Prediction_Model/
│
├── data/
│   ├── raw/               # Original, immutable data dumps
│   ├── processed/         # Cleaned & feature-engineered datasets
│
├── documents/             # Scope of work
│
├── models/                # Saved model and scaler objects 
│
├── mvp/                   # Minimum Viable Product (Streamlit app)
│
├── notebooks/             # Jupyter notebooks organized by purpose
│
├── visuals/               # Mockups and model preview
│
├── README.md              # High-level project overview
├── .gitignore             # Ignore data, models, logs if using Git

```

<h2><a class="anchor" id="data-cleaning-preparation"></a>🧼 Data Cleaning & Preparation</h2>

### **Data Cleaning**
- Detected 9.1% rows of `authorities_contacted` with missing values and imputed them with mode.
- Detected and handled anomalies :
  - Umbrella Limit : -1000000 (min value) - used absolute values
  - Income, Sanction Amount, Loan Amount, Processing Fee, GST, Net Disbursement, Bank Balance at Application : 0 (min value) - Filtered the data
- Used Hartigan’s Dip Test to identify bimodality and segment dataset.
- Corrected categories of collision_type, property_damage, police_report_available : '?'

### **Feature Engineering**
- Derived five new time features using existing date feature
- Applied feature scaling
- Used VIF to detect mutlicolinearity and eliminate those with higher value
- Used WoE and IV to detech categorical features with low predictive power and eliminated them
- Applied One Hot Encoding
- Applied Label Encoding

<h2><a class="anchor" id="model-development"></a>🤖 Model Development</h2>

### **Model Training**
Performance table of different models with test rank:
| Classifier | 0 | | 1 | | Test Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | recall | f1-score | recall | f1-score | |
| Logistic Regression | 0.65       | 0.72         | 0.54       | 0.43         | 4         |
| SVM                 | 0.87       | 0.84         | 0.43       | 0.48         | 6         |
| Decision Tree       | 0.83       | 0.85         | 0.63       | 0.60         | 3         |
| Majority Voting     | 0.87       | 0.86         | 0.54       | 0.57         | 4         |
| Random Forest       | 0.81       | 0.85         | 0.77       | 0.67         | 1         |
| XGBoost             | 0.82       | 0.84         | 0.66       | 0.61         | 2         |


<h2><a class="anchor" id="streamlit-app"></a>📱 Streamlit App</h2>

- Exported model using `joblib`
- Developed and deployed a most viable product (MVP) using `streamlit`
- This MVP will be utilized by underwriters for 3 to 6 months for feedback and improvement before production

<h2><a class="anchor" id="author-contact"></a>📝 Author & Contact</h2>

**Gaurav Patil** (Data Analyst) 
- 🔗 [LinkedIn](https://www.linkedin.com/in/gaurav-patil-in/)


