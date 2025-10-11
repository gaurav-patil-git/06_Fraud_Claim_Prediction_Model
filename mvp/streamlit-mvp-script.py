import pandas as pd
import joblib
import streamlit as st

model = joblib.load('model/model_obj.pkl')

scaler_obj = joblib.load('model/scaler_obj.pkl')

input_data = scaler_obj['input_features']
scaler = scaler_obj['normal_scaler']
col_to_scale = scaler_obj['cols_to_scale']
processed_features = scaler_obj['processed_features']

# Set the page configuration and title
st.set_page_config(page_title="Insurance Claim Risk Predictor", page_icon="📊")
st.title("Insurance Claim Risk Predictor")

# Sorted dropdown options
insured_occupation_options = sorted([
    'craft-repair', 'sales', 'armed-forces', 'tech-support', 'prof-specialty',
    'other-service', 'priv-house-serv', 'exec-managerial', 'protective-serv',
    'machine-op-inspct', 'transport-moving', 'handlers-cleaners',
    'adm-clerical', 'farming-fishing'
])

insured_hobbies_options = sorted([
    'sleeping', 'board-games', 'bungie-jumping', 'base-jumping', 'golf', 'camping',
    'dancing', 'skydiving', 'reading', 'movies', 'hiking', 'yachting', 'paintball',
    'chess', 'kayaking', 'polo', 'basketball', 'video-games', 'cross-fit', 'exercise'
])

incident_severity_options = ['Minor Damage', 'Major Damage', 'Total Loss']
incident_hour_options = list(range(1, 25))

auto_model_options = sorted([
    '92x', 'RAM', 'Tahoe', '95', 'Pathfinder', 'A5', 'Camry', 'F150', 'A3', 'Neon', 'MDX',
    'Maxima', 'Legacy', 'TL', 'Impreza', 'RSX', 'Forrestor', 'Escape', 'Corolla',
    '3 Series', 'C300', 'Wrangler', 'M5', 'X5', 'E400', 'Highlander', 'Civic',
    'Silverado', 'CRV', '93', 'Accord', 'X6', 'Malibu', 'Fusion', 'ML350', 'Passat',
    'Ultima', 'Jetta', 'Grand Cherokee'
])

auto_year_options = list(range(1995, 2016))
day_options = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# -----------------------------------
# 🧩 Input Layout (Grid of 4 columns)
# -----------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
        policy_deductible = st.number_input("Policy Deductible", min_value=0, step=100)
        insured_occupation = st.selectbox("Insured Occupation", insured_occupation_options)
        capital_gains = st.number_input("Capital Gains", step=100)
        incident_hour_of_the_day = st.selectbox("Incident Hour of the Day", incident_hour_options)

with col2:
        policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0.0, step=10.0)
        insured_hobbies = st.selectbox("Insured Hobbies", insured_hobbies_options)
        capital_loss = st.number_input("Capital Loss", step=100)
        total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, step=100.0)

with col3:
        umbrella_limit = st.number_input("Umbrella Limit", step=1000)
        incident_severity = st.selectbox("Incident Severity", incident_severity_options)
        auto_model = st.selectbox("Auto Model", auto_model_options)
        auto_year = st.selectbox("Auto Year", auto_year_options)

with col4:
        day = st.selectbox("Day of Week", day_options)
        # Transform Mon–Sun to numerical (0–6)
        day_mapping = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        day_num = day_mapping[day]

submit_button = st.button("🚀 Calculate Risk")

if submit_button:
    df = pd.DataFrame(0, columns= processed_features, index= [0])

    input_data = {
        'policy_deductible': policy_deductible,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_occupation': insured_occupation,
        'insured_hobbies': insured_hobbies,
        'capital-gains': capital_gains,
        'capital-loss': capital_loss,
        'incident_severity': incident_severity,
        'incident_hour_of_the_day': incident_hour_of_the_day,
        'total_claim_amount': total_claim_amount,
        'auto_model': auto_model,
        'auto_year': auto_year,
        'day': day_num
    }

    numeric_features = [
        'policy_deductable',
        'policy_annual_premium',
        'umbrella_limit',
        'capital-gains',
        'capital-loss',
        'incident_hour_of_the_day',
        'total_claim_amount',
        'auto_year',
        'day'
    ]

    for col in numeric_features:
        df[col] = input_data[col]

    occ = input_data['insured_occupation']
    for col in df.columns:
        if col.startswith('insured_occupation_'):
            category = col.replace('insured_occupation_', '')
            if occ == category:
                df[col] = 1
            else:
                df[col] = 0

    hobby = input_data['insured_hobbies']
    for col in df.columns:
        if col.startswith('insured_hobbies_'):
            category = col.replace('insured_hobbies_', '')
            if hobby == category:
                df[col] = 1
            else:
                df[col] = 0

    model = input_data['auto_model']
    for col in df.columns:
        if col.startswith('auto_model_'):
            category = col.replace('auto_model_', '')
            if model == category:
                df[col] = 1
            else:
                df[col] = 0

    severity_map = {
        'Minor Damage': 0,
        'Major Damage': 1,
        'Total Loss': 2
    }

    if input_data['incident_severity'] in severity_map:
        df['incident_severity'] = severity_map[input_data['incident_severity']]
    else:
        df['incident_severity'] = 0  # default / unknown case

    df[col_to_scale] = scaler.transform(df[col_to_scale])

    probability = model.predict_proba(df)[:,1][0]

    st.write(f"Fraud Claim Probability: {probability:.2%}")

