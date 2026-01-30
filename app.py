import streamlit as st
import pandas as pd
import pickle

# -------- LOAD MODEL & SCALER --------
with open("personality_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
    
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


# IMPORTANT: features manually define ya CSV se lo
feature_columns = [
    'social_energy', 'alone_time_preference', 'talkativeness',
    'deep_reflection', 'group_comfort', 'party_liking',
    'listening_skill', 'empathy', 'organization',
    'leadership', 'risk_taking', 'public_speaking_comfort',
    'curiosity', 'routine_preference', 'excitement_seeking',
    'friendliness', 'planning', 'spontaneity',
    'adventurousness', 'reading_habit', 'sports_interest',
    'online_social_usage', 'travel_desire', 'gadget_usage',
    'work_style_collaborative', 'decision_speed'
]

# -------- UI --------
st.title("🧠 Personality Prediction App")
st.write("Fill the sliders. Model will quietly judge you.")

# Create 3 columns
col1, col2, col3 = st.columns(3)
cols = [col1, col2, col3]

user_input = {}

for idx, feature in enumerate(feature_columns):
    with cols[idx % 3]:
        user_input[feature] = st.slider(
            feature.replace("_", " ").title(),
            min_value=0,
            max_value=10,
            value=5
        )


if st.button("Predict Personality"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    pred_encoded = model.predict(input_scaled)
    pred_original = label_encoder.inverse_transform(pred_encoded)

    st.success(f"Predicted Personality Type: **{pred_original[0]}**")

