import streamlit as st
import numpy as np
import joblib

model = joblib.load("mushroom_nb.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Mushroom Classification App (Naive Bayes)")

st.write("Enter mushroom details to classify as **Edible or Poisonous**")

cap_diameter = st.number_input("Cap Diameter (cm)", 1.0, 20.0)
cap_shape = st.selectbox("Cap Shape", ["Bell", "Convex", "Flat", "Sunken"])
gill_attachment = st.selectbox("Gill Attachment", ["Attached", "Free"])
gill_color = st.selectbox("Gill Color", ["White", "Brown", "Pink", "Black"])
stem_height = st.number_input("Stem Height (cm)", 1.0, 30.0)
stem_width = st.number_input("Stem Width (cm)", 0.1, 10.0)
stem_color = st.selectbox("Stem Color", ["White", "Brown", "Yellow"])
season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

cap_shape_map = {"Bell":0, "Convex":1, "Flat":2, "Sunken":3}
gill_attach_map = {"Attached":0, "Free":1}
gill_color_map = {"White":0, "Brown":1, "Pink":2, "Black":3}
stem_color_map = {"White":0, "Brown":1, "Yellow":2}
season_map = {"Spring":0, "Summer":1, "Autumn":2, "Winter":3}

cap_shape = cap_shape_map[cap_shape]
gill_attachment = gill_attach_map[gill_attachment]
gill_color = gill_color_map[gill_color]
stem_color = stem_color_map[stem_color]
season = season_map[season]

if st.button("Predict"):
    features = np.array([[cap_diameter, cap_shape, gill_attachment, gill_color,
                          stem_height, stem_width, stem_color, season]])
    
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]

    result = "🍏 Edible" if pred == 0 else "☠️ Poisonous"
    st.success(f"Prediction: **{result}**")
