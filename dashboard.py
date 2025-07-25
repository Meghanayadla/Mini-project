import streamlit as st
import requests

st.title("Ola S1 Safety Monitoring")

temperature = st.slider("Battery Temperature (°C)", 20, 70)
voltage = st.slider("Voltage (V)", 40, 60)
current = st.slider("Current (A)", 0, 10)
charge_cycles = st.slider("Charge Cycles", 0, 1500)

if st.button("Check Burn Risk"):
    data = {
        "temperature": temperature,
        "voltage": voltage,
        "current": current,
        "charge_cycles": charge_cycles
    }
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    result = response.json()
    if result['burn_risk'] == 1:
        st.error("🔥 Warning: High Risk of Battery Burn!")
    else:
        st.success("✅ Battery is Safe.")
