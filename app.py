import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

# Load model
@st.cache_resource
def load_model():
    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, encoder

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('data/ai4i2020.csv')

# Page config
st.set_page_config(
    page_title="NovaGuard - Predictive Maintenance",
    page_icon="🛡️",
    layout="wide"
)

# Header
st.title("🛡️ NovaGuard")
st.subheader("AI-Powered Predictive Maintenance for Precision Manufacturing")
st.markdown("---")

# Load
model, encoder = load_model()
df = load_data()

# Sidebar
st.sidebar.title("Machine Sensor Input")
st.sidebar.markdown("Enter real-time sensor readings:")

machine_type = st.sidebar.selectbox("Machine Type", ["L - Low", "M - Medium", "H - High"])
air_temp = st.sidebar.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
process_temp = st.sidebar.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)
rpm = st.sidebar.slider("Rotational Speed (RPM)", 1168, 2886, 1500)
torque = st.sidebar.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 253, 100)

predict_btn = st.sidebar.button("Analyze Machine Health", type="primary")

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    failure_count = df['Machine failure'].sum()
    st.metric("Total Failures", f"{failure_count:,}")
with col3:
    failure_rate = df['Machine failure'].mean()
    st.metric("Failure Rate", f"{failure_rate:.2%}")
with col4:
    healthy = len(df) - failure_count
    st.metric("Healthy Machines", f"{healthy:,}")

st.markdown("---")

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Failure Distribution by Machine Type")
    type_failure = df.groupby('Type')['Machine failure'].agg(['sum', 'count']).reset_index()
    type_failure.columns = ['Type', 'Failures', 'Total']
    type_failure['Failure Rate'] = type_failure['Failures'] / type_failure['Total']
    fig1 = px.bar(type_failure, x='Type', y='Failures',
                  color='Failure Rate', color_continuous_scale='RdYlGn_r',
                  title="Failures by Machine Type")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Tool Wear vs Torque (Failure Risk)")
    sample = df.sample(500, random_state=42)
    fig2 = px.scatter(sample, x='Tool wear', y='Torque',
                      color='Machine failure',
                      color_discrete_map={0: 'green', 1: 'red'},
                      labels={'Machine failure': 'Failed'},
                      title="Tool Wear vs Torque")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Prediction section
st.subheader("🔍 Real-Time Machine Health Analysis")

if predict_btn:
    type_map = {"L - Low": "L", "M - Medium": "M", "H - High": "H"}
    type_encoded = encoder.transform([type_map[machine_type]])[0]

    input_data = pd.DataFrame([[type_encoded, air_temp, process_temp, rpm, torque, tool_wear]],
                               columns=['Type', 'Air temperature', 'Process temperature',
                                       'Rotational speed', 'Torque', 'Tool wear'])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.error("⚠️ FAILURE RISK DETECTED")
        else:
            st.success("✅ MACHINE HEALTHY")

    with col2:
        st.metric("Failure Probability", f"{probability:.1%}")

    with col3:
        risk_level = "HIGH" if probability > 0.5 else "MEDIUM" if probability > 0.2 else "LOW"
        st.metric("Risk Level", risk_level)

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Probability %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "orange" if probability > 0.2 else "green"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 50], 'color': "lightyellow"},
                {'range': [50, 100], 'color': "lightcoral"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("👈 Adjust the sensor values in the sidebar and click 'Analyze Machine Health'")