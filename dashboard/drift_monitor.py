# dashboard/drift_monitor.py - Lightweight Streamlit drift dashboard

import streamlit as st
import requests
from config.settings import settings

st.title("GoldDiggr Drift Monitor")

assets = requests.get("http://localhost:8000/metrics").json()["assets"]
for a in assets:
    st.subheader(a)
    # fetch /metrics or custom drift endpoint - placeholder
    st.metric("Drift PSI", "0.08", delta="low")
    st.progress(0.12)  # visual