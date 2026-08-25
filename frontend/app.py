"""
Streamlit frontend.

Run with:
    streamlit run frontend/app.py

Assumes the FastAPI backend is running at http://127.0.0.1:8000
(start it separately with: uvicorn backend.main:app --reload)
"""

import base64
import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/detect"

st.set_page_config(page_title="Image Forgery Detection", layout="centered")
st.title("AI-Driven Multi-Technique Image Forgery Detection")
st.caption("Upload an image to check whether it's authentic or forged.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze", type="primary"):
        with st.spinner("Running detection techniques..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the backend. Start it with: `uvicorn backend.main:app --reload`")
                st.stop()

        if response.status_code != 200:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        else:
            result = response.json()

            with col2:
                highlighted_bytes = base64.b64decode(result["highlighted_image_base64"])
                st.image(Image.open(io.BytesIO(highlighted_bytes)),
                          caption="Highlighted result", use_container_width=True)

            verdict = result["verdict"]
            confidence = result["confidence"]

            if verdict == "Forged":
                st.error(f"**Verdict: {verdict}**  (confidence: {confidence*100:.1f}%)")
            else:
                st.success(f"**Verdict: {verdict}**  (confidence: {confidence*100:.1f}%)")

            st.subheader("Per-technique scores")
            st.json(result["technique_scores"])

            st.subheader("How much each technique influenced the decision")
            st.json(result["technique_contributions"])
