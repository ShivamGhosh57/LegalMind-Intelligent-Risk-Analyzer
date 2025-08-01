import os
import io
import json
import threading
import asyncio

import requests
import PyPDF2
import streamlit as st
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Backend (Flask API) ---
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import logging

flask_app = Flask(__name__)

async def call_gemini_api(prompt, api_key):
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-2.0-flash:generateContent?key={api_key}"
    )
    concise_prompt = (
        prompt +
        "\n\nIMPORTANT: Keep the Suggestion concise (under 100 words)."
    )
    payload = {"contents":[{"role":"user","parts":[{"text":concise_prompt}]}]}
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: requests.post(api_url, json=payload, headers={'Content-Type':'application/json'})
    )
    if resp.status_code == 200:
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    elif resp.status_code == 400:
        return "Invalid API Key or malformed request."
    else:
        return f"AI error: status {resp.status_code}"

@flask_app.route("/predict_clause", methods=['POST'])
def predict_clause():
    try:
        payload = request.get_json()
        clause = payload['contract_text']
        # Get API key from environment variables
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY not found in .env file."}), 400

        pipeline = PredictionPipeline()
        result = pipeline.predict(clause)

        if result["risk_level"] in ['high','medium']:
            suggestion = asyncio.run(
                call_gemini_api(result["suggestion_prompt"], api_key)
            )
            result["suggestion"] = suggestion
        else:
            result["suggestion"] = (
                "No suggestions needed for a low-risk clause."
            )
        result.pop("suggestion_prompt", None)
        return jsonify(result)
    except Exception as e:
        logging.error(f"/predict_clause error: {e}")
        return jsonify({"error":str(e)}), 500

@flask_app.route("/analyze_document", methods=['POST'])
def analyze_document():
    try:
        payload = request.get_json()
        full_text = payload['contract_text']

        # split by sentences, filter out very short ones
        clauses = [
            s.strip() for s in full_text.split('.')
            if len(s.strip()) > 20
        ]

        pipeline = PredictionPipeline()
        risky = []
        for c in clauses:
            res = pipeline.predict(c)
            if res['risk_level'] in ['high','medium']:
                risky.append({
                    "clause": c,
                    "risk_level": res['risk_level']
                })

        # sort high → medium → low
        risky.sort(
            key=lambda x: (x['risk_level']=='high', x['risk_level']=='medium'),
            reverse=True
        )
        return jsonify({"risky_clauses": risky[:10]})
    except Exception as e:
        logging.error(f"/analyze_document error: {e}")
        return jsonify({"error":str(e)}), 500

def run_flask():
    flask_app.run(port=5001)


# --- Frontend (Streamlit) ---

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    full = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # collapse newlines and extra spaces
        cleaned = " ".join(text.split())
        full.append(cleaned)
    return " ".join(full)

def run_streamlit():
    st.set_page_config(
        page_title="Litigation Risk Analyzer",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("AI-Powered Litigation Risk Analyzer ⚖️")
    st.markdown(
        "Analyze clauses or upload a PDF to spot high‑risk language "
        "and get AI‑powered rewrites."
    )

    # Check if the API key is available and show a persistent warning if not.
    if not os.getenv("GEMINI_API_KEY"):
        st.warning("Gemini API Key not found. Please create a `.env` file with `GEMINI_API_KEY='your-key'` to enable AI suggestions.", icon="⚠️")


    tab1, tab2 = st.tabs(
        ["Single Clause", "Full Document (PDF)"]
    )

    with tab1:
        clause_input = st.text_area(
            "Enter a contract clause:", height=150
        )
        if st.button("Analyze Clause"):
            if not clause_input:
                st.warning("Please enter clause text.")
            else:
                with st.spinner("Analyzing…"):
                    resp = requests.post(
                        "http://127.0.0.1:5001/predict_clause",
                        json={"contract_text": clause_input}
                    )
                if resp.ok:
                    data = resp.json()
                    risk = data.get("risk_level", "Unknown")
                    suggestion = data.get("suggestion", "N/A")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Predicted Risk", risk.capitalize()
                        )
                        if risk=='high':
                            st.error("High litigation risk!")
                        elif risk=='medium':
                            st.warning("Moderate risk.")
                        else:
                            st.success("Low risk.")

                    with col2:
                        st.subheader("AI Suggestion")
                        st.markdown(suggestion)
                else:
                    st.error(f"API error: {resp.json().get('error', resp.text)}")

    with tab2:
        uploaded = st.file_uploader(
            "Upload PDF", type="pdf"
        )
        if st.button("Analyze Document"):
            if uploaded is None:
                st.warning("Please upload a PDF.")
            else:
                with st.spinner("Extracting and analyzing…"):
                    text = extract_text_from_pdf(uploaded)
                    resp = requests.post(
                        "http://127.0.0.1:5001/analyze_document",
                        json={"contract_text": text}
                    )
                if resp.ok:
                    st.session_state['risky'] = resp.json().get('risky_clauses', [])
                    st.success(
                        f"Found {len(st.session_state['risky'])} risky clauses."
                    )
                else:
                    st.error(f"API error: {resp.text}")

    # Initialize suggestions dict if not present
    if 'suggestions' not in st.session_state:
        st.session_state['suggestions'] = {}

    # display document results
    if st.session_state.get('risky'):
        st.markdown("---")
        st.header("Risky Clauses Identified")
        for idx, item in enumerate(st.session_state['risky']):
            # collapse newlines & wrap
            clause = item['clause'].replace('\n',' ').strip()
            color = "#dc3545" if item['risk_level']=='high' else "#fd7e14"
            st.markdown(
                f"""
                <div style="
                    background:#f8f9fa;
                    border-left:5px solid {color};
                    padding:10px;
                    margin-bottom:10px;
                    white-space:normal;
                    overflow-x:auto;
                ">
                  {clause}
                </div>
                """, unsafe_allow_html=True
            )

            # Button to fetch suggestion (only if not already fetched)
            if idx not in st.session_state['suggestions']:
                if st.button(
                    f"Get AI Suggestion for clause {idx+1}",
                    key=f"sugg_{idx}"
                ):
                    with st.spinner("Generating suggestion…"):
                        resp = requests.post(
                            "http://127.0.0.1:5001/predict_clause",
                            json={"contract_text": clause}
                        )
                        if resp.ok:
                            st.session_state['suggestions'][idx] = resp.json().get('suggestion','')
                        else:
                            st.session_state['suggestions'][idx] = f"API error: {resp.json().get('error', resp.text)}"

            # If we have a suggestion stored, display it
            if idx in st.session_state['suggestions']:
                st.info(st.session_state['suggestions'][idx])

if __name__ == "__main__":
    # start Flask in background thread
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    # launch Streamlit UI
    run_streamlit()
