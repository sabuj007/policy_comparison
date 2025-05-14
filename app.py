import streamlit as st
import os
import pandas as pd
import tempfile
import subprocess
import json
from openai import OpenAI
from io import BytesIO
import re
import numpy as np

# Configuration - Set your OpenAI API key here
OPENAI_API_KEY = "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"
MODEL_NAME = "gpt-4o"
MAX_TEXT_LENGTH = 15000


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdftotext or fallback to PyPDF2."""
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        return result.stdout
    except Exception as e:
        st.warning(f"pdftotext failed, trying PyPDF2 fallback: {e}")
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            return "\n".join(page.extract_text() for page in reader.pages)
        except Exception as fallback_e:
            st.error(f"Both pdftotext and PyPDF2 failed: {fallback_e}")
            return None


def clean_and_standardize_data(data):
    """Clean and standardize the extracted data."""
    if not isinstance(data, dict):
        return data

    for field in ['coverage_limits', 'key_deductibles']:
        if field in data:
            if isinstance(data[field], dict):
                for v in data[field].values():
                    if isinstance(v, (int, float)):
                        data[field] = v
                        break
                    elif isinstance(v, str):
                        numbers = re.findall(r'\d+', v.replace(',', ''))
                        if numbers:
                            data[field] = int(numbers[0])
                            break
                else:
                    data[field] = np.nan
            elif isinstance(data[field], str):
                numbers = re.findall(r'\d+', data[field].replace(',', ''))
                data[field] = int(numbers[0]) if numbers else np.nan
            elif not isinstance(data[field], (int, float)):
                data[field] = np.nan

    if 'annual_premium' in data and isinstance(data['annual_premium'], str):
        numbers = re.findall(r'\d+', data['annual_premium'].replace(',', ''))
        if numbers:
            data['annual_premium'] = float(numbers[0])
        else:
            data['annual_premium'] = np.nan

    for key in data:
        if isinstance(data[key], list) and not data[key]:
            data[key] = []
        elif data[key] == "" or data[key] is None:
            data[key] = np.nan

    return data


def format_for_display(value):
    """Format values for display in the DataFrame"""
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            return "None"
        return ", ".join(str(v) for v in value)
    if pd.isna(value):
        return "None"
    if isinstance(value, (int, float)):
        return f"{value:,}" if value >= 1000 else str(value)
    return str(value)


def get_structured_data(policy_text):
    if not policy_text:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)
    processed_text = policy_text[:MAX_TEXT_LENGTH]

    prompt = f"""
    Analyze the policy text and extract key comparison factors.
    Return ONLY a valid JSON object with these fields:
    - policy_type (string)
    - coverage_limits (number)
    - key_deductibles (number) 
    - annual_premium (number)
    - key_exclusions (array)
    - notable_endorsements (array)

    Policy Text:
    ```
    {processed_text}
    ```
    Return ONLY valid JSON:
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You extract clean, numeric-only values."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        json_string = response.choices[0].message.content.strip()
        json_string = re.sub(r'^```json\s*|\s*```$', '', json_string, flags=re.MULTILINE)
        data = json.loads(json_string)
        return clean_and_standardize_data(data)
    except Exception as e:
        st.error(f"Error processing policy: {e}")
        return None


def get_recommendation(df_markdown, preferences):
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
    Analyze this insurance policy data and preferences.
    Provide recommendation in this format:

    POLICY RECOMMENDATION: [Policy Name]

    BEST CHOICE:
    [1-2 sentences]

    KEY ADVANTAGES:
    1. Cost:
       - Premium: [amount]
       - Deductible: [amount]
    2. Coverage:
       - [Key benefit]

    COMPARISON TO OTHER OPTIONS:
    - [Policy]: [difference]

    IMPORTANT NOTES:
    - [Note]

    Policy Data:
    {df_markdown}

    Customer Preferences:
    {preferences}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You provide clean text recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        recommendation = response.choices[0].message.content.strip()
        return '\n\n'.join([para.strip() for para in recommendation.split('\n\n') if para.strip()])
    except Exception as e:
        return f"Error generating recommendation: {e}"


# Streamlit UI
st.set_page_config(page_title="Insurance Policy Comparator", layout="wide")
st.title("Insurance Policy Comparator")

with st.sidebar:
    st.header("Configuration")
    preferences = st.text_area(
        "Customer Preferences",
        placeholder="e.g., Need low deductible, must include flood coverage",
        height=150
    )
    uploaded_files = st.file_uploader(
        "Upload Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

if uploaded_files and preferences:
    all_data = []
    with st.status("Processing policies..."):
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                text = extract_text_from_pdf(tmp_path)
                os.unlink(tmp_path)

                if text:
                    structured = get_structured_data(text)
                    if structured:
                        structured["filename"] = uploaded_file.name
                        all_data.append(structured)
                        st.success(f"Processed {uploaded_file.name}")
                    else:
                        st.warning(f"Could not extract data from {uploaded_file.name}")
                else:
                    st.warning(f"Could not read text from {uploaded_file.name}")

    if all_data:
        df = pd.DataFrame(all_data)
        cols = ['filename'] + [col for col in df.columns if col != 'filename']
        df = df[cols]

        st.subheader("Policy Comparison")
        display_df = df.copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(format_for_display)

        # Clean table display without extra boxes
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        csv_bytes = BytesIO()
        df.fillna("NA").to_csv(csv_bytes, index=False)
        st.download_button(
            "Download Data",
            data=csv_bytes.getvalue(),
            file_name="policy_comparison.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("Recommendation")

        with st.spinner("Generating recommendation..."):
            recommendation = get_recommendation(display_df.to_markdown(index=False), preferences)

        st.text_area(
            "Recommendation Details",
            value=recommendation,
            height=300,
            label_visibility="hidden"
        )

        st.success("Analysis complete!")
    else:
        st.warning("No valid policy data extracted")
else:
    st.info("Please upload PDFs and enter preferences to begin")