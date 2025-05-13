# Libraries
import streamlit as st
import torch
import pandas as pd
import json
import time

# Utilities
from utils.helpers import preprocess_url, is_valid_url, load_model, process_url
from utils.vae_cnn import VAE_CNN

# --------------------------
# Streamlit App Setup
# --------------------------
st.set_page_config(page_title="PHISHABILITY", page_icon="🕵️‍♀️")

# Header
st.markdown("<h1 style='text-align: center; color: #CF7914;'>🕵️‍♀️ PHISHABILITY</h1>", unsafe_allow_html=True)
st.divider()
st.markdown("<h3 style='text-align: center; color: #FFFF;'>URL-based Phishing Detection with VAE-CNN</h3>", unsafe_allow_html=True)

# Styling
st.markdown(
    """
    <style>
    /* Footer Styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0E1117; 
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #F4F2EF; 
        z-index: 9999; 
    }
    .footer a {
        color: #CF7914; 
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }

    .main .block-container {
        padding-bottom: 60px; 
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #213E60; 
    }
    [data-testid="stSidebar"] h2 {
        color: #CF7914;
    }
    </style>
    <div class="footer">
        <div class="footer-content">
            <span class="footer-logo">©</span>
            <span>2025. Developed wholeheartedly by <a href="https://ygsyportfolio.vercel.app" target="_blank">YGSYP</a></span>
        </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
threshold = st.sidebar.number_input(
    "Detection Threshold",
    min_value=0.0000,
    max_value=0.0100,
    value=0.0049,
    step=0.0001,
    format="%.4f"
)
with st.sidebar.expander("🎯 Why 0.0049?"):
    st.info("""
    The default threshold of 0.0049 was chosen based on comprehensive evaluation to achieve an optimal balance between:
            
    🔍 False Positives: Legitimate URLs wrongly detected as phishing
    🔐 False Negatives: Phishing URLs incorrectly labeled as safe

    - Raising the threshold makes the model more cautious — fewer phishing alerts, but higher risk of missing actual threats (↑ false negatives).
    - Lowering the threshold makes the model more aggressive — more phishing URLs are caught, but at the cost of flagging more safe URLs (↑ false positives).
            
Optimized using **Precision, Recall, and F1-score**.
""")
st.sidebar.markdown("<h2 style='color: #CF7914;'>📘About PHISHABILITY</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
**PHISHABILITY** is a deep learning-based phishing URL detector using hybrid model **VAE-CNN**.

### How It Works:
1. The model uses a **Variational Autoencoder (VAE)** to reconstruct input URLs.
2. Legitimate URLs are usually reconstructed with **low reconstruction loss** (i.e., they resemble normal patterns).
3. Phishing URLs tend to result in **higher loss**, indicating anomalies.
4. A **Convolutional Neural Network (CNN)** is used to extract features from the URL, enhancing the model's ability to detect phishing attempts.
5. If the reconstruction loss **exceeds a certain threshold**, the URL is flagged as **possibly phishing**.
""")

# Load vocab and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('models/vocabulary.json') as f:
    char_to_idx = {k: int(v) for k, v in json.load(f).items()}
vocab_size = len(char_to_idx) + 1
max_len = 100
model = load_model(device, vocab_size)

# Main Section
option = st.radio("Choose Detection Mode:", ("Manual URL Entry", "CSV Upload"))

if option == "Manual URL Entry":
    input_url = st.text_input("🔗 Enter a URL to detect:")
    if st.button("🔍 Detect"):
        if not input_url:
            st.warning("Please enter a URL.")
        elif not is_valid_url(input_url):
            st.error("Invalid format. Example: https://ygsyp.com")
        else:
            start_time = time.time() 
            with st.spinner("Analyzing URL... Please wait."):
                loss = process_url(input_url, model, char_to_idx, max_len, vocab_size, device)
            end_time = time.time()  

            # Calculate detection time
            detection_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Display Results
            st.success("Detection complete!")
            st.write(f"🧮 Loss: `{loss:.6f}`")
            st.write(f"⏱️ Detection Time: `{detection_time:.2f} ms`")

            if loss > threshold:
                st.error(f"🚨 **Possibly PHISHING** (Threshold: {threshold:.4f})")
                st.markdown(
                    f"""
                    The reconstruction loss for this URL is **{loss:.6f}**, which is **greater than the threshold** of **{threshold:.4f}**. 
                    This indicates that the URL exhibits patterns that deviate significantly from legitimate URLs, 
                    making it more likely to be a phishing attempt.
                    """
                )
            else:
                st.success(f"✅ **Possibly LEGITIMATE** (Threshold: {threshold:.4f})")
                st.markdown(
                    f"""
                    The reconstruction loss for this URL is **{loss:.6f}**, which is **less than or equal to the threshold** of **{threshold:.4f}**. 
                    This suggests that the URL closely resembles legitimate patterns, making it less likely to be a phishing attempt.
                    """
                )

elif option == "CSV Upload":
    uploaded_file = st.file_uploader("📁 Upload a CSV with a 'URL' column", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
            if "URL" not in df.columns:
                st.error("CSV must contain a column named 'URL'.")
            else:
                results = []
                phishing_count = 0
                legitimate_count = 0

                # Start timing for the entire CSV processing
                total_start_time = time.time()

                # Initialize progress bar
                progress_bar = st.progress(0)
                total_urls = len(df)

                for index, url in enumerate(df["URL"]):
                    start_time = time.time()  # Start timing for each URL
                    if not is_valid_url(url):
                        results.append({
                            "URL": url,
                            "Loss": None,
                            "Prediction": "Invalid URL",
                            "Detection Time (ms)": None,
                            "Explanation": "The URL format is invalid and could not be analyzed."
                        })
                    else:
                        loss = process_url(url, model, char_to_idx, max_len, vocab_size, device)
                        end_time = time.time()  # End timing for each URL

                        # Calculate detection time for each URL
                        detection_time = (end_time - start_time) * 1000  # Convert to milliseconds

                        # Prediction
                        prediction = "PHISHING" if loss > threshold else "LEGITIMATE"
                        if prediction == "PHISHING":
                            phishing_count += 1
                        else:
                            legitimate_count += 1

                        # Explanation
                        explanation = (
                            f"The reconstruction loss for this URL is **{loss:.6f}**, which is "
                            f"{'greater than' if loss > threshold else 'less than or equal to'} the threshold of **{threshold:.4f}**. "
                            f"This suggests that the URL is {'more likely to be a phishing attempt' if loss > threshold else 'less likely to be a phishing attempt'}."
                        )

                        results.append({
                            "URL": url,
                            "Loss": loss,
                            "Prediction": prediction,
                            "Detection Time (ms)": round(detection_time, 2),
                            "Explanation": explanation
                        })

                    # Update progress bar
                    progress = (index + 1) / total_urls
                    progress_bar.progress(progress)

                # End timing for the entire CSV processing
                total_end_time = time.time()
                total_duration = (total_end_time - total_start_time)  # Total duration in seconds

                # Create DataFrame
                result_df = pd.DataFrame(results)

                # Display Results
                st.dataframe(result_df)

                # Display Summary
                st.markdown(f"### 📊 Detection Summary")
                st.write(f"**Total URLs Processed:** {len(df)}")
                st.write(f"**Legitimate URLs:** {legitimate_count}")
                st.write(f"**Phishing URLs:** {phishing_count}")
                st.write(f"**Total Detection Time:** {total_duration:.2f} seconds")

                # Download Button
                st.download_button(
                    "📥 Download Results",
                    result_df.to_csv(index=False).encode('utf-8'),
                    "phishing_detection_results.csv",
                    "text/csv"
                )
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV file: {e}")