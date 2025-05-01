import streamlit as st
import torch
import json
import numpy as np
import pandas as pd
import time

from utils.preprocessing import preprocess_url
from utils.vae_model import VAE_CNN

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary
with open('models/vocabulary_tested.json') as f:
    char_to_idx = json.load(f)
char_to_idx = {k: int(v) for k, v in char_to_idx.items()}
vocab_size = len(char_to_idx) + 1
max_len = 100

# Load model
model = VAE_CNN(latent_dim=8).to(device)
model.load_state_dict(torch.load('models/Best_Model.pth', map_location=device))
model.eval()

# Threshold slider
st.set_page_config(page_title="PHISHABILITY", page_icon="🕵️‍♀️")
st.title("🔒 PHISHABILITY - Phishing URL Detector")

st.sidebar.header("⚙️ Pengaturan Deteksi")
threshold = st.sidebar.slider("Set Threshold", min_value=0.001, max_value=0.01, value=0.0049, step=0.0001)

# Mode Input
option = st.radio("Pilih mode deteksi:", ("Input URL Manual", "Upload CSV"))

if option == "Input URL Manual":
    input_url = st.text_input("Masukkan URL untuk dianalisis:")
    if st.button("🔍 Deteksi"):
        if input_url:
            url_tensor = preprocess_url(input_url, char_to_idx, max_len, vocab_size).to(device)

            start_time = time.time()
            with torch.no_grad():
                recon, _, _ = model(url_tensor)
                loss = torch.mean((recon - url_tensor) ** 2).item()
            end_time = time.time()

            st.write(f"🔧 Loss: `{loss:.6f}`")
            st.write(f"⏱️ Waktu Deteksi: `{(end_time - start_time)*1000:.2f} ms`")

            if loss > threshold:
                st.error("🚨 Deteksi: Kemungkinan **PHISHING**")
            else:
                st.success("✅ Deteksi: Kemungkinan **LEGITIMATE**")

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Unggah file CSV berisi kolom URL", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "URL" not in df.columns:
            st.warning("CSV harus memiliki kolom bernama 'URL'")
        else:
            results = []
            with torch.no_grad():
                for url in df["URL"]:
                    try:
                        tensor = preprocess_url(url, char_to_idx, max_len, vocab_size).to(device)
                        recon, _, _ = model(tensor)
                        loss = torch.mean((recon - tensor) ** 2).item()
                        pred = "PHISHING" if loss > threshold else "LEGITIMATE"
                        results.append({"URL": url, "Loss": loss, "Prediction": pred})
                    except:
                        results.append({"URL": url, "Loss": None, "Prediction": "Error"})
            result_df = pd.DataFrame(results)
            st.dataframe(result_df)
            st.download_button("📥 Download Hasil", result_df.to_csv(index=False), file_name="phishability_results.csv")
