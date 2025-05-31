import torch
import numpy as np
import re
import json
from utils.vae_cnn import VAE_CNN

def preprocess_url(url, char_to_idx, max_len=100, vocab_size=87):
    encoded = np.zeros((max_len, vocab_size), dtype=np.float32)
    for i, ch in enumerate(url[:max_len]):
        if ch in char_to_idx:
            encoded[i, char_to_idx[ch]] = 1.0
    return torch.tensor(encoded).unsqueeze(0).unsqueeze(0)

def is_valid_url(url):
    pattern = re.compile(
        r'^(https?://)?'                       # Optional http/https
        r'(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|'    # Domain name
        r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|'  # OR IPv4 address
        r'xn--[a-zA-Z0-9]+(\.[a-zA-Z]{2,})?)'  # OR punycode domain
        r'(:\d+)?'                             # Optional port
        r'(\/[^\s]*)?'                         # Optional path
        r'(\?[^\s]*)?$'                        # Optional query string
    )
    return re.match(pattern, url) is not None

def load_model(device, vocab_size, latent_dim=8):
    model = VAE_CNN(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.eval()
    return model

def process_url(url, model, char_to_idx, max_len, vocab_size, device):
    url_tensor = preprocess_url(url, char_to_idx, max_len, vocab_size).to(device)
    with torch.no_grad():
        recon, _, _ = model(url_tensor)
        return torch.mean((recon - url_tensor) ** 2).item()