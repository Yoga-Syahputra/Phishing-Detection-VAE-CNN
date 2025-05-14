# URL-based Phishing Detection Using Variational Autoencoder and Convolutional Neural Network (VAE-CNN) 

## Table of Contents 

- [Abstract](#abstract)
- [About This Project](#about-this-project)
  - [Theoretically](#theoretically-)
  - [Practically](#practically-)
- [Dataset Details](#dataset-details-)
- [VAE-CNN Model Architecture](#vae-cnn-model-architecture)
  - [Layer Details](#layer-details)
  - [Model Summary](#model-summary)
  - [Memory Estimate](#memory-estimate)
- [Folder Details](#folder-details)
- [How to Run the App](#how-to-run-the-app)
- [Evaluation Metrics](#evaluation-metrics)
- [Authors](#authors)
- [Publication](#publication)
- [Contact](#contact)

## Abstract 

### English 

Phishing is a dangerous cybersecurity threat that deceives users by disguising malicious content as legitimate. This study suggests an unsupervised anomaly detection using Variational Autoencoder (VAE) and Convolutional Neural Network (CNN) to identify phishing URLs. The model utilizes CNN as an encoder for URL feature extraction, applies probabilistic latent space sampling via the reparameterization trick and reconstruct the input through a decoder. The model was trained on 7.000 legitimate URLs and tested on 3.000 URLs (1.500 legitimate and 1500 phishing) from an imbalanced dataset of 10.000 samples. Evaluation results demonstrate the potential of VAE-CNN in enhancing phishing detection through unsupervised learning and contribute to developing cybersecurity risk control strategies.


### Indonesian

Phishing merupakan ancaman berbahaya dalam keamanan siber yang mengelabui pengguna dengan menyamarkan konten berbahaya sebagai sesuatu yang sah. Penelitian ini mengusulkan metode deteksi anomali berbasis pembelajaran tidak terawasi dengan mengimplementasikan model hybrid deep learning yang menggabungkan Variational Autoencoder (VAE) dan Convolutional Neural Network (CNN) untuk mengidentifikasi URL phishing. Model menggunakan CNN sebagai encoder untuk mengekstraksi fitur URL, melalukan sampling ruang laten secara probabilistik melalui reparameterization trick, dan merekonstruksi input melalui decoder. Model dilatih menggunakan 7.000 URL legitimate dan diuji pada 3.000 URL (1.500 legitimate dan 1.500 phishing) dari total 10.000 data yang memiliki distribusi tidak seimbang. Hasil evaluasi menunjukkan performa yang seimbang dengan nilai presisi, recall, dan F1-score sebesar 0,82 pada ambang batas optimal 0,0049 berdasarkan analisis reconstruction loss. Temuan ini menunjukkan potensi model VAE-CNN dalam meningkatkan deteksi phishing melalui pembelajaran tidak terawasi dan berkontribusi terhadap pengembangan strategi pengendalian risiko keamanan siber.

---

## About This Project 

### Theoretically 

This project was submitted as part of the undergraduate thesis in Informatics Engineering at Universitas Maritim Raja Ali Haji (UMRAH). The primary goal in this project is to develop a hybrid deep learning model for URL-based phishing detection by combining Variational Autoencoder (VAE), which is effective in capturing latent representations, with Convolutional Neural Network (CNN), known for the ability in extracting local patterns in data. The model applies an unsupervised anomaly detection approach, where it is trained solely on legitimate URLs to calculate reconstruction loss and then tested using both legitimate and phishing URLs to evaluate the detection performance. The testing process includes comparing the model's predictions against ground truth labels to evaluate its realibility. 

### Practically 

To simulate the practical application of the model, a user-friendly interface named **PHISHABILITY** was built using the Streamlit framework in Python. The app allows real-time phishing detection and provides two modes of input for flexibilty and usability:
1. Manual URL Entry - Users can input a single URL for immediate detection (non-URL inputs will be rejected with a warning)
2. CSV File Upload - Users can upload a CSV file containing multiple URLs, allowing for efficient batch processing and detection.

In addition, users can manually configure the detection **threshold** via the sidebar to customize the sensitivity. By default, the threshold is set to the most optimal value derived from model evaluation to ensure a balanced trade-off between precision and recall (see the optimal threshold here: [Evaluation Metrics](#evaluation-metrics)). 

---

## Dataset Details 

- Total URLs: **10,000**
  - **7,000 legitimate** (used for training)
  - **1,500 legitimate + 1,500 phishing** (used for evaluation/testing)
  - Sources include publicly available datasets for legitimate URLs (collected from [PhishDataset](https://github.com/ESDAUNG/PhishDataset)) and phishing URLs. The phishing URLs were gathered from the [URLScan.io Phishing URL Feed](https://urlscan.io/pricing/phishingfeed/) and crawled from the X social media platform (formerly Twitter) using [Tweet Harvest](https://github.com/helmisatria/tweet-harvest).
---

## VAE-CNN Model Architecture 

### Layer Details

The model was developed using the PyTorch library in Python. Below is a detailed summary of the encoder and decoder layers used in the VAE-CNN architecture:

| Layer (Type)           | Output Shape            |  Parameters   |
|------------------------|-------------------------|---------------|
| Conv2d-1               | `[1, 32, 50, 43]`       | 544           |
| BatchNorm2d-2          | `[1, 32, 50, 43]`       | 64            |
| ReLU-3                 | `[1, 32, 50, 43]`       | 0             |
| Conv2d-4               | `[1, 64, 25, 21]`       | 32,832        |
| BatchNorm2d-5          | `[1, 64, 25, 21]`       | 128           |
| ReLU-6                 | `[1, 64, 25, 21]`       | 0             |
| Conv2d-7               | `[1, 128, 12, 10]`      | 131,200       |
| BatchNorm2d-8          | `[1, 128, 12, 10]`      | 256           |
| ReLU-9                 | `[1, 128, 12, 10]`      | 0             |
| Linear-10 *(μ)*        | `[1, 4]`                | 61,444        |
| Linear-11 *(σ²)*       | `[1, 4]`                | 61,444        |
| Linear-12              | `[1, 15360]`            | 76,800        |
| ConvTranspose2d-13     | `[1, 64, 24, 20]`       | 131,136       |
| BatchNorm2d-14         | `[1, 64, 24, 20]`       | 128           |
| ReLU-15                | `[1, 64, 24, 20]`       | 0             |
| ConvTranspose2d-16     | `[1, 32, 48, 40]`       | 32,800        |
| BatchNorm2d-17         | `[1, 32, 48, 40]`       | 64            |
| ReLU-18                | `[1, 32, 48, 40]`       | 0             |
| ConvTranspose2d-19     | `[1, 1, 100, 87]`       | 1281          |

### Model Summary 

- **Total Parameters**: `530,121`
- **Trainable Parameters**: `530,121`
- **Non-trainable Parameters**: `0`

### Memory Estimate 

| Description              | Size      |
|--------------------------|-----------|
| Input size               | `0.03 MB` |
| Forward/Backward pass    | `4.99 MB` |
| Params size              | `2.02 MB` |
| Estimated Total Size     | `7.04 MB` |

---

## Folder Details 

### /app

This is the main directory for the detection app built using Streamlit.

### /data

This directory contains the dataset, organized into two different subdirectories:
1. /data/dev - Contains the primary dataset used for the model development, including legitimate and phishing URLs
2. /data/test - Contains various samples used for simulation in the app via the CSV file upload

### /models

This directory contains the best VAE-CNN model, along with the vocabulary file generated during the preprocessing

### /notebook

This directory includes two subdirectories:
1. /notebook/models - Contains the **Jupyter Notebook** used for developing the VAE-CNN model, with multiple experiment setups
2. /notebook/utils - Contains the **Jupyter Notebook** files for data preparation and preprocessing

### /utils

This directory consists of VAE-CNN model architecture and helper scripts used in the app preprocessing

---

## How to Run the App 

1. Clone the repository:
```bash
git clone https://github.com/Yoga-Syahputra/Phishing-Detection-VAE-CNN.git
cd Phishing-Detection-VAE-CNN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Input the URL manually or via upload CSV file to begin detection!

---

## Evaluation Metrics 

| Metric     | Value |
|------------|-------|
| Precision  | 0.82  |
| Recall     | 0.82  |
| F1-Score   | 0.82  |
| Threshold  | 0.0049 (Optimal @ 82nd percentile) |

---

## Authors 

| Name | Role | Link |
|------|------|------|
| **Yoga Syahputra** | Author / Developer | [LinkedIn](linkedin.com/in/ygsyp) |
| **Hendra Kurniawan, S.Kom., M.Sc.Eng., Ph.D.** | Advisor | [LinkedIn](https://www.linkedin.com/in/hendra-kurniawan-583140179/) • [Google Scholar](https://scholar.google.co.id/citations?user=qWtmdGwAAAAJ&hl=en) |
| **Novrizal Fattah Famitra, S.Kom., M.Kom.** | Co-advisor | [LinkedIn](https://linkedin.com/in/novrizal-fahmitra/)  • [Google Scholar](https://scholar.google.com/citations?user=XB58YosAAAAJ&hl=id) |

---

## Publication 

```
a = "Coming out very soon!"
print(a)
```

## Contact 

Questions, suggestions, or collaboration are welcome! Feel free to reach out!

- ✉️ Email: ygsyp01@gmail.com

- 🌐 [Portfolio](https://ygsyportfolio.vercel.app)
