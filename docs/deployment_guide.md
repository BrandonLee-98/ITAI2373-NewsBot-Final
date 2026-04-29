# 🚀 NewsBot 2.0 Deployment Guide

This document provides comprehensive instructions for deploying the **NewsBot 2.0 Intelligence System** across various environments. Whether you are running a quick demonstration in Google Colab or deploying a containerized version for production, follow the steps below.

---

## 🛠️ Prerequisites

Before deployment, ensure you have the following:
* **Python 3.12+** environment.
* **Git** installed for repository cloning.
* **Hardware:** Minimum 8GB RAM (16GB recommended for Transformer models).
* **API Access:** Valid keys for translation or LLM services (if applicable).

---

## ☁️ Option 1: Google Colab (Development & Demo)

Google Colab is the recommended environment for academic reviews and live demonstrations.

1. **Environment Initialization:**
   Run the following block to clone the repository and install all required dependencies:
   ```python
   !git clone [https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git](https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git)
   %cd ITAI2373-NewsBot-Final
   !pip install -r requirements.txt
   !python -m spacy download en_core_web_sm
   ```

2. **Configuration Setup:**

   Ensure config/settings.py is configured for your desired analytical thresholds.


4. **Launch the Flask Dashboard:**
   ```python
   from google.colab.output import eval_js
   print(eval_js("google.colab.kernel.proxyPort(5000)"))
   !python app.py


---

## 🏗️ Option 2: Production Deployment (Professional)

For professional-grade scaling, NewsBot 2.0 supports modern deployment workflows.

### A. Heroku (Free Tier/PaaS)
* **Best for:** Small team sharing and simple web access.
* **Process:**
    1.  Ensure a `Procfile` exists in the root directory: `web: gunicorn app:app`.
    2.  Link your GitHub repository to your Heroku dashboard.
    3.  Deploy the `main` branch.

### B. Docker (Containerized Deployment)
* **Best for:** Scalability and ensuring "Write Once, Run Anywhere" consistency.
* **Process:**
    1.  Build the image: `docker build -t newsbot-2.0 .`
    2.  Run the container: `docker run -p 5000:5000 newsbot-2.0`

### C. Streamlit (Rapid Prototyping Alternative)
* **Best for:** Data-heavy internal tools.
* **Process:**
    1.  Replace the Flask `app.py` entry point with a Streamlit interface.
    2.  Deploy via Streamlit Cloud for instant URL generation.

---

## ⚙️ Configuration Management

The system behavior is controlled via `config/settings.py`.

* **API Keys:** Never commit real keys to GitHub. Use the provided `config/api_keys_template.txt` as a reference to set up your environment variables.
* **Model Paths:** Ensure the `data/models/` path is write-accessible if you are training custom LDA or NMF models locally.
* **Thresholds:** The `SENTIMENT_THRESHOLD` is currently calibrated to **0.05** to handle the objective nature of news text without "Sentiment Dilution."

---

## 🛡️ Troubleshooting

* **Binary Mismatches:** If you encounter `numpy.dtype` errors, ensure your environment is running NumPy 2.0+ as per the project's 2026 technical requirements.
* **Model Loading:** If spaCy fails to load, verify the `en_core_web_sm` model was downloaded correctly during the setup phase.
* **Port Conflicts:** If port 5000 is in use, modify the `app.run()` parameters in `app.py` to a different port (e.g., 8080).

---

**Author:** Brandon Matias  
**Project:** NewsBot Intelligence System 2.0  
**Course:** ITAI 2373 | Houston City College
