# 📰 NewsBot Intelligence System 2.0
**Advanced NLP Integration and Analysis Platform**

**Author:** Brandon Matias  
**Institution:** Houston City College  
**Contact:** bmatias98@outlook.com | [LinkedIn](www.linkedin.com/in/brandonmatias)

---

## 📖 Comprehensive Project Overview

Welcome to my NewsBot 2.0! As an up-and-coming software developer, this project serves as my final capstone, transforming a foundational script into a production-ready news analysis platform. It demonstrates the practical application of advanced Natural Language Processing (NLP) techniques integrated into a cohesive, user-friendly web dashboard. 

## ✨ Core Features & Modules
This system integrates several specialized NLP pipelines:

* **Module A: Advanced Content Analysis Engine**
  * **Enhanced Classification:** Categorizes articles dynamically using Hugging Face's `distilbart-mnli` Zero-Shot model with multi-level confidence scoring.
  * **Topic Discovery:** Utilizes `scikit-learn` Latent Dirichlet Allocation (LDA) to extract hidden statistical themes from the text.
  * **Entity Relationship Mapping:** Extracts key People, Organizations, and Locations using `spaCy`.
  * **Sentiment Analysis:** Calculates the emotional tone using calibrated polarity thresholds.
* **Module B: Language Understanding and Generation**
  * **Intelligent Summarization:** Generates abstractive, human-like executive summaries using a dedicated `distilbart-cnn` transformer model.
* **Module C: Multilingual Intelligence**
  * **Language Detection:** Automatically identifies the source language of the pasted article using `langdetect`.
  * **Translation Integration:** Seamlessly translates non-English text to English using `deep-translator` to ensure the NLP pipeline can accurately process global news.
* **Module D: Conversational Interface**
  * **Natural Language Queries:** Features an extractive Question Answering (QA) chatbot powered by `minilm-uncased-squad2` that reads the article context to answer specific user questions interactively.
* **Bonus: Web Application Frontend**
  * A complete Single Page Application (SPA) built with Flask, HTML/JS, and Bootstrap to provide an interactive intelligence dashboard.

---

## 🚀 Colab Deployment Instructions

To run the NewsBot Intelligence System 2.0 in Google Colab, create two separate code cells and run them in order.

### Step 1: Environment Setup
This block downloads the repository, installs all necessary machine learning libraries, and downloads the required NLP models.

```python
import os
import shutil

# Target GitHub repository
repo_name = "ITAI2373-NewsBot-Final"
repo_url = "https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git" 

# Clean up previous runs to avoid version conflicts
if os.path.exists(repo_name):
    shutil.rmtree(repo_name)

# Clone the fresh repository
!git clone {repo_url}

# Enter the project directory
%cd {repo_name}

# Install dependencies and NLP models
if os.path.exists("requirements.txt"):
    print("Installing dependencies... This may take a minute.")
    !pip install -r requirements.txt --quiet
    
    # Ensure all advanced NLP libraries are installed
    !pip install deep-translator langdetect scikit-learn transformers torch --quiet
    
    # Download language dictionaries
    !python -m spacy download en_core_web_sm --quiet
    !python -m textblob.download_corpora --quiet
    print("✅ Environment ready!")
else:
    print("❌ Error: requirements.txt not found. Check repository structure.")
```
### Step 2: Launch the Dashboard
This block ensures the environment is in the correct working directory, generates a secure web link, and starts the Flask server safely.

```python
import os
import subprocess
from google.colab.output import eval_js
from IPython.display import clear_output

# Force Colab to enter the project folder
%cd /content/ITAI2373-NewsBot-Final/

print("🚀 Booting up the NewsBot 2.0 Server...")
print("⏳ Loading NLP Models (This takes about 25-35 seconds, please wait)...")

# Generate the public viewing link
proxy_link = eval_js("google.colab.kernel.proxyPort(5000)")

# Launch the app in the background so we can monitor its startup
process = subprocess.Popen(['python', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Read the background logs until we see Flask announce it is running
for line in iter(process.stdout.readline, ''):
    if 'Running on' in line:
        break 
    if 'Error' in line or 'Traceback' in line:
        print(line, end='')

# Clear all the messy loading text
clear_output()

# Print the clean, prominent link
print("=" * 65)
print("✅ SYSTEM ONLINE AND READY!")
print(f"🌐 Click here to open the NLP NewsBot 2.0 (May need time to fully load): {proxy_link}")
print("=" * 65)
print("\nLive Server Logs (Leave this cell running while you test):")

for line in iter(process.stdout.readline, ''):
    print(line, end='')
```
---

## 📁 Repository Structure

```
ITAI2373-NewsBot-Final/
├── README.md               # Comprehensive project overview
├── requirements.txt        # All dependencies with versions
├── app.py                  # Flask web application frontend
├── src/                    # Source code for all NLP modules
│   ├── analysis/           # Classification, Sentiment, NER, and Topic Modeling
│   ├── language_models/    # Text Summarization logic
│   ├── multilingual/       # Translation and Language Detection Services
│   └── conversation/       # Natural language query handling
├── templates/              # HTML frontend files
│   └── dashboard.html      # User Interface
└── docs/                   # Documentation folder
    └── individual_contributions.md # Individual contribution summary
```
---
## 🤝 Individual Contributions
**Brandon Matias**: Project Lead. Designed and implemented the complete NLP pipeline (Zero-Shot Classification, Abstractive Summarization, NER, LDA Topic Modeling), integrated the Hugging Face Extractive QA conversational interface, implemented multilingual detection and translation architectures, and developed the Flask/Bootstrap web dashboard.
