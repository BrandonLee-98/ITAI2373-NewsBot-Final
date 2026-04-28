# 📰 NewsBot 2.0

An intelligent AI-driven news analysis platform that provides summarization, sentiment evolution tracking, and entity mapping via a professional web dashboard.

---

## 🚀 Quick Start (Google Colab)

To launch the NewsBot 2.0 environment:

1. **Setup:**
   ```python
   !git clone [https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git](https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git)
   %cd ITAI2373-NewsBot-Final
   !pip install -r requirements.txt --quiet
   !python -m spacy download en_core_web_sm --quiet

2. **Launch:**
   ```python
   from google.colab.output import eval_js
   print(eval_js("google.colab.kernel.proxyPort(5000)"))
   !python app.py
---

## ✨ Key Features

* **Interactive Dashboard:** Full-stack Flask UI branded for Houston City College.
* **Sentiment Evolution:** Tracks news emotional trajectory with a calibrated $\pm 0.05$ threshold.
* **Multilingual Support:** Auto-detection and translation of global news sources.
* **2026 Ready:** Native support for NumPy 2.0 and the modern AI ecosystem.

---

## 📚 Documentation

For a deep dive into the architecture, dependency resolutions (NumPy 2.0 / Translation Wrapper), and module logic, please see:
👉 **[TECHNICAL_DOCUMENTATION.md](./technical_doc.md)**

---

## 🧰 Tech Stack

* **Backend:** Flask, Python 3.12+
* **AI/NLP:** spaCy, Transformers, TextBlob, NLTK
* **Data:** NumPy 2.0+, SciPy 1.14+, Pandas 2.0+

---

## 👨‍💻 Author

**Brandon Matias** | Houston City College  
**Email:** [bmatias98@outlook.com](mailto:bmatias98@outlook.com)  
**LinkedIn:** [linkedin.com/in/brandonmatias](https://www.linkedin.com/in/brandonmatias)

---

## 💡 Academic Context

Final capstone project for **ITAI 2373**. Satisfies all requirements for Modules A, B, and C, with bonus implementations for web deployment and multilingual processing.

---

## 📂 Repository Structure
| Folder | Content Description |
| :--- | :--- |
| `/src` | Core NLP packages, intelligence modules, and data processing. |
| `/docs` | Technical specifications, API reference, and user guides. |
| `/notebooks` | Experimental and development notebooks for system validation. |
| `/reports` | Executive Summary and Final Technical Report (PDF). |
| `/tests` | Unit tests for preprocessing, classification, and system integration. |
| `/templates` | HTML templates for the Flask web application interface. |
| `/models` | Directory for storing pre-trained model files (.pkl). |
