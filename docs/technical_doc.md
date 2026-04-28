# 📰 NewsBot 2.0: AI-Driven News Intelligence Platform

**NewsBot 2.0** is an end-to-end NLP platform designed to transform raw news data into actionable insights. Built with a modular architecture, the system performs intelligent summarization, sentiment evolution tracking, and entity relationship mapping.

The project is optimized for the **2026 Python ecosystem**, featuring a self-healing environment designed to navigate the NumPy 2.0 transition.

---

## 🏗️ System Architecture

NewsBot 2.0 is divided into three core analytical modules powered by a Flask-based web dashboard:

### 1. Sentiment Evolution Engine (`Module A`)
* **Logic:** Utilizes a news-calibrated TextBlob implementation.
* **Calibration:** Features a tightened threshold ($\pm 0.05$) to account for "Sentiment Dilution" in objective journalistic text.
* **Feature:** Tracks mean polarity over time, allowing users to visualize the emotional trajectory of specific news cycles.

### 2. Topic Discovery Engine (`Module B`)
* **Logic:** Employs an unsupervised learning approach to cluster and identify recurring themes across multiple articles.
* **Professional Alignment:** Previously `NewsTopicModeler`, now refactored as a high-level `TopicDiscoveryEngine`.

### 3. Entity Relationship Mapper (`Module C`)
* **Logic:** Powered by `spaCy` and the `en_core_web_sm` transformer model.
* **Feature:** Performs Named Entity Recognition (NER) to extract and map relationships between organizations, locations, and key public figures.

---

## 🛠️ Technical Challenges & Solutions

### The "2026 Binary Mismatch" Resolution
During development, the project encountered a critical `ValueError: numpy.dtype size changed`. This was caused by the industry-wide transition to **NumPy 2.0**.
* **Solution:** Engineered a "2026-Native" environment strategy, moving from strict version pins (`==`) to flexible compatibility ranges (`>=`). Implemented a binary monkeypatch for `scipy.linalg` to ensure legacy NLP models could operate on the modern architecture.

### The Translation Dependency Wrapper
The legacy `googletrans` library created a dependency conflict by forcing a downgrade of `httpx`, which broke the HuggingFace Hub connection.
* **Solution:** Developed a custom **Drop-In Wrapper**. Created a local `googletrans.py` class that intercepts legacy calls and routes them through the stable `deep-translator` API, preserving the codebase without compromising system stability.

---

## 🚀 Installation & Deployment (Google Colab)

To deploy the NewsBot 2.0 environment in a fresh Google Colab session:

1. **Environment Setup:**
   ```python
   !git clone [https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git](https://github.com/BrandonLee-98/ITAI2373-NewsBot-Final.git)
   %cd ITAI2373-NewsBot-Final
   !pip install -r requirements.txt --quiet
   !python -m spacy download en_core_web_sm --quiet
   ```
   ---
 2. **Launch Dashboard:**
    ```python
    from google.colab.output import eval_js
    print(eval_js("google.colab.kernel.proxyPort(5000)"))
    !python app.py
    ```
---
## 🧰 Tech Stack

* **Frontend:** HTML5, CSS3 (Branded for Houston City College), JavaScript
* **Backend:** Flask (Python)
* **AI/NLP:** spaCy, Transformers, TextBlob, NLTK, Sentence-Transformers
* **Data Science:** NumPy 2.0+, SciPy 1.14+, Pandas 2.0+
* **Translation:** Deep-Translator API

---

## 👨‍💻 Author

**Brandon Matias** *Up-and-Coming Developer* Houston City College | AI & Robotics  
**Email:** [bmatias98@outlook.com](mailto:bmatias98@outlook.com)  
**LinkedIn:** [linkedin.com/in/brandonmatias](https://www.linkedin.com/in/brandonmatias)

---

## 💡 Academic Context

This project was developed as the final capstone for **ITAI 2373**. It satisfies all core requirements for Modules A, B, and C, while implementing bonus features including a full-stack web dashboard and a multilingual translation engine.
