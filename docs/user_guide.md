# 📘 User Guide: NewsBot Intelligence System 2.0
**Project:** NewsBot Intelligence System 2.0  
**Course:** ITAI 2373 - Advanced NLP  
**Developer:** Brandon Matias | Houston City College

---

## 1. Introduction
Welcome to **NewsBot 2.0**, your personal AI-driven news analyst. In an era of information overload, this tool is designed to cut through the noise by providing high-speed, automated intelligence. Whether you are tracking market trends, tech breakthroughs, or legal updates, NewsBot 2.0 handles the reading, translating, and summarizing so you can focus on the big picture.

## 2. Key Capabilities
This platform is more than just a search tool; it is a full-scale intelligence engine:
* **Instant Categorization:** Automatically tags articles into categories like Finance, Tech, Politics, Health, or Legal.
* **Smart Summarization:** Provides concise, human-like summaries that capture the "core message" of long-form articles.
* **Language Barrier Removal:** Supports automated detection and translation of Spanish, French, and German news into English.
* **Interactive Chat:** Allows you to ask follow-up questions directly to the article for specific data points.

---

## 3. Getting Started
To begin using the NewsBot 2.0 dashboard, follow these simple steps:

### Launching the Dashboard
1. **Access the Environment:** Open the project in Google Colab or your designated local environment.
2. **Install Dependencies:** Ensure all required libraries are loaded by running the setup cells or executing `pip install -r requirements.txt`.
3. **Run the App:** Execute the `app.py` script. A local or proxy URL will be generated.
4. **Open the Interface:** Click the link to open the web-based Intelligence Dashboard.

## 4. Usage Instructions
### Analyzing a Single Article
1. **Locate an Article:** Find a news story you want to analyze (in English, Spanish, French, or German).
2. **Copy & Paste:** Copy the text and paste it into the "Article Input" box on the dashboard.
3. **Process:** Click the **"Run NLP Pipeline"** button.
4. **Review Results:** Within seconds, the dashboard will populate with the Category, Sentiment, Executive Summary, and discovered Topic clusters.

### Using the Conversational QA
On the right-hand side of the dashboard, you will find the **NewsBot Chat** window:
* **Ask a Question:** Type a specific question about the article you just analyzed (e.g., "What was the total investment mentioned?" or "Who is the primary spokesperson?").
* **Get Answers:** The bot will extract the precise answer from the text and display it instantly.

---

## 5. Frequently Asked Questions (FAQ)

**Q: Which news sources work best with NewsBot 2.0?**
A: The system is optimized for high-quality, text-heavy news articles. It works best with structured reporting from sources like Reuters, AP, or specialized technical journals.

**Q: Do I need a high-end GPU to run this system?**
A: No. NewsBot 2.0 uses "Distilled" transformer models that are highly efficient. It is designed to run smoothly on standard Google Colab CPUs or modern laptop hardware.

**Q: Is my pasted text saved permanently?**
A: By default, the system processes text in real-time. Analysis results are only saved if you explicitly use the "Export" feature to save findings to the `results/` folder.

**Q: Can I add support for more languages?**
A: Yes. The system is built with a modular translation pivot. Additional languages can be added by updating the configuration in `src/multilingual/translator.py`.

**Q: Why does the Topic Modeling (LDA) require longer articles?**
A: Statistical topic modeling requires a sufficient "bag of words" to identify meaningful patterns. For articles under 50 words, the system may prioritize general categorization over specific topic clusters.

---

## 6. Troubleshooting
* **"No Data Found":** Ensure you have clicked the "Run NLP Pipeline" button before trying to use the chatbot.
* **Translation Lag:** If processing a foreign language article, it may take 1-2 additional seconds for the translation engine to verify the source.
* **Visualizations Not Loading:** Ensure your browser allows scripts from the dashboard URL, as interactive charts require JavaScript to render.

---
*Developed by Brandon Matias | Houston City College | ITAI 2373 Final Project Submission*
