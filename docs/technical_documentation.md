# 🛠️ Technical Documentation: NewsBot Intelligence System 2.0
**Project:** NewsBot Intelligence System 2.0  
**Course:** ITAI 2373 - Advanced NLP  
**Developer:** Brandon Matias  
**Institution:** Houston City College  
**Date:** April 2026

---

## 1. System Architecture Overview
The NewsBot 2.0 is a production-ready news analysis platform designed to demonstrate mastery of advanced NLP techniques. The architecture consists of four integrated modules designed for deep text understanding, multilingual analysis, and intelligent content generation.

### Core Modules
* **Module A: Advanced Content Analysis Engine**
    * Multi-level categorization with confidence scoring.
    * Automatic identification of emerging themes via Topic Discovery.
    * Emotional tone tracking and sentiment evolution.
    * Entity Relationship Mapping between people, organizations, and events.
* **Module B: Language Understanding and Generation**
    * Abstractive and accurate article summarization.
    * Content enhancement using contextual information.
    * Identification of key findings and narrative patterns.
* **Module C: Multilingual Intelligence**
    * Automatic language detection for global sources.
    * Translation integration for seamless content access.
    * Cross-language analysis for regional perspective comparisons.
* **Module D: Conversational Interface**
    * Natural language query handling via extractive QA.
    * Interactive exploration of specific topics or entities.
    * Generation of on-demand reports and visualizations.

---

## 2. NLP Model Specifications
The system leverages state-of-the-art transformer architectures and statistical algorithms to fulfill technical requirements:
* **Zero-Shot Classification:** Utilizes NLI-based models (`distilbart-mnli`) for multi-level categorization without the need for pre-defined training labels.
* **Topic Modeling:** Implements Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF) for content discovery and trend analysis.
* **Summarization:** Employs `distilbart-cnn-12-6` for abstractive text generation, synthesizing new sentences rather than simply extracting them.
* **Conversational QA:** Uses a `minilm-uncased-squad2` model for context-aware natural language processing.
* **Named Entity Recognition (NER):** Extracts people, organizations, and locations via spaCy and Transformer pipelines.

---

## 3. Repository Structure
The project follows a modular design to ensure a clear separation of concerns and system maintainability.

```text
ITAI2373-NewsBot-Final/
├── README.md               # Project overview and individual contribution summary
├── requirements.txt        # All dependencies with specific versions
├── config/                 # Configuration and setting management
│   ├── settings.py         # Global parameters and paths
│   └── api_keys_template.txt
├── src/                    # Main source code
│   ├── data_processing/    # Preprocessing and feature extraction
│   ├── analysis/           # Classification, Sentiment, NER, and Topic Modeling
│   ├── language_models/    # Summarization and generation logic
│   ├── multilingual/       # Translation and cross-lingual services
│   └── conversation/       # Natural language query processing
├── notebooks/              # Experimental design and module walkthroughs
├── tests/                  # Unit tests for core functions and integration
├── models/                 # Serialized trained model files (.joblib)
├── results/                # Analysis outputs and saved visualizations
├── reports/                # Professional PDF documentation
└── docs/                   # Markdown-native documentation
```
---
## 4. API Reference & Core Logic
The system implements professional-grade Python classes to handle automated data flow.

### TopicModeler (`src/analysis/topic_modeler.py`)
* **Purpose:** Discover hidden topics in unstructured news content.
* **`fit_transform(documents)`:** Trains the statistical model and transforms the corpus into a topic distribution.
* **`get_topic_words(topic_id)`:** Retrieves top weighted terms to define discovered themes.

### NewsClassifier (`src/analysis/classifier.py`)
* **Purpose:** Enhanced multi-label classification system.
* **`predict(text)`:** Returns the highest-probability category labels with associated confidence scores.
* **`get_sentiment(text)`:** Analyzes emotional tone changes and polarity.

### QueryProcessor (`src/conversation/query_processor.py`)
* **Purpose:** Handles natural language query interaction.
* **`process(query)`:** Performs intent detection and generates responses grounded in the article context.

## 5. Performance Optimization & Quality Standards
* **Modular Design:** Clear separation of concerns with reusable components for feature extraction and analysis.
* **Data Validation:** Rigorous quality checks on input text to handle noise and formatting issues.
* **Persistence:** Custom-trained components are serialized to the `models/` directory to optimize inference speed and reduce retraining overhead.
* **Testing:** A comprehensive unit testing suite ensures the reliability of the preprocessing, classification, and integration modules.

---
*Developed by Brandon Matias | Houston City College | ITAI 2373 Final Project Submission*
