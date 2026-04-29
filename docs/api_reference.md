# 🛠️ NewsBot 2.0 API Reference

This document provides complete technical documentation for the functions, classes, and web endpoints within the NewsBot 2.0 Intelligence System.

---

## 🌐 Web API Endpoints (Flask)

The system operates a backend web service to power the interactive dashboard. All requests should be sent with `Content-Type: application/json`.

### `POST /analyze`
Processes a single news article for comprehensive NLP analysis.
* **Payload:** `{"text": "string"}`
* **Returns:**
    * `classification`: Predicted category and confidence score.
    * `sentiment`: Polarity and subjectivity metrics.
    * `entities`: List of identified people, organizations, and locations.
    * `topics`: Identified themes from the Topic Discovery Engine.
    * `summary`: AI-generated executive summary.

### `POST /query`
Handles natural language interactions via the Conversational Interface.
* **Payload:** `{"query": "string"}`
* **Returns:** `{"response": "string"}` — An AI-generated answer based on the analyzed news corpus.

---

## 🧠 Core Analytical Classes (src/analysis)

### `TopicModeler`
Implements unsupervised learning (LDA/NMF) to discover hidden trends.
* **`fit_transform(documents)`**: Trains the model on a provided list of text strings.
* **`get_topic_words(topic_id, n_words=10)`**: Returns the most significant keywords for a specific discovered trend.
* **`visualize_topics()`**: Generates data structures for interactive topic clustering.

### `SentimentAnalyzer`
Tracks emotional tone and temporal evolution.
* **`get_sentiment_metrics(text)`**: Returns polarity scores calibrated to a ±0.05 threshold.
* **`track_temporal_evolution(data)`**: Calculates mean sentiment changes over a chronological sequence of articles.

### `EntityRelationshipMapper`
Performs Named Entity Recognition (NER).
* **`extract_entities(text)`**: Identifies and labels key actors (PER, ORG, GPE).
* **`map_relationships(doc)`**: Analyzes syntactic dependencies to link entities to specific events or actions.

---

## 🌍 Multilingual Intelligence (src/multilingual)

### `MultilingualEngine`
Handles cross-language analysis and translation workflows.
* **`detect_language(text)`**: Returns the ISO language code for the provided input.
* **`translate_to_english(text)`**: Routes foreign-language content through a stable translation wrapper for processing.
* **`cross_lingual_compare(source_a, source_b)`**: Analyzes coverage differences between different language sources on the same event.

---

## 💬 Conversational Interface (src/conversation)

### `QueryProcessor`
The primary controller for natural language interaction.
* **`process(user_query)`**: Orchestrates intent detection and retrieves relevant analysis to formulate a response.
* **`maintain_context(state)`**: Manages the conversation state for multi-part user questions.

### `IntentClassifier`
Determines the user's objective behind a query.
* **`predict_intent(text)`**: Categorizes the query (e.g., "Summarization Request," "Sentiment Inquiry," "Topic Search").

---

## 📝 Data Processing (src/data_processing)

### `TextPreprocessor`
The enhanced cleaning pipeline evolved from the midterm foundation.
* **`clean(text)`**: Performs normalization, stop-word removal, and lemmatization.
* **`validate(data)`**: Ensures data quality standards are met before analysis.

### `FeatureExtractor`
Converts text into numerical representations for machine learning.
* **`get_embeddings(text)`**: Generates semantic vectors using transformer-based models.
* **`get_tfidf_features(corpus)`**: Returns the TF-IDF matrix for statistical analysis.

---

**Author:** Brandon Matias  
**Version:** 2.0 (2026 Production Build)  
**Institution:** Houston City College | AI & Robotics
