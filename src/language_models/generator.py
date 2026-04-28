# This section contains the code for the NewsBotIntelligenceSystem class, which encapsulates the entire pipeline for processing new articles, including preprocessing, sentiment analysis, classification, entity extraction, and insight generation.
class NewsBotIntelligenceSystem:
    """
    Complete NewsBot Intelligence System

    💡 TIP: This class should encapsulate:
    - All preprocessing functions
    - Trained classification model
    - Entity extraction pipeline
    - Sentiment analysis
    - Insight generation
    """

    def __init__(self, classifier, vectorizer, sentiment_analyzer, pos_feature_names):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.sentiment_analyzer = sentiment_analyzer
        self.nlp = nlp  # spaCy model
        self.pos_feature_names = pos_feature_names # Store POS feature names

    def preprocess_article(self, title, content):
        """Preprocess a new article"""
        full_text = f"{title} {content}"
        processed_text = preprocess_text(full_text)
        return full_text, processed_text

    def classify_article(self, title, processed_text, full_text, sentiment):
        """Classify article category"""
        # 🚀 YOUR CODE HERE: Implement classification
        # Step 1: TF-IDF features
        features_tfidf = self.vectorizer.transform([processed_text]).toarray()

        # Step 2: Sentiment features (use the already calculated sentiment)
        article_sentiment_features = np.array([[sentiment['compound'], sentiment['pos'], sentiment['neu'], sentiment['neg']]])

        # Step 3: Length features
        article_length_features = np.array([[len(full_text), len(full_text.split()), len(title)]])

        # Step 4: POS features
        new_article_pos_proportions = analyze_pos_patterns(full_text) # Returns a dict

        # Create a zero vector for all possible POS tags seen during training
        article_pos_features_vector = np.zeros(len(self.pos_feature_names))

        # Fill the vector with proportions from the new article
        for i, tag_name in enumerate(self.pos_feature_names):
            article_pos_features_vector[i] = new_article_pos_proportions.get(tag_name, 0.0)

        article_pos_features = np.array([article_pos_features_vector]) # Reshape to (1, num_pos_features)


        # Combine all features
        features_combined = np.hstack([
            features_tfidf,
            article_sentiment_features,
            article_length_features,
            article_pos_features
        ])

        # Predict category and probability
        prediction = self.classifier.predict(features_combined)[0]
        probabilities = None
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_combined)[0]
            class_probs = dict(zip(self.classifier.classes_, probabilities))
        else:
            # Handle LinearSVC case where predict_proba is not available
            # This is a heuristic approximation using decision_function scores
            decision_scores = self.classifier.decision_function(features_combined)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores)) # subtract max for numerical stability
            probabilities = exp_scores / exp_scores.sum()
            class_probs = dict(zip(self.classifier.classes_, probabilities))


        return prediction, class_probs

    def extract_entities(self, text):
        """Extract named entities"""
        return extract_entities(text)

    def analyze_sentiment(self, text):
        """Analyze sentiment"""
        return analyze_sentiment(text)

    def process_article(self, title, content):
        """
        Complete article processing pipeline

        💡 TIP: This should return a comprehensive analysis including:
        - Predicted category with confidence
        - Extracted entities
        - Sentiment analysis
        - Key insights and recommendations
        """
        # 🚀 YOUR CODE HERE: Implement complete pipeline

        # Step 1: Preprocess
        full_text, processed_text = self.preprocess_article(title, content)

        # Step 2: Analyze sentiment (moved up as it's needed for classify_article)
        sentiment = self.analyze_sentiment(full_text)

        # Step 3: Classify
        category, category_probs = self.classify_article(title, processed_text, full_text, sentiment)

        # Step 4: Extract entities
        entities = self.extract_entities(full_text)

        # Step 5: Generate insights
        insights = self.generate_insights(category, entities, sentiment, category_probs)

        return {
            'title': title,
            'content': content[:200] + '...' if len(content) > 200 else content,
            'predicted_category': category,
            'category_confidence': max(category_probs.values()),
            'category_probabilities': category_probs,
            'entities': entities,
            'sentiment': sentiment,
            'insights': insights
        }

    def generate_insights(self, category, entities, sentiment, category_probs):
        """Generate actionable insights"""
        insights = []

        # Classification insights
        confidence = max(category_probs.values())
        if confidence > 0.8:
            insights.append(f"✅ High confidence {category} classification ({confidence:.2%})")
        else:
            insights.append(f"⚠️ Uncertain classification - consider manual review")

        # Sentiment insights
        if sentiment['compound'] > 0.1:
            insights.append(f"😊 Positive sentiment detected ({sentiment['compound']:.3f})")
        elif sentiment['compound'] < -0.1:
            insights.append(f"😞 Negative sentiment detected ({sentiment['compound']:.3f})")
        else:
            insights.append(f"😐 Neutral sentiment ({sentiment['compound']:.3f})")

        # Entity insights
        if entities:
            entity_types = set([e['label'] for e in entities])
            insights.append(f"🔍 Found {len(entities)} entities of {len(entity_types)} types")

            # Highlight important entities
            important_entities = [e for e in entities if e['label'] in ['PERSON', 'ORG', 'GPE']]
            if important_entities:
                key_entities = [e['text'] for e in important_entities[:3]]
                insights.append(f"🎯 Key entities: {', '.join(key_entities)}")
        else:
            insights.append("ℹ️ No named entities detected")

        return insights

# Get POS feature names from the training data for consistent feature generation
pos_feature_names = pos_df.drop(columns=['article_id', 'category']).columns.tolist()

# Initialize the complete system
newsbot = NewsBotIntelligenceSystem(
    classifier=best_model,
    vectorizer=tfidf_vectorizer,
    sentiment_analyzer=sia,
    pos_feature_names=pos_feature_names
)

print("🤖 NewsBot Intelligence System initialized!")
print("✅ Ready to process new articles")

