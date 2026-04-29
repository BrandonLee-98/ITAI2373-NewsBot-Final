from flask import Flask, render_template, request, jsonify, session

# Core Analysis Imports
from src.analysis.classifier import NewsClassifier
from src.analysis.topic_modeler import TopicModeler
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.ner_extractor import EntityRelationshipMapper

# Language Models & Conversation Imports
from src.language_models.summarizer import Summarizer
from src.conversation.query_processor import QueryProcessor

app = Flask(__name__)
app.secret_key = 'newsbot-2026-secure-key'

# Initialize all NLP components separately to maintain modular design
classifier = NewsClassifier()
topic_modeler = TopicModeler()
sentiment_analyzer = SentimentAnalyzer()
entity_mapper = EntityRelationshipMapper()
summarizer = Summarizer()
query_processor = QueryProcessor()

@app.route('/')
def dashboard():
    """Renders the main frontend dashboard."""
    return render_template('dashboard.html')

@app.route('/analyze', methods=['POST'])
def analyze_article():
    """Processes a single article through the entire NLP pipeline."""
    data = request.json
    article_text = data.get('text', '')

    if not article_text:
        return jsonify({'error': 'No text provided'}), 400

    # Perform comprehensive analysis using the specialized modules
    results = {
        'classification': classifier.predict(article_text),
        'sentiment': sentiment_analyzer.get_sentiment_metrics(article_text),
        'entities': entity_mapper.extract_entities(article_text),
        'topics': topic_modeler.get_article_topics(article_text) if hasattr(topic_modeler, 'get_article_topics') else topic_modeler.get_topic_words(0), 
        'summary': summarizer.summarize(article_text)
    }
    
    return jsonify(results)

@app.route('/query', methods=['POST'])
def process_query():
    """Handles natural language interaction via the conversational interface."""
    data = request.json
    user_query = data.get('query', '')
    
    if not user_query:
         return jsonify({'error': 'No query provided'}), 400

    # Process the user's intent and generate a response
    response = query_processor.process(user_query)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    # Set host to '0.0.0.0' to ensure it runs correctly within Google Colab
    app.run(host='0.0.0.0', port=5000, debug=True)
