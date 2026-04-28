from flask import Flask, render_template, request, redirect, url_for
import os
import sys

# Ensure Python can see the 'src' folder
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import your modular logic
# This assumes your src/__init__.py exposes QueryProcessor
try:
    from src import QueryProcessor
except ImportError:
    # Fallback if the __init__.py isn't set up to export it directly
    from src.conversation.query_processor import QueryProcessor

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hcc-final-project-2026')

# Initialize the modular engine once when the app starts
processor = QueryProcessor()

@app.route('/')
def index():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the form submission from index.html."""
    # 1. Get the article text from the HTML form
    article_text = request.form.get('article_text', '')
    
    if not article_text.strip():
        return redirect(url_for('index'))

    # 2. Process the content using your modular backend
    # We default the intent to 'summarize' for the main button
    summary_result = processor.process("summarize", article_text=article_text)
    
    # 3. Get sentiment/tone metrics for the results page
    # This calls your SentimentAnalyzer via the QueryProcessor
    sentiment_result = processor.process("analyze", article_text=article_text)

    # 4. Render the results page with the data
    return render_template(
        'results.html',
        summary=summary_result,
        sentiment_text=sentiment_result,
        original_text=article_text
    )

if __name__ == '__main__':
    # Threaded=True allows the app to handle multiple requests in Colab
    app.run(host='0.0.0.0', port=5000, debug=True)
