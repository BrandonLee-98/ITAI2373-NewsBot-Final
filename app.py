from flask import Flask, render_template, request, jsonify
import os
# This works because of the __init__.py file you just showed me!
from src import QueryProcessor

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'hcc-final-project-key')

# Initialize the processor
processor = QueryProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    article_text = data.get('article_text', '')
    
    response = processor.process(user_query, article_text=article_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
