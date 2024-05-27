from flask import Flask, render_template, request, jsonify
import requests
from transformers import AutoTokenizer

app = Flask(__name__)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# URL of the IBM Granite LLM service
LLM_SERVICE_URL = 'https://xxx'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    response = requests.post(LLM_SERVICE_URL, json={'query': user_query})
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to get response from LLM service'}), response.status_code

@app.route('/api/query', methods=['POST'])
def api_query():
    data = request.json
    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']

    # Tokenize the query
    tokens = tokenizer.encode(user_query, return_tensors='pt')

    # Send the tokenized query to the LLM service
    response = requests.post(LLM_SERVICE_URL, json={'query': tokens.tolist()})
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to get response from LLM service'}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, debug=True)
