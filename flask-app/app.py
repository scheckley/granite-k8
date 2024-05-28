from flask import Flask, render_template, request, jsonify
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Initialize the tokenizer
model = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-3b-code-base")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3b-code-base")

# URL of the IBM Granite LLM service
LLM_SERVICE_URL = 'https://granite-k8-scp012-dxm01.apps.ocp.osprey.hartree.stfc.ac.uk/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['PUT'])
def query():
    user_query = request.form['query']
    response = requests.put(LLM_SERVICE_URL, json={'query': user_query})
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to get response from LLM service'}), response.status_code

@app.route('/api/query', methods=['PUT'])
def api_query():
    data = request.json
    if 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    user_query = data['query']

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)

    # Tokenize the query
    tokens = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Send the tokenized query to the LLM service
    response = requests.put(LLM_SERVICE_URL, json={'query': tokens.tolist()})
    
    if response.status_code == 200:
        result = response.json()
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to get response from LLM service'}), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8443, debug=True)
