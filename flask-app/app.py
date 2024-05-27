from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Replace this with the actual URL of your IBM Granite LLM service
LLM_SERVICE_URL = 'https://granite-k8-scp012-dxm01.apps.ocp.osprey.hartree.stfc.ac.uk/api/query'

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
