from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
import plotly.utils
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
from fraud_detector import CryptoFraudDetector
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize the fraud detector
fraud_detector = CryptoFraudDetector()

# Sample wallet addresses for demonstration
SAMPLE_WALLETS = [
    "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",  # Binance hot wallet
    "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance cold wallet
    "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",  # Another exchange wallet
    "0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",  # Binance hot wallet 2
    "0xD551234Ae421e3BCBA99A0Da6d736074f22192FF"   # Binance hot wallet 3
]

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

def clean_nans(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    else:
        return obj
@app.route('/api/analyze', methods=['POST'])
def analyze_wallet():
    """API endpoint to analyze a wallet"""
    try:
        data = request.get_json()
        print("Received data:", data)
        wallet_address = data.get('wallet_address', '').strip()
        print("Wallet address:", wallet_address)
        
        if not wallet_address:
            return jsonify({'error': 'Wallet address is required'}), 400
        
        # Analyze the wallet
        result = fraud_detector.analyze_wallet(wallet_address, save_results=True)
        
        if 'error' in result:
            return jsonify(result), 400
        
        # Convert Plotly figures to JSON for frontend
        if 'visualizations' in result:
            for viz_name, viz_obj in result['visualizations'].items():
                if hasattr(viz_obj, 'to_json'):
                    result['visualizations'][viz_name] = json.loads(viz_obj.to_json())
        
        result = remove_bools(result)
        result = clean_nans(result)  # <-- Clean NaN/inf values here
        
        return app.response_class(
            response=json.dumps(result, default=str),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
def remove_bools(obj):
    if isinstance(obj, bool):
        return None  # Remove bools by returning None (will be skipped in dicts)
    elif isinstance(obj, dict):
        return {k: remove_bools(v) for k, v in obj.items() if not isinstance(v, bool)}
    elif isinstance(obj, list):
        return [remove_bools(v) for v in obj if not isinstance(v, bool)]
    else:
        return obj

@app.route('/api/train', methods=['POST'])
def train_model():
    """API endpoint to train the model"""
    try:
        data = request.get_json()
        wallet_addresses = data.get('wallet_addresses', [])
        
        if not wallet_addresses:
            return jsonify({'error': 'At least one wallet address is required'}), 400
        
        # Train the model
        result = fraud_detector.train_model(wallet_addresses, save_model=True)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """API endpoint to analyze multiple wallets"""
    try:
        data = request.get_json()
        wallet_addresses = data.get('wallet_addresses', [])
        
        if not wallet_addresses:
            return jsonify({'error': 'At least one wallet address is required'}), 400
        
        # Analyze wallets in batch
        results = fraud_detector.batch_analyze(wallet_addresses)
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': f'Batch analysis failed: {str(e)}'}), 500

@app.route('/api/model-info')
def get_model_info():
    """Get information about the current model"""
    try:
        info = fraud_detector.get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load a trained model"""
    try:
        data = request.get_json()
        filepath = data.get('filepath', 'models/crypto_fraud_detector')
        
        fraud_detector.load_model(filepath)
        
        return jsonify({'message': 'Model loaded successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Failed to load model: {str(e)}'}), 500

@app.route('/api/sample-wallets')
def get_sample_wallets():
    """Get sample wallet addresses for testing"""
    return jsonify({'wallets': SAMPLE_WALLETS})

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate a text report for a wallet"""
    try:
        data = request.get_json()
        wallet_address = data.get('wallet_address', '').strip()
        
        if not wallet_address:
            return jsonify({'error': 'Wallet address is required'}), 400
        
        # Generate report
        report = fraud_detector.generate_report(wallet_address)
        
        return jsonify({'report': report})
        
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_trained': fraud_detector.is_trained
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=8080)
