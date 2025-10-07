# WalletScope - AI-Powered Crypto Fraud Detector

A comprehensive machine learning system for detecting suspicious cryptocurrency wallet activity using Ethereum blockchain data. This system combines advanced anomaly detection algorithms with real-time blockchain data analysis to identify potential fraud patterns.

## 🚀 Features

- **Real-time Blockchain Analysis**: Fetches transaction data from Ethereum blockchain via Etherscan API v2
- **Advanced ML Models**: Uses ensemble of Isolation Forest, One-Class SVM, and Autoencoder neural networks
- **Comprehensive Feature Engineering**: Extracts 22+ behavioral and statistical features from transaction data
- **Interactive Web Interface**: Modern, responsive dashboard with real-time visualizations
- **Single Wallet Analysis**: Detailed analysis of individual Ethereum addresses
- **Network Visualization**: Interactive graphs showing wallet interaction patterns
- **Time Series Analysis**: Track transaction patterns over time

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Flask Backend  │    │  ML Pipeline    │
│   (HTML/CSS/JS) │◄──►│   (REST API)    │◄──►│  (Python)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Etherscan API  │    │  Model Storage  │
                       │    (v2/api)     │    │  (Local Files)  │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- Flask web framework
- Machine learning libraries (scikit-learn, PyTorch)
- Data processing (pandas, numpy)
- Visualization (plotly, seaborn)
- Network analysis (networkx)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/satnamkhokhar/walletscope.git
   cd walletscope
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   ETHERSCAN_API_KEY=your_etherscan_api_key_here
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   ```

5. **Get an Etherscan API key**:
   - Visit [Etherscan](https://etherscan.io/apis)
   - Sign up for a free account
   - Generate an API key
   - Add it to your `.env` file

## 🚀 Quick Start

1. **Start the web application**:
   ```bash
   python3 app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Analyze a wallet**:
   - Enter an Ethereum wallet address (e.g., `0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6`)
   - Click "Analyze" to get fraud detection results
   - View interactive visualizations and risk assessment

## 📊 Usage Examples

### Command Line Usage

```python
from fraud_detector import CryptoFraudDetector

# Initialize the detector
detector = CryptoFraudDetector()

# Analyze a single wallet (models are pre-trained)
result = detector.analyze_wallet("0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")
print(f"Risk Level: {result['summary']['risk_level']}")
print(f"Anomaly Score: {result['summary']['anomaly_score']:.3f}")
```

### Web Interface Features

- **Single Wallet Analysis**: Enter any Ethereum address for detailed analysis
- **Interactive Visualizations**: 
  - Transaction timeline charts
  - Network interaction graphs
  - Feature analysis plots
  - Risk assessment metrics
- **Real-time Results**: Get instant fraud detection scores and explanations

## 🔍 Feature Engineering

The system extracts 22+ features from transaction data:

### Basic Statistics
- Total transactions count
- Unique counterparties
- Average/standard deviation of transaction values
- Maximum/minimum transaction values
- Total transaction volume

### Time-based Features
- Average time between transactions
- Transaction frequency (per day)
- Time pattern analysis

### Network Features
- New address interaction ratio
- Incoming vs outgoing transaction patterns
- Network connectivity metrics

### Behavioral Features
- Internal transaction ratio
- High-value transaction patterns
- Gas fee analysis
- Suspicious pattern scoring

## 🤖 Machine Learning Models

### 1. Isolation Forest
- Unsupervised anomaly detection
- Identifies outliers based on feature isolation
- Fast and scalable for large datasets

### 2. One-Class SVM
- Learns normal transaction patterns
- Flags deviations from learned behavior
- Robust to noise in training data

### 3. Autoencoder Neural Network
- Compresses transaction patterns to latent space
- Reconstruction error indicates anomalies
- Captures complex non-linear patterns

### Ensemble Approach
- Combines predictions from all three models
- Weighted voting for final anomaly score
- Improved accuracy and robustness

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Model parameters
ANOMALY_THRESHOLD = 0.8  # Threshold for flagging suspicious activity
MIN_TRANSACTIONS = 5     # Minimum transactions required for analysis

# API settings (Etherscan v2)
ETHERSCAN_API_KEY = "your_api_key"
ETHERSCAN_BASE_URL = 'https://api.etherscan.io/v2/api'
MAX_TRANSACTIONS = 1000  # Maximum transactions to fetch per wallet

# Feature engineering
TIME_WINDOW_DAYS = 30    # Analysis time window
```

## 📁 Project Structure

```
walletscope/
├── app.py                 # Flask web application
├── fraud_detector.py      # Main fraud detection class
├── data_fetcher.py        # Blockchain data retrieval (Etherscan v2)
├── feature_engineering.py # Feature extraction
├── anomaly_detector.py    # ML models implementation
├── visualizer.py          # Data visualization
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── wallets.json          # Sample wallet addresses
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── models/               # Pre-trained ML models
│   ├── crypto_fraud_detector_features.pkl
│   ├── crypto_fraud_detector_isolation_forest.pkl
│   ├── crypto_fraud_detector_one_class_svm.pkl
│   ├── crypto_fraud_detector_scaler.pkl
│   └── crypto_fraud_detector_training_data.json
└── README.md
```

## 🧪 Testing

### Sample Wallet Addresses

The system includes sample wallet addresses for testing:

- `0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6` (Binance hot wallet)
- `0x28C6c06298d514Db089934071355E5743bf21d60` (Binance cold wallet)
- `0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549` (Exchange wallet)

### API Testing

```bash
# Test the API endpoints
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"}'

# Check system health
curl http://localhost:5000/api/health

# Get model information
curl http://localhost:5000/api/model-info
```

## 🔒 Security Considerations

- **API Rate Limiting**: Respects Etherscan API v2 rate limits with automatic retries
- **Input Validation**: All wallet addresses are validated before processing
- **Error Handling**: Comprehensive error handling and logging
- **Data Privacy**: No sensitive data is stored permanently
- **Environment Variables**: Sensitive API keys stored in `.env` file (excluded from git)

## 🚨 Limitations

- **API Dependencies**: Requires active internet connection and valid Etherscan API key
- **Ethereum Only**: Currently supports only Ethereum blockchain
- **Historical Data**: Analysis limited to available transaction history
- **Rate Limits**: Subject to Etherscan API rate limiting (5 requests/second for free tier)
- **Pre-trained Models**: Uses existing trained models (training functionality available but not required)

## 🛠️ Technical Details

### API Endpoints

- `GET /api/health` - System health check
- `POST /api/analyze` - Analyze a single wallet
- `GET /api/model-info` - Get model information
- `POST /api/train` - Train models (optional)

### Dependencies

Key Python libraries used:
- **Flask & Flask-CORS**: Web framework and CORS handling
- **pandas & numpy**: Data processing and numerical computations
- **scikit-learn**: Machine learning algorithms (Isolation Forest, One-Class SVM)
- **torch**: Neural network implementation (Autoencoder)
- **plotly & seaborn**: Interactive visualizations and statistical plots
- **requests**: API communication with Etherscan
- **python-dotenv**: Environment variable management
- **json & datetime**: Data handling and time processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 🔮 Future Enhancements

- [ ] Support for multiple blockchains (Bitcoin, Polygon, BSC)
- [ ] Real-time monitoring and alert system
- [ ] Advanced network analysis (clustering, community detection)
- [ ] Batch wallet analysis via web interface
- [ ] API rate limiting and caching improvements
- [ ] Model retraining with new data
- [ ] Export analysis reports to PDF
- [ ] Integration with DeFi protocol analysis
- [ ] Mobile-responsive improvements
- [ ] Cloud deployment documentation

---

**Disclaimer**: This tool is for educational and research purposes. Always verify results independently and consult with financial/legal professionals for compliance requirements. The system provides risk assessments based on transaction patterns and should not be the sole basis for fraud determinations.