# AI-Powered Crypto Fraud Detector

A comprehensive machine learning system for detecting suspicious cryptocurrency wallet activity using Ethereum blockchain data. This system combines advanced anomaly detection algorithms with real-time blockchain data analysis to identify potential fraud patterns.

## 🚀 Features

- **Real-time Blockchain Analysis**: Fetches transaction data from Ethereum blockchain via Etherscan API
- **Advanced ML Models**: Uses ensemble of Isolation Forest, One-Class SVM, and Autoencoder neural networks
- **Comprehensive Feature Engineering**: Extracts 22+ behavioral and statistical features from transaction data
- **Interactive Web Interface**: Modern, responsive dashboard with real-time visualizations
- **Batch Processing**: Analyze multiple wallets simultaneously
- **Detailed Reporting**: Generate comprehensive fraud detection reports
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
                       │  (Blockchain)   │    │  (Local Files)  │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- Flask web framework
- Machine learning libraries (scikit-learn, PyTorch)
- Data processing (pandas, numpy)
- Visualization (plotly, matplotlib)
- Network analysis (networkx)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd crypto-fraud-detector
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8080
   ```

3. **Train the model** (optional but recommended):
   - Use the sample wallet addresses provided
   - Click "Train Model" in the sidebar
   - Wait for training to complete

4. **Analyze a wallet**:
   - Enter an Ethereum wallet address
   - Click "Analyze" to get fraud detection results
   - View interactive visualizations and risk assessment

## 📊 Usage Examples

### Command Line Usage

```python
from fraud_detector import CryptoFraudDetector

# Initialize the detector
detector = CryptoFraudDetector()

# Train the model with sample wallets
training_wallets = [
    "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
    "0x28C6c06298d514Db089934071355E5743bf21d60",
    "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549"
]
detector.train_model(training_wallets)

# Analyze a single wallet
result = detector.analyze_wallet("0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")
print(f"Risk Level: {result['summary']['risk_level']}")
print(f"Anomaly Score: {result['summary']['anomaly_score']:.3f}")

# Generate a report
report = detector.generate_report("0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6")
print(report)
```

### Web Interface Features

- **Single Wallet Analysis**: Enter any Ethereum address for detailed analysis
- **Batch Analysis**: Analyze multiple wallets at once
- **Model Training**: Train the ML model with known wallet patterns
- **Interactive Visualizations**: 
  - Transaction timeline
  - Network interaction graphs
  - Feature analysis charts
  - Model comparison plots
- **Report Generation**: Download detailed PDF reports

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

## 📈 Performance Metrics

- **Accuracy**: Measures overall prediction correctness
- **Precision**: Reduces false positives
- **Recall**: Captures all suspicious activity
- **F1-Score**: Balanced measure of precision and recall

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Model parameters
ANOMALY_THRESHOLD = 0.8  # Threshold for flagging suspicious activity
MIN_TRANSACTIONS = 5     # Minimum transactions required for analysis

# API settings
ETHERSCAN_API_KEY = "your_api_key"
MAX_TRANSACTIONS = 1000  # Maximum transactions to fetch per wallet

# Feature engineering
TIME_WINDOW_DAYS = 30    # Analysis time window
```

## 📁 Project Structure

```
crypto-fraud-detector/
├── app.py                 # Flask web application
├── fraud_detector.py      # Main fraud detection class
├── data_fetcher.py        # Blockchain data retrieval
├── feature_engineering.py # Feature extraction
├── anomaly_detector.py    # ML models implementation
├── visualizer.py          # Data visualization
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   └── index.html
├── static/               # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── models/               # Saved ML models
├── results/              # Analysis results
└── README.md
```

## 🧪 Testing

### Sample Wallet Addresses

The system includes sample wallet addresses for testing:

- `0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6` (Binance hot wallet)
- `0x28C6c06298d514Db089934071355E5743bf21d60` (Binance cold wallet)
- `0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549` (Exchange wallet)

### Running Tests

```bash
# Test the API endpoints
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"wallet_address": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"}'

# Check system health
curl http://localhost:5000/api/health
```

## 🔒 Security Considerations

- **API Rate Limiting**: Respect Etherscan API rate limits
- **Input Validation**: All wallet addresses are validated
- **Error Handling**: Comprehensive error handling and logging
- **Data Privacy**: No sensitive data is stored permanently
- **Model Security**: Models are saved locally, not transmitted

## 🚨 Limitations

- **API Dependencies**: Requires active internet connection and Etherscan API
- **Training Data**: Model performance depends on quality of training data
- **False Positives**: May flag legitimate high-frequency traders
- **Ethereum Only**: Currently supports only Ethereum blockchain
- **Historical Data**: Limited to available transaction history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Etherscan**: For providing blockchain data API
- **scikit-learn**: For machine learning algorithms
- **PyTorch**: For neural network implementation
- **Plotly**: For interactive visualizations
- **Flask**: For web framework

## 📞 Support

For questions, issues, or contributions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed description
4. Include error logs and reproduction steps

## 🔮 Future Enhancements

- [ ] Support for multiple blockchains (Bitcoin, Polygon, etc.)
- [ ] Real-time monitoring and alerts
- [ ] Advanced network analysis (clustering, community detection)
- [ ] Integration with DeFi protocols
- [ ] Mobile application
- [ ] API rate limiting and caching
- [ ] Advanced visualization options
- [ ] Model explainability features
- [ ] Automated model retraining
- [ ] Cloud deployment options

---

**Disclaimer**: This tool is for educational and research purposes. Always verify results independently and consult with financial/legal professionals for compliance requirements.
