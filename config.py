import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Etherscan API configuration - Updated to V2
    ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', 'YourApiKeyToken')
    ETHERSCAN_BASE_URL = 'https://api.etherscan.io/v2/api'  # Changed from /api to /v2/api
    
    # Model configuration
    ANOMALY_THRESHOLD = 0.8  # Threshold for flagging suspicious activity
    MIN_TRANSACTIONS = 5     # Minimum transactions required for analysis
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Feature engineering parameters
    TIME_WINDOW_DAYS = 30    # Time window for transaction analysis
    MAX_TRANSACTIONS = 1000  # Maximum transactions to fetch per wallet