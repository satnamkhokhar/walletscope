import os
import importlib.util

# Proxy import: load CryptoFraudDetector from the nested implementation
_NESTED_PATH = os.path.join(os.path.dirname(__file__), 'crypto-fraud-detector', 'fraud_detector.py')

if not os.path.exists(_NESTED_PATH):
    raise FileNotFoundError(f"Nested fraud_detector not found at {_NESTED_PATH}")

_spec = importlib.util.spec_from_file_location('nested_fraud_detector', _NESTED_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

CryptoFraudDetector = _module.CryptoFraudDetector


