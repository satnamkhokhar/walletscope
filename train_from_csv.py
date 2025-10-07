#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd

from fraud_detector import CryptoFraudDetector
from config import Config


def read_wallets_from_csv(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    candidate_columns = [
        'wallet_address', 'address', 'wallet', 'eth_address', 'ethereum_address'
    ]
    # Build a case-insensitive column map
    lower_map = {c.lower(): c for c in df.columns}
    found_key = next((c for c in candidate_columns if c in lower_map), None)
    if found_key is None:
        raise ValueError(
            f"CSV must include one of columns: {candidate_columns}. Found: {list(df.columns)}"
        )
    column = lower_map[found_key]
    wallets = (
        df[column]
        .astype(str)
        .str.strip()
        .replace({"nan": None})
        .dropna()
        .unique()
        .tolist()
    )
    # Basic normalization: keep 0x-prefixed entries of length 42
    wallets = [w for w in wallets if isinstance(w, str) and w.startswith('0x') and len(w) == 42]
    if not wallets:
        raise ValueError("No valid Ethereum wallet addresses found in CSV")
    return wallets


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), 'eth_addresses.csv')
    print(f"Loading wallets from: {csv_path}")
    wallets = read_wallets_from_csv(csv_path)
    print(f"Found {len(wallets)} wallet addresses.")

    # Safety: ensure API key is configured
    if not Config.ETHERSCAN_API_KEY or Config.ETHERSCAN_API_KEY == 'YourApiKeyToken':
        raise EnvironmentError("ETHERSCAN_API_KEY is not set. Configure it in your environment or .env file.")

    # Optional: limit to avoid API rate limits for a first pass
    limit = int(os.getenv('TRAIN_LIMIT', '50'))
    if len(wallets) > limit:
        print(f"Limiting training to first {limit} wallets to avoid rate limits. Set TRAIN_LIMIT to override.")
        wallets = wallets[:limit]
    print(f"Starting training with {len(wallets)} wallets...")

    detector = CryptoFraudDetector()
    result = detector.train_model(wallets, save_model=True)

    print("Training completed.")
    print(json.dumps({
        'wallets_processed': result.get('wallets_processed'),
        'wallets_with_data': result.get('wallets_with_data'),
        'models_trained': list(result.get('model_training_results', {}).keys())
    }, indent=2))


if __name__ == '__main__':
    main()


