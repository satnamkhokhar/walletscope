import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
from config import Config

print(f"Etherscan key detected: {Config.ETHERSCAN_API_KEY[:4]}...{Config.ETHERSCAN_API_KEY[-4:]}")  # mask

class EthereumDataFetcher:
    def __init__(self):
        self.api_key = Config.ETHERSCAN_API_KEY
        self.base_url = Config.ETHERSCAN_BASE_URL
        # Debug: confirm key is loaded (masked)
        try:
            masked = (self.api_key[:4] + "..." + self.api_key[-4:]) if self.api_key else "MISSING"
            print(f"Etherscan key loaded: {masked}")
        except Exception:
            pass

    def _call_etherscan(self, params: Dict, retries: int = 3, backoff: float = 1.5):
        for attempt in range(retries):
            try:
                resp = requests.get(self.base_url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                status = data.get('status')
                if status == '1':
                    return data.get('result', [])
                # Log full detail on NOTOK/0
                print(f"Etherscan NOTOK: message={data.get('message')} result={data.get('result')}")
                # Handle rate limiting with backoff
                if isinstance(data.get('result'), str) and 'Max rate limit' in data['result']:
                    time.sleep(backoff * (attempt + 1))
                    continue
                # If invalid API key or other permanent error, stop retrying
                if isinstance(data.get('result'), str) and 'Invalid API Key' in data['result']:
                    return []
                # No transactions found -> empty list
                if data.get('message') == 'No transactions found':
                    return []
                return []
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt+1}): {e}")
                time.sleep(backoff * (attempt + 1))
        return []

    def get_wallet_transactions(self, wallet_address: str, start_block: int = 0, end_block: int = 99999999) -> List[Dict]:
        params = {
            'chainid': 1,  # Add this line for Ethereum mainnet
            'module': 'account',
            'action': 'txlist',
            'address': wallet_address,
            'startblock': start_block,
            'endblock': end_block,
            'sort': 'asc',
            'apikey': self.api_key
        }
        return self._call_etherscan(params)

    def get_internal_transactions(self, wallet_address: str) -> List[Dict]:
        params = {
            'chainid': 1,  # Add this line
            'module': 'account',
            'action': 'txlistinternal',
            'address': wallet_address,
            'startblock': 0,
            'endblock': 99999999,
            'sort': 'asc',
            'apikey': self.api_key
        }
        return self._call_etherscan(params)

    def get_wallet_balance(self, wallet_address: str) -> float:
        params = {
            'chainid': 1,  # Add this line
            'module': 'account',
            'action': 'balance',
            'address': wallet_address,
            'tag': 'latest',
            'apikey': self.api_key
        }
        res = self._call_etherscan(params, retries=2)
        if isinstance(res, list) and len(res) > 0:
            try:
                return float(res[0]) / 1e18
            except Exception:
                return 0.0
        return 0.0

    def fetch_wallet_data(self, wallet_address: str, max_transactions: int = None) -> pd.DataFrame:
        if max_transactions is None:
            max_transactions = Config.MAX_TRANSACTIONS

        transactions = self.get_wallet_transactions(wallet_address)[:max_transactions]
        internal_transactions = self.get_internal_transactions(wallet_address)[: max_transactions // 2]

        all_transactions = []
        for tx in transactions:
            all_transactions.append({
                'hash': tx.get('hash'),
                'from': tx.get('from'),
                'to': tx.get('to'),
                'value': float(tx.get('value', 0)) / 1e18,
                'gas': int(tx.get('gas', 0) or 0),
                'gas_price': int(tx.get('gasPrice', 0) or 0),
                'gas_used': int(tx.get('gasUsed', 0) or 0),
                'timestamp': int(tx.get('timeStamp', 0) or 0),
                'block_number': int(tx.get('blockNumber', 0) or 0),
                'is_internal': False,
                'is_incoming': str(tx.get('to', '')).lower() == wallet_address.lower(),
                'is_outgoing': str(tx.get('from', '')).lower() == wallet_address.lower()
            })

        for tx in internal_transactions:
            all_transactions.append({
                'hash': tx.get('hash'),
                'from': tx.get('from'),
                'to': tx.get('to'),
                'value': float(tx.get('value', 0)) / 1e18,
                'gas': 0,
                'gas_price': 0,
                'gas_used': 0,
                'timestamp': int(tx.get('timeStamp', 0) or 0),
                'block_number': int(tx.get('blockNumber', 0) or 0),
                'is_internal': True,
                'is_incoming': str(tx.get('to', '')).lower() == wallet_address.lower(),
                'is_outgoing': str(tx.get('from', '')).lower() == wallet_address.lower()
            })

        df = pd.DataFrame(all_transactions)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df = df.sort_values('timestamp')
            df['gas_fee'] = df['gas_used'] * df['gas_price'] / 1e18
        return df

    def is_valid_address(self, address: str) -> bool:
        if not address or len(address) != 42:
            return False
        if not address.startswith('0x'):
            return False
        try:
            int(address[2:], 16)
            return True
        except ValueError:
            return False