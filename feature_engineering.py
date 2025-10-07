import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, df: pd.DataFrame, wallet_address: str) -> Dict[str, float]:
        """
        Extract comprehensive features from transaction data for anomaly detection
        """
        if df.empty:
            return self._get_empty_features()
            
        features = {}
        
        # Basic transaction statistics
        features.update(self._extract_basic_stats(df))
        
        # Time-based features
        features.update(self._extract_time_features(df))
        
        # Value-based features
        features.update(self._extract_value_features(df))
        
        # Network features
        features.update(self._extract_network_features(df, wallet_address))
        
        # Gas and fee features
        features.update(self._extract_gas_features(df))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(df))
        
        # Risk indicators
        features.update(self._extract_risk_features(df))
        
        self.feature_names = list(features.keys())
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return default features for empty datasets"""
        return {
            'total_transactions': 0,
            'unique_counterparties': 0,
            'avg_transaction_value': 0,
            'std_transaction_value': 0,
            'max_transaction_value': 0,
            'min_transaction_value': 0,
            'total_volume': 0,
            'incoming_transactions': 0,
            'outgoing_transactions': 0,
            'incoming_volume': 0,
            'outgoing_volume': 0,
            'avg_time_between_tx': 0,
            'std_time_between_tx': 0,
            'transaction_frequency': 0,
            'new_address_ratio': 0,
            'avg_gas_price': 0,
            'avg_gas_used': 0,
            'total_gas_fees': 0,
            'internal_tx_ratio': 0,
            'high_value_tx_ratio': 0,
            'rapid_transaction_ratio': 0,
            'suspicious_pattern_score': 0
        }
    
    def _extract_basic_stats(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract basic transaction statistics"""
        return {
            'total_transactions': len(df),
            'unique_counterparties': len(set(df['from'].tolist() + df['to'].tolist())),
            'avg_transaction_value': df['value'].mean(),
            'std_transaction_value': df['value'].std(),
            'max_transaction_value': df['value'].max(),
            'min_transaction_value': df['value'].min(),
            'total_volume': df['value'].sum()
        }
    
    def _extract_time_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract time-based features"""
        if len(df) < 2:
            return {
                'avg_time_between_tx': 0,
                'std_time_between_tx': 0,
                'transaction_frequency': 0
            }
        
        # Calculate time differences between consecutive transactions
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        
        # Convert to hours
        time_diffs_hours = time_diffs / 3600
        
        # Calculate transaction frequency (transactions per day)
        time_span_days = (df['timestamp'].max() - df['timestamp'].min()) / (24 * 3600)
        if time_span_days > 0:
            tx_frequency = len(df) / time_span_days
        else:
            tx_frequency = len(df)
        
        return {
            'avg_time_between_tx': time_diffs_hours.mean(),
            'std_time_between_tx': time_diffs_hours.std(),
            'transaction_frequency': tx_frequency
        }
    
    def _extract_value_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract value-based features"""
        incoming_tx = df[df['is_incoming'] == True]
        outgoing_tx = df[df['is_outgoing'] == True]
        
        return {
            'incoming_transactions': len(incoming_tx),
            'outgoing_transactions': len(outgoing_tx),
            'incoming_volume': incoming_tx['value'].sum(),
            'outgoing_volume': outgoing_tx['value'].sum()
        }
    
    def _extract_network_features(self, df: pd.DataFrame, wallet_address: str) -> Dict[str, float]:
        """Extract network interaction features"""
        # Get unique addresses this wallet has interacted with
        all_addresses = set(df['from'].tolist() + df['to'].tolist())
        all_addresses.discard(wallet_address.lower())
        
        # Calculate ratio of transactions to new addresses
        # (simplified: consider addresses that appear only once as "new")
        address_counts = {}
        for addr in df['from']:
            if addr.lower() != wallet_address.lower():
                address_counts[addr.lower()] = address_counts.get(addr.lower(), 0) + 1
        for addr in df['to']:
            if addr.lower() != wallet_address.lower():
                address_counts[addr.lower()] = address_counts.get(addr.lower(), 0) + 1
        
        new_addresses = sum(1 for count in address_counts.values() if count == 1)
        new_address_ratio = new_addresses / len(all_addresses) if all_addresses else 0
        
        return {
            'new_address_ratio': new_address_ratio
        }
    
    def _extract_gas_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract gas and fee-related features"""
        # Filter out internal transactions for gas analysis
        regular_tx = df[df['is_internal'] == False]
        
        if regular_tx.empty:
            return {
                'avg_gas_price': 0,
                'avg_gas_used': 0,
                'total_gas_fees': 0
            }
        
        return {
            'avg_gas_price': regular_tx['gas_price'].mean(),
            'avg_gas_used': regular_tx['gas_used'].mean(),
            'total_gas_fees': regular_tx['gas_fee'].sum()
        }
    
    def _extract_behavioral_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        # Ratio of internal transactions
        internal_ratio = len(df[df['is_internal'] == True]) / len(df) if len(df) > 0 else 0
        
        # Ratio of high-value transactions (above 75th percentile)
        if len(df) > 0:
            high_value_threshold = df['value'].quantile(0.75)
            high_value_ratio = len(df[df['value'] > high_value_threshold]) / len(df)
        else:
            high_value_ratio = 0
        
        return {
            'internal_tx_ratio': internal_ratio,
            'high_value_tx_ratio': high_value_ratio
        }
    
    def _extract_risk_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract risk indicator features"""
        if len(df) < 2:
            return {
                'rapid_transaction_ratio': 0,
                'suspicious_pattern_score': 0
            }
        
        # Rapid transactions (less than 1 hour apart)
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()
        rapid_tx = len(time_diffs[time_diffs < 3600])  # Less than 1 hour
        rapid_ratio = rapid_tx / len(time_diffs) if len(time_diffs) > 0 else 0
        
        # Suspicious pattern score (combination of multiple risk factors)
        risk_factors = []
        
        # High transaction frequency
        if len(df) > 10:
            time_span_days = (df['timestamp'].max() - df['timestamp'].min()) / (24 * 3600)
            if time_span_days > 0:
                tx_per_day = len(df) / time_span_days
                if tx_per_day > 50:  # More than 50 transactions per day
                    risk_factors.append(1)
        
        # High value transactions
        if df['value'].max() > 100:  # Transactions over 100 ETH
            risk_factors.append(1)
        
        # Many unique counterparties
        unique_counterparties = len(set(df['from'].tolist() + df['to'].tolist()))
        if unique_counterparties > 50:  # More than 50 unique addresses
            risk_factors.append(1)
        
        # High gas fees
        if 'gas_fee' in df.columns and df['gas_fee'].sum() > 10:  # Total gas fees over 10 ETH
            risk_factors.append(1)
        
        suspicious_score = sum(risk_factors) / 4  # Normalize to 0-1
        
        return {
            'rapid_transaction_ratio': rapid_ratio,
            'suspicious_pattern_score': suspicious_score
        }
    
    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dictionary to numpy array for model input"""
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        return np.array(feature_vector).reshape(1, -1)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order"""
        return self.feature_names












