import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class AnomalyDetector:
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.models = {}
        self.is_trained = False
        self.feature_names = []
        
    def train(self, features_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
        """
        Train the anomaly detection model(s)
        """
        if features_df.empty:
            raise ValueError("No training data provided")
        
        self.feature_names = feature_names
        X = features_df[feature_names].values
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        training_results = {}
        
        if self.model_type == 'ensemble' or self.model_type == 'isolation_forest':
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            iso_forest.fit(X_scaled)
            self.models['isolation_forest'] = iso_forest
            training_results['isolation_forest'] = {'status': 'trained'}
        
        if self.model_type == 'ensemble' or self.model_type == 'one_class_svm':
            # Train One-Class SVM
            oc_svm = OneClassSVM(
                kernel='rbf',
                nu=0.1,
                gamma='scale'
            )
            oc_svm.fit(X_scaled)
            self.models['one_class_svm'] = oc_svm
            training_results['one_class_svm'] = {'status': 'trained'}
        
        if self.model_type == 'ensemble' or self.model_type == 'autoencoder':
            # Train Autoencoder
            autoencoder = self._train_autoencoder(X_scaled)
            self.models['autoencoder'] = autoencoder
            training_results['autoencoder'] = {'status': 'trained'}
        
        self.is_trained = True
        return training_results
    
    def _train_autoencoder(self, X_scaled: np.ndarray, epochs: int = 100) -> Autoencoder:
        """Train the autoencoder model"""
        input_dim = X_scaled.shape[1]
        autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=max(4, input_dim // 4))
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
        
        # Training loop
        autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = autoencoder(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        return autoencoder
    
    def predict(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Predict anomaly score and classification for given features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert features to vector
        feature_vector = []
        for feature_name in self.feature_names:
            feature_vector.append(features.get(feature_name, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        results = {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'confidence': 0.0,
            'model_predictions': {}
        }
        
        predictions = []
        scores = []
        
        # Get predictions from each model
        if 'isolation_forest' in self.models:
            iso_pred = self.models['isolation_forest'].predict(X_scaled)[0]
            iso_score = self.models['isolation_forest'].score_samples(X_scaled)[0]
            predictions.append(1 if iso_pred == -1 else 0)  # -1 is anomaly in Isolation Forest
            scores.append(-iso_score)  # Higher score = more anomalous
            results['model_predictions']['isolation_forest'] = {
                'prediction': 'anomaly' if iso_pred == -1 else 'normal',
                'score': -iso_score
            }
        
        if 'one_class_svm' in self.models:
            svm_pred = self.models['one_class_svm'].predict(X_scaled)[0]
            svm_score = self.models['one_class_svm'].score_samples(X_scaled)[0]
            predictions.append(1 if svm_pred == -1 else 0)  # -1 is anomaly in One-Class SVM
            scores.append(-svm_score)
            results['model_predictions']['one_class_svm'] = {
                'prediction': 'anomaly' if svm_pred == -1 else 'normal',
                'score': -svm_score
            }
        
        if 'autoencoder' in self.models:
            autoencoder = self.models['autoencoder']
            X_tensor = torch.FloatTensor(X_scaled)
            autoencoder.eval()
            with torch.no_grad():
                reconstructed = autoencoder(X_tensor)
                reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2).item()
            
            # Normalize reconstruction error to 0-1 range
            # This is a simplified approach - in practice, you'd use validation data
            normalized_error = min(reconstruction_error * 100, 1.0)
            predictions.append(1 if normalized_error > 0.5 else 0)
            scores.append(normalized_error)
            results['model_predictions']['autoencoder'] = {
                'prediction': 'anomaly' if normalized_error > 0.5 else 'normal',
                'score': normalized_error
            }
        
        # Ensemble decision
        if predictions:
            avg_score = np.mean(scores)
            avg_prediction = np.mean(predictions)
            
            results['anomaly_score'] = avg_score
            results['confidence'] = abs(avg_prediction - 0.5) * 2  # Convert to 0-1 confidence
            results['is_anomaly'] = avg_prediction > 0.5
        
        return results
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        
        # Save models
        for model_name, model in self.models.items():
            if model_name == 'autoencoder':
                torch.save(model.state_dict(), f"{filepath}_{model_name}.pth")
            else:
                joblib.dump(model, f"{filepath}_{model_name}.pkl")
        
        # Save feature names
        joblib.dump(self.feature_names, f"{filepath}_features.pkl")
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            # Load scaler
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            
            # Load feature names
            self.feature_names = joblib.load(f"{filepath}_features.pkl")
            
            # Load models
            self.models = {}
            
            # Try to load Isolation Forest
            try:
                self.models['isolation_forest'] = joblib.load(f"{filepath}_isolation_forest.pkl")
            except FileNotFoundError:
                pass
            
            # Try to load One-Class SVM
            try:
                self.models['one_class_svm'] = joblib.load(f"{filepath}_one_class_svm.pkl")
            except FileNotFoundError:
                pass
            
            # Try to load Autoencoder
            try:
                input_dim = len(self.feature_names)
                autoencoder = Autoencoder(input_dim=input_dim, encoding_dim=max(4, input_dim // 4))
                autoencoder.load_state_dict(torch.load(f"{filepath}_autoencoder.pth"))
                autoencoder.eval()
                self.models['autoencoder'] = autoencoder
            except FileNotFoundError:
                pass
            
            self.is_trained = True
            print(f"Models loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the trained models"""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_type': self.model_type,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'models': list(self.models.keys())
        }












