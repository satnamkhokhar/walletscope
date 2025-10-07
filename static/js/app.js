// Crypto Fraud Detector Frontend JavaScript

class CryptoFraudDetectorApp {
    constructor() {
        this.initializeEventListeners();
        this.checkSystemStatus();
        this.updateModelStatus();
    }

    initializeEventListeners() {
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeWallet();
        });

        // Sample button
        document.getElementById('sampleBtn').addEventListener('click', () => {
            this.useSampleWallet();
        });

        // Train button
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.trainModel();
        });

        // Batch analysis button
        document.getElementById('batchBtn').addEventListener('click', () => {
            this.batchAnalyze();
        });

        // Report generation button
        document.getElementById('reportBtn').addEventListener('click', () => {
            this.generateReport();
        });

        // Enter key in wallet address input
        document.getElementById('walletAddress').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeWallet();
            }
        });
    }

    async analyzeWallet() {
        const walletAddress = document.getElementById('walletAddress').value.trim();
        
        if (!walletAddress) {
            this.showError('Please enter a wallet address');
            return;
        }

        if (!this.isValidEthereumAddress(walletAddress)) {
            this.showError('Please enter a valid Ethereum address (0x...)');
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ wallet_address: walletAddress })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayResults(result);
                this.showSuccess('Analysis completed successfully');
            } else {
                this.showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async trainModel() {
        const trainingWalletsText = document.getElementById('trainingWallets').value.trim();
        
        if (!trainingWalletsText) {
            this.showError('Please enter at least one wallet address for training');
            return;
        }

        const walletAddresses = trainingWalletsText.split('\n')
            .map(addr => addr.trim())
            .filter(addr => addr.length > 0);

        if (walletAddresses.length === 0) {
            this.showError('Please enter at least one valid wallet address');
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ wallet_addresses: walletAddresses })
            });

            const result = await response.json();

            if (response.ok) {
                this.showSuccess(`Model trained successfully with ${result.wallets_with_data} wallets`);
                this.updateModelStatus();
            } else {
                this.showError(result.error || 'Training failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async batchAnalyze() {
        const batchWalletsText = document.getElementById('batchWallets').value.trim();
        
        if (!batchWalletsText) {
            this.showError('Please enter wallet addresses for batch analysis');
            return;
        }

        const walletAddresses = batchWalletsText.split('\n')
            .map(addr => addr.trim())
            .filter(addr => addr.length > 0);

        if (walletAddresses.length === 0) {
            this.showError('Please enter at least one valid wallet address');
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await fetch('/api/batch-analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ wallet_addresses: walletAddresses })
            });

            const result = await response.json();

            if (response.ok) {
                this.showBatchResults(result.results);
                this.showSuccess(`Batch analysis completed for ${result.results.length} wallets`);
            } else {
                this.showError(result.error || 'Batch analysis failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async generateReport() {
        const walletAddress = document.getElementById('walletAddress').value.trim();
        
        if (!walletAddress) {
            this.showError('Please enter a wallet address first');
            return;
        }

        this.showLoading(true);
        
        try {
            const response = await fetch('/api/generate-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ wallet_address: walletAddress })
            });

            const result = await response.json();

            if (response.ok) {
                this.downloadReport(result.report, walletAddress);
                this.showSuccess('Report generated successfully');
            } else {
                this.showError(result.error || 'Report generation failed');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    async useSampleWallet() {
        try {
            const response = await fetch('/api/sample-wallets');
            const result = await response.json();
            
            if (result.wallets && result.wallets.length > 0) {
                const randomWallet = result.wallets[Math.floor(Math.random() * result.wallets.length)];
                document.getElementById('walletAddress').value = randomWallet;
                this.analyzeWallet();
            }
        } catch (error) {
            this.showError('Failed to get sample wallet');
        }
    }

    displayResults(result) {
        // Update summary information
        const summary = result.summary;
        
        // Risk level
        const riskLevelElement = document.getElementById('riskLevel');
        riskLevelElement.textContent = summary.risk_level;
        riskLevelElement.className = `display-6 fw-bold risk-${summary.risk_level.toLowerCase()}`;
        
        document.getElementById('riskDescription').textContent = 
            summary.is_suspicious ? 'Suspicious activity detected' : 'Normal activity';

        // Anomaly score
        const anomalyScore = (summary.anomaly_score * 100).toFixed(1);
        document.getElementById('anomalyScore').textContent = anomalyScore + '%';
        
        const progressBar = document.getElementById('anomalyProgress');
        progressBar.style.width = anomalyScore + '%';
        progressBar.className = `progress-bar ${this.getProgressBarColor(summary.anomaly_score)}`;

        // Transaction summary
        document.getElementById('totalTx').textContent = summary.total_transactions.toLocaleString();
        document.getElementById('totalVolume').textContent = summary.total_volume_eth.toFixed(4);
        document.getElementById('avgValue').textContent = summary.average_transaction_value.toFixed(4);
        document.getElementById('currentBalance').textContent = result.balance_eth.toFixed(4);

        // Display visualizations
        this.displayVisualizations(result.visualizations);

        // Show results section
        document.getElementById('resultsSection').classList.remove('d-none');
        document.getElementById('resultsSection').classList.add('results-fade-in');
    }

    displayVisualizations(visualizations) {
        // Timeline chart
        if (visualizations.timeline) {
            Plotly.newPlot('timelineChart', visualizations.timeline.data, visualizations.timeline.layout);
        }

        // Network chart
        if (visualizations.network) {
            Plotly.newPlot('networkChart', visualizations.network.data, visualizations.network.layout);
        }

        // Analysis chart
        if (visualizations.anomaly_analysis) {
            Plotly.newPlot('analysisChart', visualizations.anomaly_analysis.data, visualizations.anomaly_analysis.layout);
        }

        // Models comparison chart
        if (visualizations.model_comparison) {
            Plotly.newPlot('modelsChart', visualizations.model_comparison.data, visualizations.model_comparison.layout);
        }
    }

    showBatchResults(results) {
        let summary = 'Batch Analysis Results:\n\n';
        
        results.forEach((result, index) => {
            if (result.error) {
                summary += `${index + 1}. ${result.wallet_address}: ERROR - ${result.error}\n`;
            } else {
                const riskLevel = result.summary.risk_level;
                const anomalyScore = (result.summary.anomaly_score * 100).toFixed(1);
                summary += `${index + 1}. ${result.wallet_address}: ${riskLevel} (${anomalyScore}%)\n`;
            }
        });

        this.showSuccess(summary);
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/api/health');
            const result = await response.json();
            
            if (response.ok) {
                document.getElementById('apiStatus').textContent = 'Online';
                document.getElementById('apiStatus').className = 'badge bg-success';
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            } else {
                document.getElementById('apiStatus').textContent = 'Offline';
                document.getElementById('apiStatus').className = 'badge bg-danger';
            }
        } catch (error) {
            document.getElementById('apiStatus').textContent = 'Offline';
            document.getElementById('apiStatus').className = 'badge bg-danger';
        }
    }

    async updateModelStatus() {
        try {
            const response = await fetch('/api/model-info');
            const result = await response.json();
            
            const modelStatusElement = document.getElementById('modelStatus');
            const modelStatusNavElement = document.getElementById('model-status');
            
            if (result.status === 'trained') {
                modelStatusElement.textContent = 'Trained';
                modelStatusElement.className = 'badge bg-success';
                modelStatusNavElement.innerHTML = '<i class="fas fa-circle text-success"></i> Model: Trained';
            } else {
                modelStatusElement.textContent = 'Not Trained';
                modelStatusElement.className = 'badge bg-warning';
                modelStatusNavElement.innerHTML = '<i class="fas fa-circle text-warning"></i> Model: Not Trained';
            }
        } catch (error) {
            document.getElementById('modelStatus').textContent = 'Unknown';
            document.getElementById('modelStatus').className = 'badge bg-secondary';
        }
    }

    showLoading(show) {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (show) {
            loadingIndicator.classList.remove('d-none');
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Analyzing...';
        } else {
            loadingIndicator.classList.add('d-none');
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-search me-1"></i> Analyze';
        }
    }

    showError(message) {
        document.getElementById('errorMessage').textContent = message;
        new bootstrap.Modal(document.getElementById('errorModal')).show();
    }

    showSuccess(message) {
        document.getElementById('successMessage').textContent = message;
        new bootstrap.Modal(document.getElementById('successModal')).show();
    }

    downloadReport(report, walletAddress) {
        const blob = new Blob([report], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fraud_detection_report_${walletAddress}_${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    isValidEthereumAddress(address) {
        return /^0x[a-fA-F0-9]{40}$/.test(address);
    }

    getProgressBarColor(score) {
        if (score > 0.7) return 'bg-danger';
        if (score > 0.4) return 'bg-warning';
        return 'bg-success';
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.cryptoFraudDetectorApp = new CryptoFraudDetectorApp();
});

// Auto-refresh system status every 30 seconds
setInterval(() => {
    if (window.cryptoFraudDetectorApp) {
        window.cryptoFraudDetectorApp.checkSystemStatus();
    }
}, 30000);












