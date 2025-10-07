import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CryptoVisualizer:
    def __init__(self):
        self.colors = {
            'normal': '#2E8B57',
            'suspicious': '#DC143C',
            'warning': '#FF8C00',
            'info': '#4169E1'
        }
    
    def create_transaction_timeline(self, df: pd.DataFrame, wallet_address: str) -> go.Figure:
        """
        Create an interactive timeline of transactions
        """
        if df.empty:
            return self._create_empty_plot("No transaction data available")
        
        # Prepare data
        df_sorted = df.sort_values('timestamp')
        
        # Create traces for incoming and outgoing transactions
        incoming_tx = df_sorted[df_sorted['is_incoming'] == True]
        outgoing_tx = df_sorted[df_sorted['is_outgoing'] == True]
        
        fig = go.Figure()
        
        # Add incoming transactions
        if not incoming_tx.empty:
            fig.add_trace(go.Scatter(
                x=incoming_tx['datetime'],
                y=incoming_tx['value'],
                mode='markers',
                name='Incoming',
                marker=dict(
                    color=self.colors['normal'],
                    size=8,
                    symbol='circle'
                ),
                hovertemplate='<b>Incoming</b><br>' +
                            'Value: %{y:.4f} ETH<br>' +
                            'Time: %{x}<br>' +
                            'From: %{text}<extra></extra>',
                text=incoming_tx['from']
            ))
        
        # Add outgoing transactions
        if not outgoing_tx.empty:
            fig.add_trace(go.Scatter(
                x=outgoing_tx['datetime'],
                y=outgoing_tx['value'],
                mode='markers',
                name='Outgoing',
                marker=dict(
                    color=self.colors['suspicious'],
                    size=8,
                    symbol='diamond'
                ),
                hovertemplate='<b>Outgoing</b><br>' +
                            'Value: %{y:.4f} ETH<br>' +
                            'Time: %{x}<br>' +
                            'To: %{text}<extra></extra>',
                text=outgoing_tx['to']
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Transaction Timeline for {wallet_address[:10]}...',
            xaxis_title='Time',
            yaxis_title='Transaction Value (ETH)',
            hovermode='closest',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_value_distribution(self, df: pd.DataFrame) -> go.Figure:
        """
        Create distribution plots for transaction values
        """
        if df.empty:
            return self._create_empty_plot("No transaction data available")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value Distribution', 'Log Value Distribution', 
                          'Incoming vs Outgoing', 'Value by Transaction Type'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Value distribution
        fig.add_trace(
            go.Histogram(x=df['value'], nbinsx=30, name='All Transactions',
                        marker_color=self.colors['info']),
            row=1, col=1
        )
        
        # Log value distribution
        log_values = np.log10(df['value'] + 1)  # Add 1 to avoid log(0)
        fig.add_trace(
            go.Histogram(x=log_values, nbinsx=30, name='Log Values',
                        marker_color=self.colors['warning']),
            row=1, col=2
        )
        
        # Incoming vs Outgoing
        incoming_values = df[df['is_incoming'] == True]['value']
        outgoing_values = df[df['is_outgoing'] == True]['value']
        
        fig.add_trace(
            go.Box(y=incoming_values, name='Incoming', marker_color=self.colors['normal']),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=outgoing_values, name='Outgoing', marker_color=self.colors['suspicious']),
            row=2, col=1
        )
        
        # Value by transaction type
        regular_tx = df[df['is_internal'] == False]['value']
        internal_tx = df[df['is_internal'] == True]['value']
        
        fig.add_trace(
            go.Box(y=regular_tx, name='Regular', marker_color=self.colors['info']),
            row=2, col=2
        )
        fig.add_trace(
            go.Box(y=internal_tx, name='Internal', marker_color=self.colors['warning']),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Transaction Value Analysis',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_network_graph(self, df: pd.DataFrame, wallet_address: str, max_nodes: int = 50) -> go.Figure:
        """
        Create a network graph showing wallet interactions
        """
        if df.empty:
            return self._create_empty_plot("No transaction data available")
        
        # Create network graph
        G = nx.Graph()
        
        # Add main wallet
        G.add_node(wallet_address, type='main', size=20)
        
        # Add transaction partners
        all_addresses = set()
        for _, row in df.iterrows():
            if row['from'].lower() != wallet_address.lower():
                all_addresses.add(row['from'])
            if row['to'].lower() != wallet_address.lower():
                all_addresses.add(row['to'])
        
        # Limit number of nodes for visualization
        if len(all_addresses) > max_nodes:
            # Get most frequent addresses
            address_counts = {}
            for addr in df['from']:
                if addr.lower() != wallet_address.lower():
                    address_counts[addr] = address_counts.get(addr, 0) + 1
            for addr in df['to']:
                if addr.lower() != wallet_address.lower():
                    address_counts[addr] = address_counts.get(addr, 0) + 1
            
            # Sort by frequency and take top nodes
            sorted_addresses = sorted(address_counts.items(), key=lambda x: x[1], reverse=True)
            all_addresses = set([addr for addr, _ in sorted_addresses[:max_nodes-1]])
        
        # Add nodes
        for addr in all_addresses:
            G.add_node(addr, type='partner', size=10)
        
        # Add edges
        for _, row in df.iterrows():
            if row['from'].lower() != wallet_address.lower() and row['from'] in all_addresses:
                G.add_edge(wallet_address, row['from'], weight=row['value'])
            if row['to'].lower() != wallet_address.lower() and row['to'] in all_addresses:
                G.add_edge(wallet_address, row['to'], weight=row['value'])
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(edge[2]['weight'])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if node == wallet_address:
                node_colors.append(self.colors['suspicious'])
                node_sizes.append(20)
                node_text.append(f"Main Wallet<br>{node[:10]}...")
            else:
                node_colors.append(self.colors['normal'])
                node_sizes.append(10)
                node_text.append(f"Partner<br>{node[:10]}...")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line_width=2
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Wallet Interaction Network<br>{wallet_address[:10]}...',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500
                       ))
        
        return fig
    
    def create_anomaly_analysis(self, features: Dict[str, float], 
                               prediction_results: Dict[str, any]) -> go.Figure:
        """
        Create anomaly analysis visualization
        """
        # Create feature importance chart
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        # Normalize values for better visualization
        feature_values_norm = np.array(feature_values)
        if feature_values_norm.max() > 0:
            feature_values_norm = feature_values_norm / feature_values_norm.max()
        
        # Color based on anomaly score
        anomaly_score = prediction_results.get('anomaly_score', 0)
        if anomaly_score > 0.7:
            color = self.colors['suspicious']
        elif anomaly_score > 0.4:
            color = self.colors['warning']
        else:
            color = self.colors['normal']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=feature_names,
            y=feature_values_norm,
            marker_color=color,
            name='Feature Values'
        ))
        
        fig.update_layout(
            title=f'Feature Analysis (Anomaly Score: {anomaly_score:.3f})',
            xaxis_title='Features',
            yaxis_title='Normalized Values',
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_model_comparison(self, prediction_results: Dict[str, any]) -> go.Figure:
        """
        Create comparison chart of different model predictions
        """
        model_predictions = prediction_results.get('model_predictions', {})
        
        if not model_predictions:
            return self._create_empty_plot("No model predictions available")
        
        models = list(model_predictions.keys())
        scores = [model_predictions[model]['score'] for model in models]
        predictions = [model_predictions[model]['prediction'] for model in models]
        
        # Color based on prediction
        colors = []
        for pred in predictions:
            if pred == 'anomaly':
                colors.append(self.colors['suspicious'])
            else:
                colors.append(self.colors['normal'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=models,
            y=scores,
            marker_color=colors,
            text=[f"{pred.title()}<br>{score:.3f}" for pred, score in zip(predictions, scores)],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Model Predictions Comparison',
            xaxis_title='Models',
            yaxis_title='Anomaly Score',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_time_series_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Create time series analysis of transaction patterns
        """
        if df.empty:
            return self._create_empty_plot("No transaction data available")
        
        # Resample data by day
        df_sorted = df.sort_values('datetime')
        df_sorted.set_index('datetime', inplace=True)
        
        daily_stats = df_sorted.resample('D').agg({
            'value': ['count', 'sum', 'mean'],
            'gas_fee': 'sum'
        }).fillna(0)
        
        daily_stats.columns = ['tx_count', 'total_value', 'avg_value', 'total_gas_fees']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Transaction Count', 'Daily Transaction Volume',
                          'Average Transaction Value', 'Daily Gas Fees'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily transaction count
        fig.add_trace(
            go.Scatter(x=daily_stats.index, y=daily_stats['tx_count'],
                      mode='lines+markers', name='Count',
                      line=dict(color=self.colors['info'])),
            row=1, col=1
        )
        
        # Daily transaction volume
        fig.add_trace(
            go.Scatter(x=daily_stats.index, y=daily_stats['total_value'],
                      mode='lines+markers', name='Volume',
                      line=dict(color=self.colors['normal'])),
            row=1, col=2
        )
        
        # Average transaction value
        fig.add_trace(
            go.Scatter(x=daily_stats.index, y=daily_stats['avg_value'],
                      mode='lines+markers', name='Avg Value',
                      line=dict(color=self.colors['warning'])),
            row=2, col=1
        )
        
        # Daily gas fees
        fig.add_trace(
            go.Scatter(x=daily_stats.index, y=daily_stats['total_gas_fees'],
                      mode='lines+markers', name='Gas Fees',
                      line=dict(color=self.colors['suspicious'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Time Series Analysis',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig












