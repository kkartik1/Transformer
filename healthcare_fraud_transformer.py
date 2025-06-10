#!/usr/bin/env python3
"""
Transformer-Based Healthcare Fraud, Waste, Abuse, and Error (FWAE) Detection System

This system applies Transformer models to detect anomalous patterns in healthcare claims
using attention mechanisms, sequence modeling, and multi-modal data fusion.

Key Components:
1. Claims Sequence Modeling
2. Multi-Head Attention for Pattern Detection
3. Anomaly Scoring and Classification
4. Provider Behavior Analysis
5. Temporal Pattern Recognition

Author: AI Assistant
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HealthcareTransformerEncoder(nn.Module):
    """
    Transformer Encoder specifically designed for healthcare claims data
    """
    def __init__(self, input_dim, model_dim=512, num_heads=8, num_layers=6, 
                 dropout=0.1, max_seq_length=100):
        super(HealthcareTransformerEncoder, self).__init__()
        
        self.model_dim = model_dim
        self.max_seq_length = max_seq_length
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding for temporal patterns
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transpose for transformer (seq_length, batch_size, model_dim)
        x = x.transpose(0, 1)
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Transpose back (batch_size, seq_length, model_dim)
        encoded = encoded.transpose(0, 1)
        
        return self.layer_norm(encoded)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequence information
    """
    def __init__(self, d_model, max_length=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FWAEDetectionModel(nn.Module):
    """
    Complete FWAE Detection Model combining multiple detection strategies
    """
    def __init__(self, input_dim, model_dim=512, num_heads=8, num_layers=6):
        super(FWAEDetectionModel, self).__init__()
        
        # Transformer encoder for sequence modeling
        self.transformer_encoder = HealthcareTransformerEncoder(
            input_dim, model_dim, num_heads, num_layers
        )
        
        # Multi-task heads for different types of fraud detection
        self.anomaly_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.fraud_classification_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, 4)  # 4 classes: Normal, Fraud, Waste, Error
        )
        
        # Provider risk assessment head
        self.provider_risk_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Attention pooling for sequence aggregation
        self.attention_pooling = AttentionPooling(model_dim)
        
    def forward(self, x, mask=None):
        # Encode sequences using transformer
        encoded_sequences = self.transformer_encoder(x, mask)
        
        # Pool sequences using attention
        pooled_representation = self.attention_pooling(encoded_sequences, mask)
        
        # Multi-task outputs
        anomaly_score = self.anomaly_head(pooled_representation)
        fraud_classification = self.fraud_classification_head(pooled_representation)
        provider_risk = self.provider_risk_head(pooled_representation)
        
        return {
            'anomaly_score': anomaly_score,
            'fraud_classification': fraud_classification,
            'provider_risk': provider_risk,
            'encoded_representation': pooled_representation
        }

class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence aggregation
    """
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequences, mask=None):
        # sequences: (batch_size, seq_length, hidden_dim)
        attention_weights = self.attention(sequences)  # (batch_size, seq_length, 1)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1), -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=1)
        pooled = torch.sum(sequences * attention_weights, dim=1)
        
        return pooled

class HealthcareClaimsProcessor:
    """
    Preprocessor for healthcare claims data
    """
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def create_fraud_indicators(self, df):
        """
        Create fraud indicator features based on domain knowledge
        """
        fraud_features = df.copy()
        
        # Billing anomalies
        fraud_features['charge_per_unit'] = fraud_features['total_charges'] / fraud_features.get('units', 1)
        fraud_features['unusual_charge_flag'] = (
            fraud_features.groupby('procedure_code')['total_charges']
            .transform(lambda x: np.abs(x - x.median()) > 2 * x.std())
        ).astype(int)
        
        # Provider patterns
        provider_stats = fraud_features.groupby('provider_npi').agg({
            'total_charges': ['mean', 'std', 'count'],
            'procedure_code': 'nunique'
        }).round(2)
        provider_stats.columns = ['provider_avg_charge', 'provider_charge_std', 
                                'provider_claim_count', 'provider_procedure_variety']
        fraud_features = fraud_features.merge(provider_stats, on='provider_npi', how='left')
        
        # Temporal patterns
        fraud_features['service_date'] = pd.to_datetime(fraud_features['service_date'])
        fraud_features['day_of_week'] = fraud_features['service_date'].dt.dayofweek
        fraud_features['is_weekend'] = (fraud_features['day_of_week'] >= 5).astype(int)
        fraud_features['hour_of_service'] = fraud_features['service_date'].dt.hour
        
        # Patient patterns
        patient_stats = fraud_features.groupby('patient_id').agg({
            'total_charges': 'sum',
            'claim_id': 'count'
        })
        patient_stats.columns = ['patient_total_charges', 'patient_claim_count']
        fraud_features = fraud_features.merge(patient_stats, on='patient_id', how='left')
        
        # Diagnosis-procedure alignment
        fraud_features['dx_proc_alignment_score'] = self._calculate_dx_proc_alignment(fraud_features)
        
        return fraud_features
    
    def _calculate_dx_proc_alignment(self, df):
        """
        Calculate alignment score between diagnoses and procedures
        """
        # Simplified alignment scoring based on common patterns
        alignment_rules = {
            ('I10', '99213'): 1.0,  # Hypertension with office visit
            ('E11.9', '80053'): 1.0,  # Diabetes with metabolic panel
            ('M25.561', '73721'): 1.0,  # Knee pain with MRI
            ('J06.9', '99213'): 1.0,  # URI with office visit
        }
        
        scores = []
        for _, row in df.iterrows():
            primary_dx = row.get('primary_diagnosis', '')
            procedure = row.get('procedure_code', '')
            score = alignment_rules.get((primary_dx, procedure), 0.5)  # Default neutral score
            scores.append(score)
        
        return scores
    
    def prepare_sequences(self, df, sequence_column='patient_id', max_length=50):
        """
        Prepare sequential data for transformer input
        """
        # Sort by patient and service date
        df_sorted = df.sort_values([sequence_column, 'service_date'])
        
        # Create sequences grouped by patient/provider
        sequences = []
        labels = []
        
        for group_id, group_data in df_sorted.groupby(sequence_column):
            if len(group_data) < 2:  # Skip sequences that are too short
                continue
            
            # Truncate or pad sequences
            group_data = group_data.head(max_length)
            
            # Extract features for sequence
            sequence_features = self._extract_sequence_features(group_data)
            sequences.append(sequence_features)
            
            # Create labels (fraud indicators)
            fraud_label = self._create_fraud_labels(group_data)
            labels.append(fraud_label)
        
        return sequences, labels
    
    def _extract_sequence_features(self, group_data):
        """
        Extract numerical features from a sequence of claims
        """
        feature_columns = [
            'total_charges', 'charge_per_unit', 'unusual_charge_flag',
            'provider_avg_charge', 'provider_charge_std', 'provider_claim_count',
            'day_of_week', 'is_weekend', 'dx_proc_alignment_score'
        ]
        
        # Fill missing values and normalize
        features = group_data[feature_columns].fillna(0)
        return features.values
    
    def _create_fraud_labels(self, group_data):
        """
        Create fraud labels based on heuristics and patterns
        """
        # Simplified fraud labeling - in practice, you'd have ground truth labels
        fraud_indicators = 0
        
        # High charges
        if group_data['total_charges'].max() > group_data['total_charges'].quantile(0.95):
            fraud_indicators += 1
            
        # Unusual patterns
        if group_data['unusual_charge_flag'].sum() > len(group_data) * 0.3:
            fraud_indicators += 1
            
        # Frequent claims
        if len(group_data) > 20:  # More than 20 claims
            fraud_indicators += 1
        
        # Weekend services (potentially suspicious)
        if group_data['is_weekend'].mean() > 0.5:
            fraud_indicators += 1
        
        # Return fraud probability
        return min(fraud_indicators / 4.0, 1.0)

class FWAETrainer:
    """
    Training pipeline for FWAE detection model
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
    def train_epoch(self, dataloader, criterion):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            sequences, labels = batch
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(sequences)
            
            # Multi-task loss
            anomaly_loss = criterion['anomaly'](outputs['anomaly_score'].squeeze(), labels)
            
            total_loss_batch = anomaly_loss
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += total_loss_batch.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        """
        Evaluate model performance
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                sequences, labels = batch
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                
                # Calculate loss
                anomaly_loss = criterion['anomaly'](outputs['anomaly_score'].squeeze(), labels)
                total_loss += anomaly_loss.item()
                
                # Store predictions
                predictions.extend(outputs['anomaly_score'].cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(dataloader), predictions, true_labels

class FWAEAnalyzer:
    """
    Analysis and interpretation tools for FWAE detection
    """
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def detect_anomalies(self, claims_data, threshold=0.7):
        """
        Detect anomalous claims using the trained model
        """
        # Preprocess data
        processed_data = self.processor.create_fraud_indicators(claims_data)
        sequences, _ = self.processor.prepare_sequences(processed_data)
        
        # Convert to tensors
        sequence_tensors = [torch.FloatTensor(seq) for seq in sequences]
        
        # Get predictions
        self.model.eval()
        anomaly_scores = []
        
        with torch.no_grad():
            for seq_tensor in sequence_tensors:
                seq_tensor = seq_tensor.unsqueeze(0)  # Add batch dimension
                outputs = self.model(seq_tensor)
                score = outputs['anomaly_score'].item()
                anomaly_scores.append(score)
        
        # Identify high-risk cases
        high_risk_indices = [i for i, score in enumerate(anomaly_scores) if score > threshold]
        
        return anomaly_scores, high_risk_indices
    
    def analyze_provider_patterns(self, claims_data):
        """
        Analyze provider-level fraud patterns
        """
        processed_data = self.processor.create_fraud_indicators(claims_data)
        
        provider_analysis = processed_data.groupby('provider_npi').agg({
            'total_charges': ['sum', 'mean', 'std', 'count'],
            'unusual_charge_flag': 'sum',
            'patient_id': 'nunique',
            'procedure_code': 'nunique',
            'is_weekend': 'mean'
        }).round(2)
        
        # Calculate risk scores
        provider_analysis['risk_score'] = (
            provider_analysis[('unusual_charge_flag', 'sum')] / 
            provider_analysis[('total_charges', 'count')] * 100
        )
        
        return provider_analysis.sort_values('risk_score', ascending=False)
    
    def generate_fraud_report(self, claims_data, output_file='fraud_analysis_report.html'):
        """
        Generate comprehensive fraud analysis report
        """
        anomaly_scores, high_risk_indices = self.detect_anomalies(claims_data)
        provider_patterns = self.analyze_provider_patterns(claims_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Anomaly score distribution
        axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Provider risk scores
        top_providers = provider_patterns.head(20)
        axes[0, 1].barh(range(len(top_providers)), top_providers['risk_score'])
        axes[0, 1].set_title('Top 20 High-Risk Providers')
        axes[0, 1].set_xlabel('Risk Score')
        
        # Charges by day of week
        processed_data = self.processor.create_fraud_indicators(claims_data)
        day_charges = processed_data.groupby('day_of_week')['total_charges'].mean()
        axes[1, 0].bar(range(7), day_charges)
        axes[1, 0].set_title('Average Charges by Day of Week')
        axes[1, 0].set_xlabel('Day of Week (0=Monday)')
        axes[1, 0].set_ylabel('Average Charges')
        
        # Procedure code frequency
        top_procedures = processed_data['procedure_code'].value_counts().head(10)
        axes[1, 1].barh(range(len(top_procedures)), top_procedures.values)
        axes[1, 1].set_title('Top 10 Most Frequent Procedures')
        axes[1, 1].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('fraud_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Summary statistics
        print("\n" + "="*60)
        print("HEALTHCARE FRAUD ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Claims Analyzed: {len(claims_data):,}")
        print(f"High-Risk Claims Identified: {len(high_risk_indices):,}")
        print(f"Fraud Detection Rate: {len(high_risk_indices)/len(claims_data)*100:.2f}%")
        print(f"Average Anomaly Score: {np.mean(anomaly_scores):.3f}")
        print(f"High-Risk Providers: {len(provider_patterns[provider_patterns['risk_score'] > 10]):,}")
        
        return {
            'anomaly_scores': anomaly_scores,
            'high_risk_claims': high_risk_indices,
            'provider_analysis': provider_patterns,
            'summary_stats': {
                'total_claims': len(claims_data),
                'high_risk_claims': len(high_risk_indices),
                'detection_rate': len(high_risk_indices)/len(claims_data)*100,
                'avg_anomaly_score': np.mean(anomaly_scores)
            }
        }

def demo_fraud_detection():
    """
    Demonstration of the FWAE detection system
    """
    print("Healthcare FWAE Detection with Transformers - Demo")
    print("="*60)
    
    # Generate synthetic claims data (you would load real data here)
    print("Generating synthetic claims data...")
    
    # Create sample data structure
    np.random.seed(42)
    n_claims = 1000
    
    sample_data = pd.DataFrame({
        'claim_id': range(n_claims),
        'patient_id': np.random.randint(1, 200, n_claims),
        'provider_npi': np.random.choice(['1234567890', '2345678901', '3456789012'], n_claims),
        'procedure_code': np.random.choice(['99213', '99214', '85025', '93000'], n_claims),
        'primary_diagnosis': np.random.choice(['I10', 'E11.9', 'M25.561', 'J06.9'], n_claims),
        'total_charges': np.random.lognormal(5, 1, n_claims),
        'units': np.random.randint(1, 5, n_claims),
        'service_date': pd.date_range('2024-01-01', periods=n_claims, freq='H')
    })
    
    # Initialize components
    print("Initializing FWAE detection system...")
    processor = HealthcareClaimsProcessor()
    
    # Create model
    input_dim = 9  # Number of features
    model = FWAEDetectionModel(input_dim, model_dim=128, num_heads=4, num_layers=3)
    
    # Initialize analyzer
    analyzer = FWAEAnalyzer(model, processor)
    
    # Run fraud detection analysis
    print("Running fraud detection analysis...")
    try:
        results = analyzer.generate_fraud_report(sample_data)
        
        print("\nAnalysis completed successfully!")
        print(f"Detected {results['summary_stats']['high_risk_claims']} high-risk claims")
        print(f"Detection rate: {results['summary_stats']['detection_rate']:.2f}%")
        
    except Exception as e:
        print(f"Demo completed with synthetic data structure: {e}")
        print("Note: This is a framework demonstration. Real implementation would require:")
        print("- Actual claims data with proper preprocessing")
        print("- Model training on labeled fraud cases")
        print("- Feature engineering specific to your data")
        print("- Validation with domain experts")

if __name__ == "__main__":
    demo_fraud_detection()
