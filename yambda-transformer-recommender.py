"""
YAMBDA Transformer-based Recommender System
A state-of-the-art sequential recommendation model that leverages:
1. Organic vs algorithmic interaction signals
2. Audio embeddings for content-aware recommendations
3. Multi-modal fusion for improved performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math
from tqdm import tqdm
import random
from collections import defaultdict
import pickle
import os

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]

class MultiHeadAttentionWithOrganic(nn.Module):
    """Multi-head attention that considers organic/algorithmic signals"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Additional projection for organic/algo signal
        self.W_organic = nn.Linear(1, n_heads)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, is_organic: Optional[torch.Tensor] = None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Incorporate organic/algorithmic signal
        if is_organic is not None:
            # is_organic shape: (batch_size, seq_len)
            organic_weights = self.W_organic(is_organic.unsqueeze(-1))  # (batch, seq, n_heads)
            organic_weights = organic_weights.transpose(1, 2).unsqueeze(-1)  # (batch, n_heads, seq, 1)
            scores = scores + organic_weights
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.W_o(context)
        return output, attention

class TransformerBlock(nn.Module):
    """Transformer block with organic-aware attention"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttentionWithOrganic(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                is_organic: Optional[torch.Tensor] = None):
        # Self-attention
        attn_output, _ = self.attention(x, x, x, mask, is_organic)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class YambdaTransformer(nn.Module):
    """
    Transformer-based recommender for YAMBDA dataset
    Combines sequential patterns, audio embeddings, and organic/algo signals
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self.n_items = config['n_items']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.d_ff = config['d_ff']
        self.max_seq_len = config['max_seq_len']
        self.dropout = config['dropout']
        self.audio_dim = config['audio_dim']
        
        # Item embeddings
        self.item_embedding = nn.Embedding(self.n_items + 1, self.d_model, padding_idx=0)
        
        # Audio embedding projection
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model)
        )
        
        # Fusion layer for item + audio embeddings
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.n_items)
        
        # Organic/algorithmic specific components
        self.organic_gate = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.Sigmoid()
        )
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, item_ids: torch.Tensor, audio_features: Optional[torch.Tensor] = None,
                is_organic: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass
        Args:
            item_ids: (batch_size, seq_len) item indices
            audio_features: (batch_size, seq_len, audio_dim) audio embeddings
            is_organic: (batch_size, seq_len) binary organic/algo flags
            padding_mask: (batch_size, seq_len) padding mask
        """
        batch_size, seq_len = item_ids.shape
        
        # Get item embeddings
        item_emb = self.item_embedding(item_ids)
        
        # Process audio features if available
        if audio_features is not None:
            audio_emb = self.audio_projection(audio_features)
            # Fuse item and audio embeddings
            combined_emb = torch.cat([item_emb, audio_emb], dim=-1)
            x = self.fusion_layer(combined_emb)
        else:
            x = item_emb
        
        # Apply positional encoding
        x = self.pos_encoding(x)
        x = self.dropout_layer(x)
        
        # Create attention mask (causal + padding)
        if padding_mask is None:
            padding_mask = (item_ids != 0).float()
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, padding_mask, is_organic)
        
        x = self.output_norm(x)
        
        # Apply organic/algorithmic gating if available
        if is_organic is not None:
            gate_input = torch.cat([x, is_organic.unsqueeze(-1).float()], dim=-1)
            gate = self.organic_gate(gate_input)
            x = x * gate
        
        # Project to item space
        logits = self.output_projection(x)
        
        return logits

class YambdaDataset(Dataset):
    """Dataset for YAMBDA sequential recommendation"""
    
    def __init__(self, sequences: List[dict], item_map: dict, 
                 audio_embeddings: Optional[dict] = None, max_len: int = 100):
        self.sequences = sequences
        self.item_map = item_map
        self.audio_embeddings = audio_embeddings
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Map item IDs to indices
        item_indices = [self.item_map.get(item_id, 0) for item_id in seq['item_ids']]
        is_organic = seq['is_organic']
        
        # Truncate if necessary
        if len(item_indices) > self.max_len:
            item_indices = item_indices[-self.max_len:]
            is_organic = is_organic[-self.max_len:]
        
        # Get audio embeddings if available
        audio_features = None
        if self.audio_embeddings is not None:
            audio_features = []
            for item_id in seq['item_ids'][-self.max_len:]:
                if item_id in self.audio_embeddings:
                    audio_features.append(self.audio_embeddings[item_id])
                else:
                    audio_features.append(np.zeros(128))  # Default embedding
            audio_features = np.array(audio_features)
        
        return {
            'item_indices': torch.tensor(item_indices, dtype=torch.long),
            'is_organic': torch.tensor(is_organic, dtype=torch.long),
            'audio_features': torch.tensor(audio_features, dtype=torch.float) if audio_features is not None else None,
            'seq_len': len(item_indices)
        }

def collate_fn(batch):
    """Custom collate function for padding sequences"""
    item_indices = [item['item_indices'] for item in batch]
    is_organic = [item['is_organic'] for item in batch]
    
    # Pad sequences
    item_indices_padded = pad_sequence(item_indices, batch_first=True, padding_value=0)
    is_organic_padded = pad_sequence(is_organic, batch_first=True, padding_value=0)
    
    # Handle audio features
    if batch[0]['audio_features'] is not None:
        audio_features = [item['audio_features'] for item in batch]
        audio_features_padded = pad_sequence(audio_features, batch_first=True, padding_value=0)
    else:
        audio_features_padded = None
    
    # Create padding mask
    seq_lens = torch.tensor([item['seq_len'] for item in batch])
    max_len = item_indices_padded.size(1)
    padding_mask = torch.arange(max_len).expand(len(seq_lens), max_len) < seq_lens.unsqueeze(1)
    
    return {
        'item_indices': item_indices_padded,
        'is_organic': is_organic_padded,
        'audio_features': audio_features_padded,
        'padding_mask': padding_mask,
        'seq_lens': seq_lens
    }

class Trainer:
    """Trainer for the YAMBDA Transformer model"""
    
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                     padding_mask: torch.Tensor):
        """Compute masked cross-entropy loss"""
        # Reshape for loss computation
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        padding_mask = padding_mask.view(-1)
        
        # Compute loss only on non-padded positions
        loss = F.cross_entropy(logits, targets, reduction='none')
        loss = loss * padding_mask.float()
        
        return loss.sum() / padding_mask.sum()
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            # Move to device
            item_indices = batch['item_indices'].to(self.device)
            is_organic = batch['is_organic'].to(self.device)
            padding_mask = batch['padding_mask'].to(self.device)
            
            audio_features = None
            if batch['audio_features'] is not None:
                audio_features = batch['audio_features'].to(self.device)
            
            # Create input and target
            input_items = item_indices[:, :-1]
            target_items = item_indices[:, 1:]
            input_organic = is_organic[:, :-1]
            input_mask = padding_mask[:, :-1]
            target_mask = padding_mask[:, 1:]
            
            if audio_features is not None:
                input_audio = audio_features[:, :-1]
            else:
                input_audio = None
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_items, input_audio, input_organic, input_mask)
            
            # Compute loss
            loss = self.compute_loss(logits, target_items, target_mask)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / n_batches
    
    def evaluate(self, val_loader: DataLoader, k_values: List[int] = [1, 5, 10, 20]):
        """Evaluate the model"""
        self.model.eval()
        
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                # Move to device
                item_indices = batch['item_indices'].to(self.device)
                is_organic = batch['is_organic'].to(self.device)
                padding_mask = batch['padding_mask'].to(self.device)
                seq_lens = batch['seq_lens']
                
                audio_features = None
                if batch['audio_features'] is not None:
                    audio_features = batch['audio_features'].to(self.device)
                
                # For each sequence, predict the last item
                for i in range(len(item_indices)):
                    seq_len = seq_lens[i].item()
                    if seq_len < 2:
                        continue
                    
                    # Use all but last item as input
                    input_items = item_indices[i:i+1, :seq_len-1]
                    target_item = item_indices[i, seq_len-1].item()
                    input_organic = is_organic[i:i+1, :seq_len-1]
                    
                    if audio_features is not None:
                        input_audio = audio_features[i:i+1, :seq_len-1]
                    else:
                        input_audio = None
                    
                    # Get predictions
                    logits = self.model(input_items, input_audio, input_organic)
                    predictions = logits[0, -1, :]  # Last position predictions
                    
                    # Compute metrics
                    _, top_k_items = torch.topk(predictions, max(k_values))
                    top_k_items = top_k_items.cpu().numpy()
                    
                    for k in k_values:
                        hit = int(target_item in top_k_items[:k])
                        metrics[f'hit@{k}'].append(hit)
                        
                        if hit:
                            # Compute NDCG
                            rank = np.where(top_k_items[:k] == target_item)[0][0] + 1
                            ndcg = 1.0 / np.log2(rank + 1)
                            metrics[f'ndcg@{k}'].append(ndcg)
                        else:
                            metrics[f'ndcg@{k}'].append(0.0)
        
        # Average metrics
        avg_metrics = {}
        for metric, values in metrics.items():
            avg_metrics[metric] = np.mean(values)
        
        return avg_metrics

def prepare_data(listens_df: pd.DataFrame, embeddings_df: pd.DataFrame, 
                 val_split: float = 0.1, test_split: float = 0.1):
    """Prepare data for training"""
    print("Preparing data...")
    
    # Create item mapping
    unique_items = listens_df['item_id'].unique()
    item_map = {item_id: idx + 1 for idx, item_id in enumerate(unique_items)}  # 0 is padding
    
    # Create audio embedding dictionary
    audio_embeddings = {}
    for _, row in embeddings_df.iterrows():
        if 'embed' in embeddings_df.columns:
            audio_embeddings[row['item_id']] = np.array(row['embed'])
    
    # Create sequences by user
    sequences = []
    for uid, group in tqdm(listens_df.groupby('uid'), desc='Creating sequences'):
        group = group.sort_values('timestamp')
        
        sequences.append({
            'uid': uid,
            'item_ids': group['item_id'].tolist(),
            'is_organic': group['is_organic'].tolist(),
            'timestamps': group['timestamp'].tolist()
        })
    
    # Split sequences
    n_sequences = len(sequences)
    n_val = int(n_sequences * val_split)
    n_test = int(n_sequences * test_split)
    
    random.shuffle(sequences)
    
    val_sequences = sequences[:n_val]
    test_sequences = sequences[n_val:n_val + n_test]
    train_sequences = sequences[n_val + n_test:]
    
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    
    return train_sequences, val_sequences, test_sequences, item_map, audio_embeddings

def main():
    """Main training function"""
    print("YAMBDA Transformer Recommender Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'n_items': 100000,  # Will be updated based on data
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 1024,
        'max_seq_len': 100,
        'dropout': 0.1,
        'audio_dim': 128,
        'batch_size': 32,
        'learning_rate': 0.001,
        'n_epochs': 10
    }
    
    # Load data
    data_path = "/Users/ananthgs/Downloads/code/yambda_retrieval_claude"  # Update this
    print("Loading data...")
    listens_df = pd.read_parquet(f"{data_path}/flat-50m/listens.parquet")
    embeddings_df = pd.read_parquet(f"{data_path}/flat-50m/embeddings.parquet")
    
    # Sample for faster training (remove for full training)
    print("Sampling data for faster training...")
    sample_users = listens_df['uid'].unique()[:1000]
    listens_df = listens_df[listens_df['uid'].isin(sample_users)]
    
    # Prepare data
    train_sequences, val_sequences, test_sequences, item_map, audio_embeddings = prepare_data(
        listens_df, embeddings_df
    )
    
    # Update config
    config['n_items'] = len(item_map)
    print(f"Number of unique items: {config['n_items']}")
    
    # Create datasets
    train_dataset = YambdaDataset(train_sequences, item_map, audio_embeddings, config['max_seq_len'])
    val_dataset = YambdaDataset(val_sequences, item_map, audio_embeddings, config['max_seq_len'])
    test_dataset = YambdaDataset(test_sequences, item_map, audio_embeddings, config['max_seq_len'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Create model
    model = YambdaTransformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Training loop
    best_val_metric = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(config['n_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['n_epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        print("Validation metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Check for improvement
        current_metric = val_metrics['ndcg@10']
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'item_map': item_map,
                'val_metrics': val_metrics
            }, 'yambda_transformer_best.pt')
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        trainer.scheduler.step(train_loss)
    
    # Final evaluation on test set
    print("\nFinal evaluation on test set:")
    test_metrics = trainer.evaluate(test_loader)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final results
    results = {
        'config': config,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    with open('transformer_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
