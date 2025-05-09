import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model
    """
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1)]

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer model for translation tasks
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_seq_length=5000):
        super(DecoderOnlyTransformer, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Create a mask to prevent attending to future tokens (causal mask)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_length, max_seq_length) * float('-inf'), diagonal=1)
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Final linear layer to predict next token
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters of the model"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None):
        """
        Args:
            src: Source sequence [batch_size, src_seq_len]
            tgt: Target sequence [batch_size, tgt_seq_len] (optional, for training)
        """
        # If target is not provided, use source for autoregressive generation
        if tgt is None:
            tgt = src
            
        # Get sequence length
        seq_len = tgt.size(1)
        
        # Create embedding
        tgt_embedding = self.embedding(tgt)
        
        # Add positional encoding
        tgt_embedding = self.positional_encoding(tgt_embedding)
        
        # Create causal mask for the current sequence length
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Pass through transformer decoder
        # For decoder-only, we don't have encoder output, so we use tgt as both inputs
        output = self.transformer_decoder(
            tgt_embedding,
            tgt_embedding,
            tgt_mask=mask
        )
        
        # Project to vocabulary size
        output = self.output_layer(output)
        
        return output
    
    def generate(self, input_ids, max_length, temperature=1.0):
        """
        Generate text autoregressively
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
        """
        batch_size = input_ids.size(0)
        
        # Start with the input sequence
        generated = input_ids
        
        # Generate tokens one by one
        for _ in range(max_length - input_ids.size(1)):
            # Get model predictions
            with torch.no_grad():
                outputs = self.forward(generated)
                
                # Get the next token logits (last token in sequence)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated

class TranslationDataset(Dataset):
    """
    Dataset for translation task
    """
    def __init__(self, source_texts, target_texts, source_tokenizer, target_tokenizer):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize
        source_tokens = self.source_tokenizer.encode(source_text)
        target_tokens = self.target_tokenizer.encode(target_text)
        
        return {
            'source': torch.tensor(source_tokens, dtype=torch.long),
            'target': torch.tensor(target_tokens, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    source_sequences = [item['source'] for item in batch]
    target_sequences = [item['target'] for item in batch]
    
    # Pad sequences
    source_padded = pad_sequence(source_sequences, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_sequences, batch_first=True, padding_value=0)
    
    return {
        'source': source_padded,
        'target': target_padded
    }

def train_translation_model(model, dataloader, optimizer, criterion, device, epochs=10):
    """
    Train the translation model
    """
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataloader:
            # Move data to device
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            outputs = model(source, target[:, :-1])  # Exclude last token from target for input
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            target = target[:, 1:].reshape(-1)  # Exclude first token (BOS) from target for loss
            
            # Calculate loss
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# Example usage
if __name__ == "__main__":
    # Define parameters
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # Print model summary
    print(model)
    
    # Example input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids)
    print(f"Output shape: {outputs.shape}")  # Should be [batch_size, seq_len, vocab_size]
    
    # Generate example
    generated = model.generate(input_ids, max_length=20)
    print(f"Generated shape: {generated.shape}")  # Should be [batch_size, max_length]
