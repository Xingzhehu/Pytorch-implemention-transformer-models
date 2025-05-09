import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SentimentTransformer(nn.Module):
    """
    Transformer-based model for sentiment classification with 3 classes
    """
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=4, 
                 hidden_dim=512, dropout=0.1, max_seq_length=512, num_classes=3):
        super(SentimentTransformer, self).__init__()
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, max_seq_length, embed_dim)
        )
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize the parameters of the model"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Create embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Apply transformer encoder
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class Vocabulary:
    """
    Vocabulary class for text tokenization
    """
    def __init__(self, min_freq=2, max_size=None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2token = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.token_counts = Counter()
        self.built = False
        
    def add_token(self, token):
        """Add a token to the vocabulary"""
        self.token_counts[token] += 1
        
    def add_tokens(self, tokens):
        """Add multiple tokens to the vocabulary"""
        self.token_counts.update(tokens)
        
    def build_vocab(self):
        """Build the vocabulary from collected tokens"""
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by minimum frequency
        tokens = [(token, count) for token, count in sorted_tokens if count >= self.min_freq]
        
        # Limit vocabulary size if specified
        if self.max_size is not None:
            tokens = tokens[:self.max_size - len(self.token2idx)]
        
        # Add tokens to vocabulary
        for idx, (token, _) in enumerate(tokens, start=len(self.token2idx)):
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            
        self.built = True
        
    def tokenize(self, text):
        """Tokenize text into list of tokens"""
        # Simple whitespace tokenization
        return text.lower().split()
        
    def encode(self, text, add_special_tokens=True):
        """Convert text to token ids"""
        if not self.built:
            raise ValueError("Vocabulary has not been built yet")
            
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = ["<BOS>"] + tokens + ["<EOS>"]
            
        # Convert to ids
        ids = [self.token2idx.get(token, self.token2idx["<UNK>"]) for token in tokens]
        
        return ids
    
    def decode(self, ids, remove_special_tokens=True):
        """Convert token ids to text"""
        tokens = [self.idx2token.get(idx, "<UNK>") for idx in ids]
        
        # Remove special tokens
        if remove_special_tokens:
            tokens = [token for token in tokens if token not in ["<PAD>", "<BOS>", "<EOS>"]]
            
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.token2idx)

class SentimentDataset(Dataset):
    """
    Dataset for sentiment classification
    """
    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and encode
        tokens = self.vocab.encode(text)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        return {
            'text': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'mask': torch.tensor([1 if t != 0 else 0 for t in tokens], dtype=torch.bool)
        }

def train_sentiment_model(model, train_loader, val_loader, criterion, optimizer, 
                          device, epochs=10, scheduler=None):
    """
    Train the sentiment classification model
    """
    model.train()
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Move data to device
            texts = batch['text'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(texts, masks)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                texts = batch['text'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(texts, masks)
                loss = criterion(outputs, labels)
                
                # Track statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_sentiment_model.pt')
            print("Saved best model!")
        
        print("-" * 50)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model

def evaluate_model(model, test_loader, criterion, device, class_names=None):
    """
    Evaluate the sentiment classification model
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            texts = batch['text'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(texts, masks)
            loss = criterion(outputs, labels)
            
            # Track statistics
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate test metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Print classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(set(all_labels)))]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_acc, all_preds, all_labels

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example data (replace with your own dataset)
    texts = [
        "I love this movie, it's amazing!",
        "This film is terrible, I hated it.",
        "The movie was okay, nothing special.",
        "Absolutely fantastic, best film ever!",
        "Worst experience, complete waste of time.",
        "It was neither good nor bad, just average."
    ]
    
    # Labels: 0 = negative, 1 = neutral, 2 = positive
    labels = [2, 0, 1, 2, 0, 1]
    
    # Create vocabulary
    vocab = Vocabulary(min_freq=1)
    for text in texts:
        tokens = vocab.tokenize(text)
        vocab.add_tokens(tokens)
    
    vocab.build_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    dataset = SentimentDataset(texts, labels, vocab)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # Create model
    model = SentimentTransformer(
        vocab_size=len(vocab),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        hidden_dim=128,
        dropout=0.1,
        num_classes=3
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Train model
    model = train_sentiment_model(
        model, train_loader, val_loader, criterion, optimizer, 
        device, epochs=5, scheduler=scheduler
    )
    
    # Evaluate model
    class_names = ["Negative", "Neutral", "Positive"]
    evaluate_model(model, test_loader, criterion, device, class_names)
