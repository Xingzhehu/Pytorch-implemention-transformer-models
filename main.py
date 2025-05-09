import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import time

# Import our models
from decoder_only_transformer import DecoderOnlyTransformer, TranslationDataset, collate_fn
from sentiment_classifier import SentimentTransformer, Vocabulary, SentimentDataset
from sentiment_classifier import train_sentiment_model, evaluate_model

def load_translation_data(file_path, max_samples=None):
    """
    Load translation data from a file
    Format: source_text\ttarget_text
    """
    source_texts = []
    target_texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            parts = line.strip().split('\t')
            if len(parts) == 2:
                source_texts.append(parts[0])
                target_texts.append(parts[1])
    
    return source_texts, target_texts

def load_sentiment_data(file_path, max_samples=None):
    """
    Load sentiment data from a file
    Format: text\tlabel
    """
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            parts = line.strip().split('\t')
            if len(parts) == 2:
                texts.append(parts[0])
                try:
                    labels.append(int(parts[1]))
                except ValueError:
                    # Skip invalid labels
                    continue
    
    return texts, labels

def create_vocabularies(source_texts, target_texts, min_freq=2, max_size=50000):
    """
    Create vocabularies for source and target languages
    """
    source_vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    target_vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    
    # Add tokens to vocabularies
    for text in source_texts:
        tokens = source_vocab.tokenize(text)
        source_vocab.add_tokens(tokens)
    
    for text in target_texts:
        tokens = target_vocab.tokenize(text)
        target_vocab.add_tokens(tokens)
    
    # Build vocabularies
    source_vocab.build_vocab()
    target_vocab.build_vocab()
    
    return source_vocab, target_vocab

def train_translation_model(model, train_loader, val_loader, criterion, optimizer, 
                           device, epochs=10, scheduler=None, save_path='best_translation_model.pt'):
    """
    Train the translation model
    """
    model.train()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move data to device
            source = batch['source'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            outputs = model(source, target[:, :-1])  # Exclude last token from target for input
            
            # Reshape for loss calculation
            outputs = outputs.reshape(-1, outputs.size(-1))
            target_labels = target[:, 1:].reshape(-1)  # Exclude first token (BOS) from target for loss
            
            # Calculate loss
            loss = criterion(outputs, target_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                source = batch['source'].to(device)
                target = batch['target'].to(device)
                
                # Forward pass
                outputs = model(source, target[:, :-1])
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_labels = target[:, 1:].reshape(-1)
                
                # Calculate loss
                loss = criterion(outputs, target_labels)
                
                # Track statistics
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Saved best model!")
        
        print("-" * 50)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Translation Model Training')
    plt.savefig('translation_training_curves.png')
    plt.show()
    
    return model

def translate(model, source_text, source_vocab, target_vocab, device, max_length=100):
    """
    Translate a source text to target language
    """
    model.eval()
    
    # Tokenize and encode source text
    source_tokens = source_vocab.encode(source_text)
    source_tensor = torch.tensor([source_tokens], dtype=torch.long).to(device)
    
    # Generate translation
    with torch.no_grad():
        generated = model.generate(source_tensor, max_length=max_length)
    
    # Decode translation
    translation = target_vocab.decode(generated[0].cpu().numpy())
    
    return translation

def analyze_sentiment(model, text, vocab, device):
    """
    Analyze sentiment of a text
    """
    model.eval()
    
    # Tokenize and encode text
    tokens = vocab.encode(text)
    
    # Pad or truncate
    max_length = 128
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    
    # Create mask
    mask = torch.tensor([[1 if t != 0 else 0 for t in tokens]], dtype=torch.bool).to(device)
    
    # Convert to tensor
    tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(tokens_tensor, mask)
        _, predicted = torch.max(outputs, 1)
    
    # Map prediction to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map[predicted.item()]
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    return sentiment, probs.cpu().numpy()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train and evaluate models')
    parser.add_argument('--task', type=str, choices=['translation', 'sentiment', 'both'], 
                        default='both', help='Task to perform')
    parser.add_argument('--translation_data', type=str, default='translation_data.txt',
                        help='Path to translation data file')
    parser.add_argument('--sentiment_data', type=str, default='sentiment_data.txt',
                        help='Path to sentiment data file')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='Maximum number of samples to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Perform tasks
    if args.task in ['translation', 'both']:
        print("\n" + "="*50)
        print("Translation Task")
        print("="*50)
        
        # Load translation data
        try:
            source_texts, target_texts = load_translation_data(args.translation_data, args.max_samples)
            print(f"Loaded {len(source_texts)} translation pairs")
        except FileNotFoundError:
            print(f"Translation data file not found: {args.translation_data}")
            print("Using example data instead")
            
            # Example data
            source_texts = [
                "hello world",
                "how are you",
                "thank you very much",
                "goodbye"
            ]
            target_texts = [
                "hola mundo",
                "cómo estás",
                "muchas gracias",
                "adiós"
            ]
        
        # Create vocabularies
        source_vocab, target_vocab = create_vocabularies(source_texts, target_texts)
        print(f"Source vocabulary size: {len(source_vocab)}")
        print(f"Target vocabulary size: {len(target_vocab)}")
        
        # Create dataset
        dataset = TranslationDataset(source_texts, target_texts, source_vocab, target_vocab)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            collate_fn=collate_fn
        )
        
        # Create model
        model = DecoderOnlyTransformer(
            vocab_size=max(len(source_vocab), len(target_vocab)),
            d_model=256,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1
        ).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        
        # Train model
        save_path = os.path.join(args.save_dir, 'best_translation_model.pt')
        model = train_translation_model(
            model, train_loader, val_loader, criterion, optimizer, 
            device, epochs=args.epochs, scheduler=scheduler, save_path=save_path
        )
        
        # Test translation
        test_source = "hello world"
        translation = translate(model, test_source, source_vocab, target_vocab, device)
        print(f"\nTest Translation:")
        print(f"Source: {test_source}")
        print(f"Translation: {translation}")
    
    if args.task in ['sentiment', 'both']:
        print("\n" + "="*50)
        print("Sentiment Classification Task")
        print("="*50)
        
        # Load sentiment data
        try:
            texts, labels = load_sentiment_data(args.sentiment_data, args.max_samples)
            print(f"Loaded {len(texts)} sentiment samples")
        except FileNotFoundError:
            print(f"Sentiment data file not found: {args.sentiment_data}")
            print("Using example data instead")
            
            # Example data
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
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
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
        save_path = os.path.join(args.save_dir, 'best_sentiment_model.pt')
        model = train_sentiment_model(
            model, train_loader, val_loader, criterion, optimizer, 
            device, epochs=args.epochs, scheduler=scheduler
        )
        
        # Evaluate model
        class_names = ["Negative", "Neutral", "Positive"]
        evaluate_model(model, test_loader, criterion, device, class_names)
        
        # Test sentiment analysis
        test_text = "I really enjoyed this movie, it was fantastic!"
        sentiment, probs = analyze_sentiment(model, test_text, vocab, device)
        print(f"\nTest Sentiment Analysis:")
        print(f"Text: {test_text}")
        print(f"Sentiment: {sentiment}")
        print(f"Probabilities: Negative: {probs[0]:.4f}, Neutral: {probs[1]:.4f}, Positive: {probs[2]:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
