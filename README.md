# PyTorch Transformer Models

This repository contains PyTorch implementations of:

1. A decoder-only Transformer for translation tasks
2. A Transformer-based sentiment classification model with 3 classes

## Files

- `decoder_only_transformer.py`: Implementation of a decoder-only Transformer for translation tasks
- `sentiment_classifier.py`: Implementation of a Transformer-based sentiment classification model
- `main.py`: Script to train and evaluate both models
- `README.md`: This file

## Requirements

```
torch>=1.9.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
seaborn>=0.11.0
```

You can install the requirements with:

```bash
pip install -r requirements.txt
```

## Decoder-Only Transformer for Translation

The decoder-only Transformer is implemented in `decoder_only_transformer.py`. It's based on the Transformer architecture from the paper "Attention is All You Need" but uses only the decoder part.

### Architecture

- Token embedding layer
- Positional encoding
- Multiple Transformer decoder layers
- Output linear layer

### Usage

```python
from decoder_only_transformer import DecoderOnlyTransformer

# Create model
model = DecoderOnlyTransformer(
    vocab_size=10000,
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

# Forward pass
outputs = model(input_ids)

# Generate text
generated = model.generate(input_ids, max_length=20)
```

## Sentiment Classification Model

The sentiment classification model is implemented in `sentiment_classifier.py`. It uses a Transformer encoder to classify text into 3 sentiment classes: negative, neutral, and positive.

### Architecture

- Token embedding layer
- Positional encoding
- Multiple Transformer encoder layers
- Global average pooling
- Classification head

### Usage

```python
from sentiment_classifier import SentimentTransformer, Vocabulary

# Create vocabulary
vocab = Vocabulary()
vocab.build_vocab()

# Create model
model = SentimentTransformer(
    vocab_size=len(vocab),
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    hidden_dim=512,
    dropout=0.1,
    num_classes=3
)

# Forward pass
outputs = model(input_ids, mask)
```

## Training and Evaluation

You can train and evaluate both models using the `main.py` script:

```bash
# Train and evaluate both models
python main.py --task both

# Train and evaluate only the translation model
python main.py --task translation

# Train and evaluate only the sentiment classification model
python main.py --task sentiment
```

### Command-line Arguments

- `--task`: Task to perform (`translation`, `sentiment`, or `both`)
- `--translation_data`: Path to translation data file
- `--sentiment_data`: Path to sentiment data file
- `--max_samples`: Maximum number of samples to use
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--save_dir`: Directory to save models

### Data Format

#### Translation Data

The translation data file should be a tab-separated file with source and target texts:

```
hello world\thola mundo
how are you\tcómo estás
thank you very much\tmuchas gracias
goodbye\tadiós
```

#### Sentiment Data

The sentiment data file should be a tab-separated file with text and sentiment label:

```
I love this movie, it's amazing!\t2
This film is terrible, I hated it.\t0
The movie was okay, nothing special.\t1
```

Where the labels are:
- 0: Negative
- 1: Neutral
- 2: Positive

## Example Usage

### Translation

```python
from decoder_only_transformer import DecoderOnlyTransformer, Vocabulary

# Create vocabularies
source_vocab = Vocabulary()
target_vocab = Vocabulary()

# Create model
model = DecoderOnlyTransformer(vocab_size=10000)

# Load trained model
model.load_state_dict(torch.load('best_translation_model.pt'))

# Translate text
source_text = "hello world"
translation = translate(model, source_text, source_vocab, target_vocab, device)
print(f"Translation: {translation}")
```

### Sentiment Analysis

```python
from sentiment_classifier import SentimentTransformer, Vocabulary

# Create vocabulary
vocab = Vocabulary()

# Create model
model = SentimentTransformer(vocab_size=len(vocab), num_classes=3)

# Load trained model
model.load_state_dict(torch.load('best_sentiment_model.pt'))

# Analyze sentiment
text = "I really enjoyed this movie, it was fantastic!"
sentiment, probs = analyze_sentiment(model, text, vocab, device)
print(f"Sentiment: {sentiment}")
print(f"Probabilities: Negative: {probs[0]:.4f}, Neutral: {probs[1]:.4f}, Positive: {probs[2]:.4f}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
