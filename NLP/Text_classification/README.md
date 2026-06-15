# Text Classification 📚

Categorize documents into predefined classes or topics.

## 📚 Overview

Text classification assigns text to one of several predefined categories:
- **News articles**: Sports, Politics, Technology, Business
- **Email**: Spam vs Ham
- **Customer feedback**: Complaint, Suggestion, Praise
- **Product reviews**: Defective, Good, Excellent

---

## 📂 Files

- **rnn.py**: RNN-based classifier
- **utils.py**: Data loading and preprocessing utilities
- **data/**: Training and test datasets

---

## 🔧 Classification Process

```
Raw Text → Tokenization → Embedding → RNN/LSTM → Classification Layer → Output
                                                   (per-document)        ↓
                                              [Class 1, Class 2, ...]
```

**Key Difference from NER**:
- NER: Per-token classification (sequence labeling)
- Text Classification: Per-document classification (single output)

---

## 🎯 Model Architecture

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.rnn(embedded)  # hidden: [batch_size, hidden_dim]
        logits = self.fc(hidden.squeeze(0))  # [batch_size, output_dim]
        return logits
```

---

## 📊 Use Cases

1. **Spam Detection**: Classify emails as spam or legitimate
2. **Sentiment Analysis**: Positive, Negative, Neutral
3. **Topic Classification**: Categorize news articles
4. **Intent Classification**: Chatbots
5. **Customer Feedback**: Route to appropriate department

---

## 🛠️ Classification Approaches

### 1. Bag of Words
- Simple word counting
- Fast, interpretable
- Loss of word order

### 2. TF-IDF
- Weighted word importance
- Better than BoW
- Still loses order

### 3. RNN/LSTM
- Processes sequences
- Captures word order
- Better accuracy

### 4. Transformer-based
- BERT, RoBERTa
- State-of-the-art
- Requires fine-tuning

---

## 🚀 Quick Example

```python
# Train text classifier
model = TextClassifier(
    vocab_size=10000,
    embedding_dim=100,
    hidden_dim=256,
    output_dim=4  # 4 classes
)

for epoch in range(10):
    for texts, labels in train_loader:
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
```

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: RNN/LSTM, PyTorch, NLP basics