# Sentiment Analysis 😊

Classify text as positive, negative, or neutral sentiment.

## 📚 Overview

Sentiment analysis (opinion mining) determines the emotional tone of text:
- **Positive**: "This movie is fantastic!"
- **Negative**: "Terrible experience, would not recommend"
- **Neutral**: "The weather is cloudy today"

---

## 📂 Files

- **sentiments.ipynb**: Complete sentiment analysis pipeline
- **data/**: Training and test datasets

---

## 🔧 Sentiment Pipeline

```
Raw Text → Preprocessing → Vectorization → Classification → Output
                                                              ↓
                                           [Positive, Negative, Neutral]
```

**Steps**:
1. Load dataset
2. Clean text (lowercase, remove punctuation)
3. Convert to vectors (TF-IDF, embeddings)
4. Train classifier (Naive Bayes, SVM, Neural Network)
5. Evaluate on test set
6. Deploy for inference

---

## 🎯 Common Datasets

- **Movie Reviews**: IMDB sentiment
- **Twitter Sentiment**: Tweet classification
- **Amazon Reviews**: Product sentiment
- **Product Reviews**: E-commerce feedback

---

## 🛠️ Implementation Approaches

### 1. Rule-based
- Dictionary of sentiment words
- Count positive vs negative
- Fast but limited

### 2. Traditional ML
- TF-IDF + Naive Bayes
- Works well for basic tasks
- Good baseline

### 3. Deep Learning
- CNN or RNN for text
- Better accuracy
- Requires more data

---

## 📊 Evaluation Metrics

- **Accuracy**: Correct predictions / total
- **Precision**: How many predicted positive are actually positive
- **Recall**: How many actual positive were found
- **F1-Score**: Harmonic mean (balanced metric)

---

## 🚀 Quick Example

```python
# Sentiment classification
sentiments = ["This is great!", "Absolutely horrible", "It's okay"]

for text in sentiments:
    # Preprocess, vectorize, classify
    sentiment = classifier.predict(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}\n")
```

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: Text processing, ML basics, PyTorch