# Natural Language Processing (NLP) Projects 📝

End-to-end NLP applications demonstrating how to build intelligent systems that understand and process human language. This module covers the complete pipeline from text preprocessing to model deployment.

## 📚 Overview

This module implements four foundational NLP tasks:
- **Chatbot**: Conversational AI with intent recognition
- **Named Entity Recognition**: Extract entities from text
- **Sentiment Analysis**: Classify text sentiment
- **Text Classification**: Categorize text into predefined classes

**Learning Objective**: Master practical NLP implementations and understand how to build production NLP systems.

---

## 📂 Directory Structure

```
NLP/
├── README.md                          # This file
├── chatbot/                           # Conversational AI system
│   └── v1/                           # Version 1 chatbot
│       ├── train.py                  # Training script
│       ├── chat.py                   # Inference/chat interface
│       ├── model.py                  # Model architecture
│       ├── intents.json              # Intent definitions
│       ├── data.pkl                  # Preprocessed data cache
│       ├── model.pth                 # Trained weights
│       ├── static/                   # CSS, JavaScript
│       └── templates/                # HTML templates
│
├── Named_entity_rec/                 # Entity recognition
│   ├── ner.ipynb                     # Standard NER implementation
│   └── ner_custom.ipynb              # Custom NER training
│
├── Sentiment_Analysis/               # Sentiment classification
│   ├── sentiments.ipynb              # Complete sentiment pipeline
│   └── data/                         # Training datasets
│
└── Text_classification/              # Text categorization
    ├── rnn.py                        # RNN-based classifier
    ├── utils.py                      # Utility functions
    └── data/                         # Training/test data
```

---

## 🎯 Project Details

### **Chatbot V1** — AI Conversational Agent

A fully functional chatbot that understands user intent and responds appropriately.

**Architecture**:
```
User Input → Tokenize → Intent Classification (Neural Network) → Response Generation → Output
```

**Components**:
- **Model**: Feedforward neural network for intent classification
- **Intents**: JSON file defining conversation patterns
- **Training**: NLTK tokenization + custom neural network
- **Deployment**: Flask web app with chat interface

**Key Files**:
- `train.py`: Trains model on intents.json
- `chat.py`: Interactive command-line chat
- `model.py`: Neural network architecture
- `intents.json`: Define conversation patterns and responses

**How It Works**:
1. User enters message
2. Model tokenizes and converts to embeddings
3. Neural network predicts intent
4. System returns appropriate response
5. Context maintained for conversational flow

**Example Intent Structure** (in intents.json):
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hello", "Hi", "Hey"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    }
  ]
}
```

**Learn**: Text preprocessing, intent classification, chatbot design patterns

---

### **Named Entity Recognition** — Entity Extraction

Extract and classify named entities (people, places, organizations) from text.

**Tasks**:
- `ner.ipynb`: Standard NER using existing models/libraries
- `ner_custom.ipynb`: Train custom NER on labeled data

**NER Process**:
```
Input: "Apple Inc. was founded by Steve Jobs in California"
Output: [COMPANY: Apple Inc.] [PERSON: Steve Jobs] [LOCATION: California]
```

**Learn**: Sequence labeling, token classification, transfer learning for NLP

---

### **Sentiment Analysis** — Opinion Mining

Classify text as positive, negative, or neutral.

**Pipeline**:
1. Load sentiment dataset
2. Preprocess text (tokenization, lowercasing, removing stopwords)
3. Convert to embeddings or vectors
4. Train classifier (Naive Bayes, SVM, Neural Network)
5. Evaluate on test set
6. Deploy for inference

**Applications**:
- Review analysis
- Social media monitoring
- Customer feedback analysis
- Market sentiment tracking

**Example**:
```
Input: "This movie was absolutely terrible!"
Output: Negative (confidence: 0.92)
```

**Learn**: Text classification, embedding techniques, model evaluation

---

### **Text Classification** — Document Categorization

Categorize documents into predefined classes.

**Key Files**:
- `rnn.py`: RNN-based text classifier
- `utils.py`: Data loading and preprocessing utilities
- `data/`: Training and test datasets

**Model Architecture**:
```
Text → Embedding → RNN/LSTM → Pooling → Dense Layers → Classification
```

**Typical Use Cases**:
- News categorization (Sports, Politics, Tech, etc.)
- Email spam detection
- Question routing (which department should answer)
- Document classification

**Learn**: RNN text processing, sequence models, multi-class classification

---

## 🚀 Quick Start

### Run the Chatbot

```bash
# Train the chatbot
cd chatbot/v1
python train.py

# Chat with the bot
python chat.py

# Or launch web interface
python app.py
# Visit http://localhost:5000
```

### Sentiment Analysis

```bash
# Open and run the notebook
jupyter notebook Sentiment_Analysis/sentiments.ipynb
```

### Text Classification with RNN

```bash
# Train RNN classifier
cd Text_classification
python rnn.py
```

### Named Entity Recognition

```bash
# Run custom NER training
jupyter notebook Named_entity_rec/ner_custom.ipynb
```

---

## 🛠️ NLP Pipeline Components

### 1. Text Preprocessing
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = "This is a sample text!"
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words('english'))
filtered = [t for t in tokens if t not in stop_words]
```

### 2. Text Vectorization
- **Bag of Words**: Count occurrences of each word
- **TF-IDF**: Term frequency-inverse document frequency
- **Word Embeddings**: Word2Vec, GloVe (distributed representations)
- **Contextual Embeddings**: BERT, GPT (transformer-based)

### 3. Model Training
```python
# Example: Sentiment analysis
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC(kernel='linear'))
])
model.fit(X_train, y_train)
```

### 4. Inference & Deployment
```python
# Predict on new text
predictions = model.predict(["Great product!", "Terrible experience"])
```

---

## 📊 Learning Progression

### **Phase 1: NLP Fundamentals** (Week 1)
- Understand text preprocessing
- Learn tokenization and vectorization
- Study NLTK basics
- Run sentiment analysis notebook

### **Phase 2: Classification Models** (Week 2)
- Implement text classification with RNN
- Study different text representations
- Understand loss functions and metrics
- Compare model performance

### **Phase 3: Advanced NLP** (Week 3)
- Train custom NER models
- Build sequence labeling systems
- Explore entity extraction techniques
- Study attention mechanisms (optional)

### **Phase 4: Conversational AI** (Week 4)
- Design chatbot intent structure
- Implement intent classification
- Build response generation
- Deploy as web application

---

## 🎓 Key Concepts

### Text Preprocessing
- **Tokenization**: Split text into tokens (words, sentences)
- **Lowercasing**: Normalize text case
- **Stopword Removal**: Remove common words
- **Stemming/Lemmatization**: Reduce words to base form

### Feature Representation
- **One-hot Encoding**: Binary vector per word
- **Bag of Words**: Count-based representation
- **TF-IDF**: Weighted frequency-based
- **Dense Embeddings**: Learned representations

### Classification Approaches
- **Naive Bayes**: Probabilistic baseline
- **SVM**: Support Vector Machine
- **Logistic Regression**: Linear classifier
- **Neural Networks**: Deep learning approaches
- **RNN/LSTM**: Sequence-aware models

### Evaluation Metrics
- **Accuracy**: Correct predictions / total
- **Precision**: True positives / predicted positives
- **Recall**: True positives / actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating curve

---

## ✅ Exercises

1. **Extend Chatbot**: Add new intents and conversation patterns
2. **Sentiment on New Dataset**: Train sentiment classifier on movie reviews
3. **Custom NER**: Label custom entity types and train
4. **Multi-language**: Adapt text classification for another language
5. **Production Deployment**: Build REST API for chatbot
6. **Model Comparison**: Compare multiple classifiers on same task
7. **Feature Engineering**: Experiment with different text representations
8. **Error Analysis**: Analyze misclassifications and improve model

---

## 📚 NLP Concepts by Difficulty

| Concept | Difficulty | Module |
|---------|-----------|--------|
| Tokenization | 🟢 | All |
| Stopword Removal | 🟢 | All |
| Bag of Words | 🟡 | Text Classification |
| Embeddings | 🟡 | Sentiment Analysis |
| Intent Classification | 🟡 | Chatbot |
| Sequence Labeling | 🟡 | NER |
| Attention Mechanisms | 🔴 | Advanced (not covered) |
| Transformer Models | 🔴 | Advanced (not covered) |

---

## 🔗 References

- [NLTK Documentation](https://www.nltk.org/)
- [spaCy NLP Library](https://spacy.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [NLP Overview (Medium)](https://medium.com/nanonets/nlp-tutorial-named-entity-recognition-ner-with-nltk-46ebf1de3b37)

---

## ⚠️ Common Challenges

**Challenge**: Limited training data
- **Solution**: Use transfer learning, data augmentation

**Challenge**: Language variations and slang
- **Solution**: Expand vocabulary, fine-tune on domain data

**Challenge**: Context dependency
- **Solution**: Use RNN/LSTM, transformer models

**Challenge**: Multiple languages
- **Solution**: Use multilingual models, language detection

---

## 🔜 Next Steps

After mastering this module:
- Explore advanced architectures: **Transformers, BERT, GPT** (not in scope)
- Apply to real applications: Customer support, content moderation
- Study other NLP tasks: **Machine translation, summarization, QA**
- Combine with [YOLO](../YOLO/) for **multimodal systems** (vision + language)

---

**Last Updated**: June 2026
**Difficulty**: 🟡 Intermediate
**Prerequisites**: Python, basic ML concepts, [Pytorch_series](../Pytorch_series/) recommended