# NLP Chatbot - Conversational AI 🤖

A complete AI chatbot implementation with intent recognition and response generation using neural networks.

## 📚 Overview

Full-stack conversational AI system with:
- Intent classification neural network
- Multi-turn conversation support
- Web interface with Flask
- JSON-based intent configuration

---

## 📂 Directory Structure

```
v1/
├── train.py           # Training script
├── chat.py            # CLI chat interface
├── model.py           # Neural network architecture
├── intents.json       # Intent definitions
├── data.pkl           # Preprocessed vocabulary
├── model.pth          # Trained model weights
├── app.py             # Flask web app (optional)
├── templates/         # HTML templates
└── static/           # CSS & JavaScript
```

---

## 🚀 Quick Start

### Train the Bot
```bash
python train.py
```

Outputs:
- `data.pkl`: Vocabulary pickle file
- `model.pth`: Trained model weights

### Chat with the Bot
```bash
python chat.py
# You: Hello
# Bot: Hello! How can I help?
```

### Run Web Interface
```bash
python app.py
# Visit http://localhost:5000
```

---

## 🔧 How It Works

### 1. Intent Definition (intents.json)

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey", "Good morning"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "Goodbye", "See you later"],
      "responses": ["Goodbye!", "See you soon!", "Bye!"]
    }
  ]
}
```

### 2. Data Preprocessing (train.py)

```python
# Tokenize patterns
# Example: "Hello there" → ["hello", "there"]

# Build vocabulary
# Unique words across all patterns

# Encode patterns as one-hot vectors
# Each word gets unique index
```

### 3. Model Architecture (model.py)

```
Input (vocab_size) → Dense(128) → ReLU → Dense(64) → ReLU → Output (num_intents)
```

### 4. Training

```python
optimizer = Adam(learning_rate=0.001)
loss_fn = CrossEntropyLoss()

for epoch in range(200):
    for pattern, intent_idx in training_data:
        prediction = model(pattern)
        loss = loss_fn(prediction, intent_idx)
        loss.backward()
        optimizer.step()
```

### 5. Inference

```python
user_input = "Hi there!"
# Tokenize: ["hi", "there"]
# Encode: [1, 0, 0, ..., 1, 0, ...]
# Predict: [0.02, 0.95, 0.03]  (probabilities)
# Intent: "greeting" (max probability)
# Response: Random from greeting responses
```

---

## 📊 Intent Structure

Each intent can have:
- **tag**: Unique identifier
- **patterns**: User input variations
- **responses**: Bot responses
- **context**: Optional conversation context
- **context_set**: Set context for next turn

---

## 🎯 Example Intents

### Greeting Intent
```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good afternoon"],
  "responses": ["Hello!", "Hi there!", "Greetings!", "How can I help?"]
}
```

### Question Intent
```json
{
  "tag": "help",
  "patterns": ["Can you help me?", "I need help", "Could you assist me?"],
  "responses": ["Of course! What do you need help with?", "Sure! Tell me what you need."]
}
```

### Multi-turn Intent
```json
{
  "tag": "pizza_order",
  "patterns": ["I want to order pizza", "Order me a pizza"],
  "responses": ["What size pizza would you like?"],
  "context_set": "ordering"
}
```

---

## 💻 File Descriptions

### **train.py**
- Loads intents from JSON
- Tokenizes patterns with NLTK
- Builds vocabulary
- Trains neural network
- Saves model and data

```python
python train.py
# Output: model.pth, data.pkl
```

### **chat.py**
- Command-line interface
- Loads trained model
- Takes user input
- Predicts intent
- Returns response

```bash
python chat.py
> You: Hello
< Bot: Hello! How can I help?
```

### **model.py**
- PyTorch `nn.Module` implementation
- 3-layer fully connected network
- Softmax output layer

```python
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out
```

### **intents.json**
- JSON file defining all intents
- User patterns (variations of user input)
- Bot responses (reply options)

---

## ✅ Exercises

1. **Add New Intents**: Add weather, jokes, or help intents
2. **Multi-turn Conversations**: Implement context tracking
3. **Response Randomness**: Add variety to responses
4. **Error Handling**: Handle low confidence predictions
5. **Web Deployment**: Deploy with Flask
6. **Logging**: Log conversations
7. **Performance**: Measure response latency
8. **Enhancement**: Add NER for entity extraction

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: PyTorch, NLTK, JSON basics