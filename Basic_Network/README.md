# Basic Network Implementations 🔧

Foundational implementations that explain how neural networks work from first principles. This directory is perfect for understanding the mathematical foundations before diving into frameworks like PyTorch.

## 📚 Overview

This module provides **ground-up implementations** of core neural network concepts:
- Complete neural network with forward/backward propagation
- Mathematical implementations of activation functions
- Multiple weight initialization strategies
- Gradient descent optimization algorithms
- Utility functions for dataset handling and visualization

**Learning Objective**: Understand the "why" behind neural network operations before using high-level frameworks.

---

## 📂 Directory Structure

```
Basic_Network/
├── NN.py                              # Core neural network implementation
├── utlis.py                           # Utility functions and helpers
├── README.md                          # This file
├── Activation Functions/              # Activation function implementations
│   ├── act_function.py               # All activation functions with derivatives
│   └── README.md                     # Detailed activation functions guide
├── weight_initialzation_techniques/  # Weight initialization strategies
│   ├── init_weight.py                # He, Xavier, LeCun initialization
│   └── README.md                     # Initialization techniques guide
└── gradient_descent/                 # Optimization algorithms
    └── README.md                     # Gradient descent variants guide
```

---

## 🎯 Key Files Explained

### **NN.py** — Core Neural Network Implementation

Complete feedforward neural network with:
- **Forward propagation**: Compute predictions from inputs
- **Backward propagation**: Compute gradients for all parameters
- **Parameter updates**: Gradient descent step
- **Kaiming initialization**: He weight initialization for ReLU networks

```python
# Example usage
from NN import NeuralNetwork

# Create network: 784 → 128 → 64 → 10 (MNIST)
nn = NeuralNetwork([784, 128, 64, 10])

# Forward pass
output = nn.forward(X)

# Backward pass
nn.backward(X, y, learning_rate=0.01)
```

**What You'll Learn**:
- How matrices are transformed through layers
- How gradients flow backward through the network
- Weight update mechanics

---

### **[Activation Functions](Activation%20Functions/)** — Non-linearity Implementations

Implements all common activation functions with mathematical explanations.

**Covered Functions**:

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | max(0, x) | Hidden layers (most popular) |
| **Leaky ReLU** | x if x > 0 else 0.01x | Solves dying ReLU problem |
| **Sigmoid** | 1/(1+e^-x) | Binary classification output |
| **Tanh** | (e^x - e^-x)/(e^x + e^-x) | Hidden layers (zero-centered) |
| **Softmax** | e^xi / Σe^xj | Multi-class classification |
| **Linear** | x | Regression output layer |

Each includes:
- Forward pass computation
- Derivative for backpropagation
- When and why to use it

**See**: [Activation Functions/README.md](Activation%20Functions/README.md)

---

### **[Weight Initialization Techniques](weight_initialzation_techniques/)** — Initialization Strategies

Different ways to initialize network weights, affecting training stability and speed.

**Covered Methods**:

| Method | Formula | Best For |
|--------|---------|----------|
| **He Initialization** | W ~ N(0, 2/n_in) | ReLU networks |
| **Xavier Initialization** | W ~ N(0, 1/n_in) | Sigmoid/Tanh networks |
| **LeCun Initialization** | W ~ N(0, 1/n_in) | LSTM networks |
| **Random Normal** | W ~ N(0, 0.01) | Baseline (often suboptimal) |

**Key Insight**: Wrong initialization can cause vanishing/exploding gradients or slow convergence.

**See**: [weight_initialzation_techniques/README.md](weight_initialzation_techniques/README.md)

---

### **[Gradient Descent](gradient_descent/)** — Optimization Fundamentals

Understanding how parameter optimization works.

**Topics Covered**:
- Batch gradient descent
- Stochastic gradient descent (SGD)
- Mini-batch gradient descent
- Learning rate effects

**See**: [gradient_descent/README.md](gradient_descent/README.md)

---

## 🚀 Quick Start

### Running the Core Network

```bash
# Train a simple network
python NN.py

# Expected output: Network converges on training task
```

### Understanding Activation Functions

```python
from Activation_Functions.act_function import ReLU, Sigmoid, Softmax

# Create activation instances
relu = ReLU()
sigmoid = Sigmoid()

# Forward pass
y = relu.forward(x)

# Backward pass (gradient)
dy = relu.backward(dy_out)
```

### Exploring Weight Initialization

```python
from weight_initialzation_techniques.init_weight import He, Xavier

# Initialize weights
he_weights = He.initialize(shape=(784, 128))
xavier_weights = Xavier.initialize(shape=(784, 128))
```

---

## 📊 Learning Progression

### Level 1: Understand the Math
1. Read `NN.py` to understand network structure
2. Study activation functions and their derivatives
3. Understand weight initialization impact

### Level 2: Experiment & Modify
1. Change network architecture in `NN.py`
2. Try different activation functions
3. Experiment with different weight initializations
4. Observe how each affects training

### Level 3: Implement Extensions
1. Implement batch normalization
2. Add dropout regularization
3. Implement different optimizers (Adam, RMSprop)
4. Add convolutional layers

---

## 🎓 Key Concepts

### Forward Propagation
```
Input → Layer1 → Activation → Layer2 → Activation → Output
  ↓      (W·x+b)   ReLU      (W·x+b)    Softmax      ↓
  X                                                   Predictions
```

### Backward Propagation
```
Loss → ∂L/∂W2 → ∂L/∂b2 → ∂L/∂A1 → ∂L/∂W1 → ∂L/∂b1
        Update    Update            Update    Update
```

### Why Activation Functions?
Without non-linearity, stacked layers become just one linear transformation:
```
(W3 · (W2 · (W1 · X))) = (W3·W2·W1) · X  # Still linear!
```

With activation functions:
```
σ(W3 · σ(W2 · σ(W1 · X)))  # Non-linear, can learn any function
```

---

## ✅ Exercises

1. **Modify Architecture**: Change the network to have 3 hidden layers instead of 2
2. **Test Initializations**: Compare training curves with He vs. Xavier initialization
3. **Gradient Checking**: Implement numerical gradient check to verify backpropagation
4. **Visualize Activations**: Plot activation function shapes and their derivatives
5. **Experiment with Learning Rates**: Find optimal learning rate for convergence

---

## 📖 Resources

- [Neural Networks from Scratch (Victor Zhou)](https://victorzhou.com/blog/intro-to-neural-networks/)
- [3Blue1Brown - Neural Networks Essence](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_LFPM5VHe)
- [Deep Learning Book - Part 1](http://www.deeplearningbook.org/)

---

## 🔗 Next Steps

After mastering this module:
- Move to **[Pytorch_series](../Pytorch_series/)** for modern framework implementations
- Study **[Paper_review_notes](../Paper_review_notes/)** for advanced architectures
- Explore **[YOLO](../YOLO/)** for practical applications

---

## ⚠️ Important Notes

- These implementations prioritize **clarity over efficiency**
- For production use, always use PyTorch/TensorFlow
- This module is best used alongside coursework or tutorials
- Running on CPU is fine for small networks; GPU recommended for larger experiments

---

**Last Updated**: June 2026
**Difficulty**: 🔴 Beginner-Intermediate
**Prerequisites**: Linear algebra, calculus, Python basics