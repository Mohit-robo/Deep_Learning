# Weight Initialization Techniques 🎲

Methods for initializing neural network weights before training. Proper initialization is crucial for training stability and convergence speed.

## 📚 Overview

Why initialization matters:
- **Bad initialization** → Exploding/vanishing gradients
- **Good initialization** → Fast convergence, stable training
- **Different architectures** → Need different strategies

**Key File**: `init_weight.py` — Contains all initialization methods

---

## 🔧 Initialization Methods

### **Random Normal (Baseline)**

**Formula**: $W \sim \mathcal{N}(0, \sigma^2)$, typically $\sigma = 0.01$

**Code**:
```python
W = np.random.normal(0, 0.01, shape=(n_in, n_out))
```

**Characteristics**:
- ✅ Simple baseline
- ❌ Often suboptimal
- ❌ May cause vanishing/exploding gradients

**When to Use**: Rarely (mostly for comparison)

---

### **Xavier (Glorot) Initialization**

**Formula**: $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in} + n_{out}}})$

**Alternative (uniform)**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$

**Code**:
```python
from init_weight import Xavier
W = Xavier.initialize(shape=(n_in, n_out))
```

**Key Insight**: Scale depends on fan-in AND fan-out, keeps variance similar across layers

**Characteristics**:
- ✅ Balanced variance across layers
- ✅ Works well with Sigmoid/Tanh
- ❌ Suboptimal for ReLU networks
- ✅ Simple and widely used

**When to Use**: 
- Sigmoid/Tanh activation functions
- Pre-ReLU networks (legacy)

---

### **He (Kaiming) Initialization**

**Formula**: $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$

**Alternative (uniform)**: $W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$

**Code**:
```python
from init_weight import He
W = He.initialize(shape=(n_in, n_out))
```

**Key Insight**: Only depends on fan-in, designed for ReLU networks

**Why Different?**: ReLU kills half the activations (negative values → 0), so needs larger initial weights

**Characteristics**:
- ✅ Designed for ReLU networks
- ✅ Faster convergence
- ✅ Better gradient flow
- ✅ Modern default choice
- ❌ Too large for other activation functions

**When to Use**: 
- **DEFAULT for modern networks** (ReLU, Leaky ReLU, etc.)
- Pre-activation layers
- CNNs with ReLU

---

### **LeCun Initialization**

**Formula**: $W \sim \mathcal{N}(0, \sqrt{\frac{1}{n_{in}}})$

**Code**:
```python
from init_weight import LeCun
W = LeCun.initialize(shape=(n_in, n_out))
```

**Key Insight**: Similar to He but with factor of 1 instead of 2

**When to Use**: 
- LSTM networks
- Selu activation
- Some specialized architectures

---

### **Uniform Initialization**

**Formula**: $W \sim \mathcal{U}(a, b)$

**Where**: Bounds calculated per method (Xavier, He, etc.)

**Characteristics**:
- ✅ Non-Gaussian distribution
- ✅ Bounded values
- ⚠️ Performance similar to Gaussian for most cases

---

## 📊 Comparison Table

| Method | Formula | Best For | Fan | Factor |
|--------|---------|----------|-----|--------|
| **Xavier** | $\sqrt{\frac{2}{n_{in} + n_{out}}}$ | Sigmoid/Tanh | Both | 2 |
| **He** | $\sqrt{\frac{2}{n_{in}}}$ | ReLU networks | In | 2 |
| **LeCun** | $\sqrt{\frac{1}{n_{in}}}$ | LSTM, Selu | In | 1 |
| **Random** | $0.01$ | Baseline | None | - |

---

## 🎯 Decision Tree

```
Choosing initialization method:

Is network new/modern?
├─ YES → Use ReLU/Leaky ReLU?
│        ├─ YES → He initialization ✓
│        └─ NO → Xavier initialization
│
└─ NO → Sigmoid/Tanh?
         ├─ YES → Xavier initialization ✓
         └─ NO → He initialization
```

---

## 📐 Mathematical Explanation

### Why Initialization Matters

**Problem**: Poor initialization leads to gradient issues

```
Forward pass:
Z1 = W1 · X          (if W1 too large: Z1 very large)
A1 = sigmoid(Z1)     (if Z1 very large: A1 ≈ 1)
Z2 = W2 · A1         (if A1 ≈ 1: Z2 normal)

Backward pass:
∂L/∂W1 ∝ ∂sigmoid(Z1) · ...
```

If $Z_1$ is very large: $\sigma'(Z_1) \approx 0$ → **Vanishing gradients**
If $Z_1$ is very small: $\sigma'(Z_1)$ very large → **Exploding gradients**

### Xavier's Solution

For sigmoid/tanh, we want to keep activations in the linear region:
- Activations centered around 0
- Variance should be consistent across layers

For a layer: $\text{Var}(Z) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(X)$

If $X$ has $\text{Var}(X) = 1$ and we want $\text{Var}(Z) = 1$:
$$n_{in} \cdot \text{Var}(W) = 1 \Rightarrow \text{Var}(W) = \frac{1}{n_{in}}$$

### He's Modification

For ReLU, half of neurons are zero during forward pass.
To compensate:
$$\text{Var}(W) = \frac{2}{n_{in}}$$

This keeps gradients flowing even with ReLU's non-linearity.

---

## 📊 Convergence Comparison

**Training curves with different initializations**:

```
Loss vs Epoch

With Random (0.01):        With Xavier:              With He:
    ^                          ^                         ^
    |      ╱╲╲                 |    ╱╲                    |  ╱╲
    |    ╱╱  ╲╲╲               |  ╱╱  ╲                   |╱╱  ╲
    |  ╱╱      ╲╲╲             |╱╱    ╲                   |      ╲
    |╱╱          ╲╲╲           |      ╲                   |       ╲
    └─────────────────          └──────────               └────────────
    (slow, unstable)            (good)                    (best for ReLU)
```

---

## ✅ Exercises

1. **Visualize Distributions**: Plot weight distributions for each method
2. **Training Comparison**: Train same network with different initializations
3. **Convergence Speed**: Measure epochs to convergence
4. **Gradient Flow**: Trace gradient norms through network
5. **Stability Test**: Check for gradient explosion/vanishing
6. **Custom Method**: Design initialization for custom activation function
7. **Batch Normalization**: Compare with and without BatchNorm

---

## 🔗 Implementation Tips

```python
import numpy as np
import torch

# PyTorch built-in
torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
torch.nn.init.xavier_uniform_(layer.weight)

# Manual implementation
fan_in = weight.shape[1]
std = np.sqrt(2.0 / fan_in)  # He initialization
weight = np.random.normal(0, std, weight.shape)
```

---

## 📚 References

- [Xavier Initialization Paper](http://proceedings.mlr.press/v9/glorot10a.html)
- [He et al. Initialization](https://arxiv.org/abs/1502.01852)
- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)

---

**Difficulty**: 🟡 Intermediate
**Prerequisites**: Linear algebra, statistics, neural networks