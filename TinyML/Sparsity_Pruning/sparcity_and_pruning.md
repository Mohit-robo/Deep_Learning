# Sparsity and Pruning Experiment Results

**Initial Evaluation:**
* Dense model has accuracy = `92.95%`

## Experiment 1

**Pruning Configuration:**
```python
{
    'backbone.conv0.weight': 0.4,
    'backbone.conv1.weight': 0.6,
    'backbone.conv2.weight': 0.6,
    'backbone.conv3.weight': 0.6,
    'backbone.conv4.weight': 0.6,
    'backbone.conv5.weight': 0.6,
    'backbone.conv6.weight': 0.6,
    'backbone.conv7.weight': 0.8,
    'classifier.weight': 0
}
```

**Results:**
* Sparse model has accuracy = `89.09%` before finetuning
* Sparse model has accuracy = `92.87%` after finetuning (Finetuning Fine-grained Pruned Sparse Model)
* Sparse model has size = `12.29 MiB` (34.91% of dense model size)

**Sorting Check:**
* Dense model has accuracy = `92.95%`
* Pruned model has accuracy = `92.95%`
* Before sorting... dense model has accuracy = `92.95%`
* After sorting... sorted model has accuracy = `92.95%`
* Check passed.

**Fine-tuning post pruning:**
* Epoch 1 Accuracy `92.93%` / Best Accuracy: `92.93%`
* Epoch 2 Accuracy `93.07%` / Best Accuracy: `93.07%`
* Epoch 3 Accuracy `92.97%` / Best Accuracy: `93.07%`
* Epoch 4 Accuracy `92.96%` / Best Accuracy: `93.07%`
* Epoch 5 Accuracy `93.02%` / Best Accuracy: `93.07%`

**Performance Comparison:**

| Metric | Original | Pruned | Reduction Ratio |
| :--- | :--- | :--- | :--- |
| Latency (ms) | 24.4 | 20.5 | 1.2 |
| MACs (M) | 606 | 606 | 1.0 |
| Param (M) | 9.23 | 9.23 | 1.0 |

---

## Experiment 2

**Pruning Configuration:**
```python
{
    'backbone.conv0.weight': 0.15,
    'backbone.conv1.weight': 0.35,
    'backbone.conv2.weight': 0.45,
    'backbone.conv3.weight': 0.45,
    'backbone.conv4.weight': 0.65,
    'backbone.conv5.weight': 0.70,
    'backbone.conv6.weight': 0.75,
    'backbone.conv7.weight': 0.85,
    'classifier.weight': 0.85
}
```

**Results:**
* Sparse model has size = `9.94 MiB` (28.25% of dense model size)
* Sparse model has accuracy = `89.82%` before finetuning
* Sparse model has accuracy = `92.84%` after finetuning

**Sorting Check:**
* Before sorting... dense model has accuracy = `92.95%`
* After sorting... sorted model has accuracy = `92.95%`
* Check passed.

**Finetune the fine-grained pruned model:**
* Epoch 1 Accuracy `92.73%` / Best Accuracy: `92.73%`
* Epoch 2 Accuracy `92.74%` / Best Accuracy: `92.74%`
* Epoch 3 Accuracy `92.70%` / Best Accuracy: `92.74%`
* Epoch 4 Accuracy `92.84%` / Best Accuracy: `92.84%`
* Epoch 5 Accuracy `92.78%` / Best Accuracy: `92.84%`

**Channel Pruning Evaluation:**
* Pruned model has accuracy = `28.14%`
* Without sorting... pruned model has accuracy = `28.14%`
* With sorting... pruned model has accuracy = `36.81%`

**Fine-tuning post pruning:**
* Epoch 1 Accuracy `91.59%` / Best Accuracy: `91.59%`
* Epoch 2 Accuracy `91.82%` / Best Accuracy: `91.82%`
* Epoch 3 Accuracy `92.07%` / Best Accuracy: `92.07%`
* Epoch 4 Accuracy `92.11%` / Best Accuracy: `92.11%`
* Epoch 5 Accuracy `92.29%` / Best Accuracy: `92.29%`

**Performance Comparison:**

| Metric | Original | Pruned | Reduction Ratio |
| :--- | :--- | :--- | :--- |
| Latency (ms) | 23.6 | 11.4 | 2.1 |
| MACs (M) | 606 | 305 | 2.0 |
| Param (M) | 9.23 | 5.01 | 1.8 |

**Final Results:**
* Sparse model has size = `19.13 MiB` (54.33% of dense model size)
* Sparse model has accuracy = `92.95%` after finetuning
