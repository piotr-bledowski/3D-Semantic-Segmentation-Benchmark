# PointNet Segmentation Training

This repository provides:

1. **Core training utilities** (`Trainig/train_model.py`)  
2. **Example “end-to-end” script** (`models/PointNet/train_model.py`)

Whether you have your own 3D point-cloud data or want to reproduce S3DIS experiments, these scripts make it easy to preprocess, train and evaluate a PointNet segmentation model.

---
## 🛠 Core Components
1. train_model.py
preprocess_batch_to_train_format(...)
• Pads or truncates each point-cloud to a fixed length
• (Optional) Randomly samples a fraction of points
• One-hot encodes per-point labels
• Returns (points_tensor, labels_onehot, lengths, cont_flag)

masked_onehot_cross_entropy(logits, targets, pad_starts)
Cross-entropy loss that ignores padding tokens.
Computes −∑_true_class log p / (# real points).

accuracy_from_one_hot(labels, preds)
Computes classification accuracy (%) over all non-padded points.

train_epoch(...) & evaluate(...)
Single-epoch training and validation loops.

train_model(model, train_loader, test_loader, mapping, …)
High-level trainer:

Moves model to GPU/CPU

Runs multiple epochs

(Optional) Records & pickles training/validation metrics

Returns the trained model

It uses a fixed Adam optimizer, and no rl scheduling.