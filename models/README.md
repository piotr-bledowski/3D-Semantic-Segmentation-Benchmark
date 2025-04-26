# MODELS
This dir contains models architectures to test

## PointNet
It is implemeted as described in the article (check out for reference the main README.md), but two features are important:
1. Model returns class probabilities (after softmax)
2. Model expects data in format - chanel first (batch size, n_dim (eg. 6 for 3 spatial dim + RGB), n_samples)
