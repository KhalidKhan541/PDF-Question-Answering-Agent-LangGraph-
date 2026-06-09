# Introduction to Transformer Architecture

## Overview

The Transformer is a deep learning model architecture introduced in the 2017 paper
"Attention Is All You Need" by Vaswani et al. It has become the foundation for
most modern large language models (LLMs) including GPT, BERT, and Claude.

## Core Components

### Self-Attention Mechanism

Self-attention allows each token in a sequence to attend to every other token.
For each token, three vectors are computed: Query (Q), Key (K), and Value (V).
The attention score between two tokens is computed as the dot product of Q and K,
scaled by the square root of the key dimension, then passed through a softmax.

### Multi-Head Attention

Instead of a single attention function, the Transformer uses multiple attention
"heads" running in parallel. Each head learns to attend to different aspects of
the input. The outputs are concatenated and linearly projected.

### Feed-Forward Network

After the attention layer, each position passes through an identical feed-forward
network consisting of two linear transformations with a ReLU activation in between.

### Positional Encoding

Since Transformers process all tokens in parallel (unlike RNNs), positional
encodings are added to the input embeddings to give the model information about
the order of tokens.

## Training

Transformers are trained using gradient descent with the Adam optimizer.
For language modeling, the objective is typically next-token prediction
(autoregressive) or masked token prediction (BERT-style).

## Advantages Over RNNs

- Parallelizable: all tokens are processed simultaneously
- Long-range dependencies: attention has O(1) path length between any two tokens
- Scalability: performance scales predictably with model and data size

## Limitations

- Quadratic complexity: self-attention scales as O(n²) with sequence length
- Large memory requirements for long contexts
- Requires large amounts of training data
