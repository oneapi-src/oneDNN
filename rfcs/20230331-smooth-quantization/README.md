Proposal for Smooth quantization support
========================================

## 1. Introduction

The motivation for Smooth quantization is to speed up inference and reduce size
of Large Language Models (LLM) [[#1]][].
Traditional quantization approaches do not help LLMs:
- per-tensor quantization does not work, because magnitude of activation is
different across feature dimension.
- per-channel quantization makes linear operations (matmul) difficult to compute fast:
  - Modern CPUs and GPUs implement s8s8s32 Matrix-Matrix multiplication instructions,
    so input per-channel f32 scales can not be applied to s8 inputs before
    Linear, because it will make inputs to not fit into s8. On the
    other side, since Linear operation sums across input channels an additional
    Reduction operation should be computed between a vector of f32 scales and s8 weights.

Smooth quantization is a new approach for Large Language Models (LLM) quantization
that allows to achieve size reduction, speed up the model and keep accuracy on
par with f32 version. This is achieved by a per-channel scaling transformation
that smoothes the magnitude of activation across channels, making the model
quantization-friendly.

The algorithm:
- Post-training calibration:
  For each Linear operation:
    1. `max(|X_j|)` and `max(|W_j|)` are computed. Where X is activation, W is weights, and j is an input channel index.
    2. Smoothing factor `s_j` is computed as `s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)`. `alpha`
       is a hyper-parameter that depends on the model architecture.
    3. A per-channel transformation is applied:
        - Activation is divided by `s_j`.
        - Weights are multiplied by `s_j`.
- Inference:
  For each Linear operation:
      1. A per-channel transformation is applied:
        - Activation is divided by `s_j`.


## 2. Implications for oneDNN

Once Smooth quantization is applied to a model there is a scale operation added
before each int8 Linear operation comparing to regular quantization flow. To
avoid overhead oneDNN could fuse scale operation into a previous operation.
LLMs are based on the transformer architecture so there are two operations that
will need scale post-operation support:
1. Layer normalization. This operation precedes Linear operations in the self-attention block.
2. Softmax. This operation precedes Batched Matmul operation in the self-attention block.

** NOTE**:
- Softmax and Layer normalization can be in f32, f16 or bf16 data types to preserve accuracy,
  so the feature can not be covered by oneDNN quantization.


## 3. Proposal

The proposal is to add a binary post-operation to Softmax and Layer normalization
primitives.

## References

1. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models][1]

[1]: https://arxiv.org/pdf/2211.10438.pdf

---

EOD
