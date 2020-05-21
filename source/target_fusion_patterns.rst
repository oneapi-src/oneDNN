======================
Target Fusion Patterns
======================

In this section, we list some common fusion patterns from popular workloads as examples to show that there are abundant fusion opportunities from real-world DL models.

**Pattern 1: Op XYZ + A chain of element-wise ops**

Here Op XYZ could be any op like Conv, Matmul or even another element-wise op. The chain of element-wise ops is appended as post-ops to Op XYZ, i.e. we insert the chain of element-wise operations to each output element of Op XYZ before storing it to memory.
Examples:
* Common pattern in CNN models: Conv + ReLU, Conv + Add + ReLU, Deconv + ReLU, Softmax + Dropout,
* MLP (recommendation models etc.): Matmul + ReLU
* BERT:  Matmul + Dropout + Add + Pre-norm of Layer Norm (Pattern 1 + Pattern 3), Post-norm of Layer Norm + Matmul + ReLU/GeLU (Pattern 1 + Pattern 2)
* LSTM: MatMul + Tanh/Sigmoid + Add + Mul + Tanh + Mul

**Pattern 2: A chain of element-wise ops + Op XYZ**

Here Op XYZ could be any op like Conv, Matmul or even another element-wise op. The chain of element-wise ops is prepended as pre-ops to Op XYZ, i.e. we insert the chain of element-wise operations to each input element of Op XYZ after loading input and before computation. This may introduce redundant computation of pre-ops because Op XYZ might load same input elements multiple times but this would favor HW with much more compute power than memory bandwidth.
Examples:
* Pre-activation CNN: Post-norm of BatchNorm + ReLU (+Add) + Conv + Pre-norm of BatchNorm (Pattern 2 + Pattern 3)

**Pattern 3: Op XYZ + Reduction part of normalization op**

Here Op XYZ could be any op like Conv, Matmul etc. Normalization op could be like Batch Normalization or Layer Normalization. These normalization ops usually do reduction ops followed by element-wise ops. We break the normalization op into two parts: Pre-norm which does the reduction ops and post-norm which does element-wise ops. The reduction ops are executed side-by-side with the output store of Op XYZ, e.g. for Batch/Layer Normalization, it is the mean and variance. Then the reduction results are input to the following element-wise ops which can be further fused with Pattern 2.
Examples: See examples from Pattern 1 and 2.

**Pattern 4: Op XYZ + Shape transformation ops**

Here Op XYZ could be any op like Conv, Matmul etc. Shape transformation ops simply relocate the output of Op XYZ so can be executed before the store of each output element of Op XYZ, similar to Pattern 1.
Examples:
* BERT: Matmul + Reshape + Transpose

**Pattern 5: Conv1x1 + Depthwise Conv**

The fusion happens on each channel since depthwise convolution does reduction per channel.
Examples: MobileNet

**Pattern 6: Conv3x3 + Conv1x1**

[TODO: add fusion mechanism]
Examples: ResNet

