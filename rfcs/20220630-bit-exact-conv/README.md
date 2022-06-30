# RFC: Optional bit exact Convolution

## Introduction

Several cpu convolution primitives in the library may compute a non accurate
result in case vnni is unavailable.
An example case is the `jit_avx512_core_x8s8s32x_convolution_fwd_t` convolution
primitive running with avx512_core ISA.

Non accurate results may occur due to the following behaviors:

1. Weights and inputs multiplication uses the VPMADDUBSW instruction. This
   instruction multiplies two pairs of u8/s8 values and then accumulates the
   two s16 products into one s16 with potential saturation.

2. Weights are scaled with a factor of 0.5f. This scaling is then compensated
   for with a factor of 2. However, for odd weights, this compensation gives
   non accurate results, as the weights are integers.

These behaviors are intentional, giving significantly better performance than
the equivalent accurate computation.
However, depending on the user, compromising accuracy may not be an option.
Adding an optional bit exact implemetation will allow the user to prioritize
correctness over performance.

## Proposal

The proposal is to let the user choose a slower implementation with guranteed
accuracy.

The link below demostrates the changes in the primitive's implemetation,
allowing accurate results.
It does not demostrate any API changes needed to expose this option to the
user, instead it uses a compile time define. One way to expose this option
may be via primitive attributes API.

https://github.com/oneapi-src/oneDNN/compare/master...maayaneh:oneDNN:master

In the link, the changes are applied to the following primitives:
* `jit_avx512_core_x8s8s32x_convolution_fwd_t`
* `jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t`

In case the user chooses to use the accurate implementation, The main changes
are:

1. Avoid weight scaling.
2. Use a different set of instructions to multiply and accumulate weights and
   inputs with no potential saturation.
3. Avoid using 128 compensation for signed input - this is not essential for
   correctness, it is simply not necessary since the alternative multiplication
   instruction does not require the input to be unsigned.
