Primitive Attributes: dropout {#dev_guide_attributes_dropout}
=============================================================

## Introduction

In many DNN and GNN models, [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout)
is used to improve training results. In some cases this layer can take a
significant amount of time. To enhance training performance, optimize dropout
by fusing it with the primitive.

## Implementation

In oneDNN, dropout is a special operation akin to a binary post-op that gets
applied to the output values of a primitive right before post-ops. It depends
on a deterministic PRNG (current implementation uses a variation of Philox
algorithm) and transforms the values as follows:

\f[
    \mathrm{mask}[:] = (\mathrm{PRNG}(S, ...) > P) \\
    \mathrm{dst}[:] = \mathrm{mask}[:] \cdot {{\mathrm{dst}[:]} \over {1 - P}}
\f]

where:

* \f$\mathrm{mask}\f$ is the output buffer (always of the same dimensions and
usually of the same layout as \f$\mathrm{dst}\f$, but potentially differing from
it in type that can only be `u8`) whose values may be either 0 if the
corresponding value in \f$\mathrm{dst}\f$ got zeroed (a.k.a. dropped out) or 1
otherwise
* \f$S\f$ is the integer seed for the PRNG algorithm
* \f$P\f$ is the probability for any given value to get dropped out,
\f$0 \leq P \leq 1\f$

## API

- C: @ref dnnl_primitive_attr_get_dropout, @ref dnnl_primitive_attr_set_dropout
- C++: @ref dnnl::primitive_attr::get_dropout, @ref dnnl::primitive_attr::set_dropout

If the dropout operation gets specified in the primitive's attributes, the user
must provide three additional buffers to it on execution:

* `DNNL_ARG_ATTR_DROPOUT_MASK`: through this ID the user has to pass the
\f$\mathrm{mask}\f$ output buffer
* `DNNL_ARG_ATTR_DROPOUT_PROBABILITY`: this is a single-value `f32` input buffer
that holds \f$P\f$
* `DNNL_ARG_ATTR_DROPOUT_SEED`: this is a single-value `s32` input buffer that
holds \f$S\f$
