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
Adding an optional bit exact implementation will allow the user to prioritize
correctness over performance.

## Proposal

The proposal is to let the user choose a slower implementation with guaranteed
accuracy.

The link below demonstrates the changes in the primitive's implementation,
allowing accurate results.
It does not demonstrate any API changes needed to expose this option to the
user, instead it uses a compile time define.

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

### Proposed API

#### Option 1: Add a build time option

Let the user set the same behavior for all convolutions with a build option.

Option name: `ONEDNN_JIT_AVX512_FORCE_BE`
Supported values: ON, OFF (default)

Pros:
* Simple implementation.
* Simple usage - after building nothing is required from the user.

Cons:
* No granularity - all primitives are configured the same.

#### Option 2: Add a primitive attribute

Extend primitive attributes with the following members:

~~~c++
struct dnnl_primitive_attr : public dnnl::impl::c_compatible {
   ...
   void set_force_be(bool force_be) { is_force_be_ = force_be; };
   ...
   bool is_force_be_;
}
~~~

Usage:
~~~c++
dnnl::primitive_attr conv_attr;

// force_be is set per primitive
conv_attr.set_force_be(true);

// create primitive descriptor with custom attributes
auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);
~~~

Pros:
* Granularity - allow the user to choose a different behavior per primitive.

Cons:
* Usage is not as simple - the attribute needs to be reset per primitive.


I recommend using option 1 for simplicity. While option 1 offers no
granularity, I believe in most cases the same user will require the same
behavior for all primitives, thus rendering granularity unnecessary.