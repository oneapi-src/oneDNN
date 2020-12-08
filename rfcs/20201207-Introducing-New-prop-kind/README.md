# Introducing new prop_kind, which inhibits Compute Library use (RFC)

## Introduction

At the moment there is no hint mechanism to prefer one convolution implementation
over another in [cpu\_convolution\_list](https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/cpu_convolution_list.cpp#L92-L111), implementations are simply checked in the order
they are listed and the first one satisfying all the requirements is chosen. However,
the Compute Library based primitives do not currently support [`BWD_D` deconvolution](https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/ref_deconvolution.hpp#L67-L73). This type of
deconvolution is called as part of the CI benchdnn tests. This has not been an issue
previously since these tests all contain "g1" in the problem descriptor (degenerate
case of the grouped convolutions with `ngroups=1`), which is also not currently supported
with the ACL backend. Providing ACL support for convolutions with "g1" necessitates
a resolution to the `BWD_D` deconvolution problem described above.

One solution might be to remove the support of `forward_training` from
[acl\_convolution\_utils.cpp](https://github.com/oneapi-src/oneDNN/blob/master/src/cpu/aarch64/acl_convolution_utils.cpp#L55-L56). On the other hand, `forward_training` is set as a `prop_kind`
in common ML frameworks (PyTorch, TensorFlow) and benchmarks (MLPerf) even in the inference regime,
thus its disabling will noticeably limit functionality. Another variant, which is presented
in this RFC, is to introduce new `prop_kind_t`, namely [`forward_training_no_acl`](https://github.com/oneapi-src/oneDNN/compare/master...alenik01:Introducing-New-prop-kind-RFC), which inhibits ACL-based
implementation to be chosen for `BWD_D` deconvolution.

## Proposal

The source code may be viewed [here](https://github.com/oneapi-src/oneDNN/compare/master...alenik01:Introducing-New-prop-kind-RFC). The key changes required are listed below:

- Adding new `prop_kind_t`, `dnnl_forward_training_no_acl`, to `dnnl_types.h` and in
the relevant files;
- Disabling of Compute Library based convolution in [src/cpu/ref_deconvolution.hpp](https://github.com/alenik01/oneDNN/blob/Introducing-New-prop-kind-RFC/src/cpu/ref_deconvolution.hpp#L69) with the help of `forward_training_no_acl`;
- Adding Compute Library support for the grouped convolutions with `ngroups==1`.

These changes cannot be split into different commits for the sake of passing all
the CI tests.

### New `prop_kind_t`:

The new prop\_kind is introduced only in one place to inhibit the call to ACL for a
deconvolution with `BWD_D`. It has the same meaning as `forward_training`, the only
difference is that `CPU_INSTANCE_AARCH64_ACL` will return `status::unimplemented`
and be skipped for this `prop_kind_t` value.

## Limited impact

The scope of this RFC is limited such that:

- There are no changes to the API in terms of `forward_training` and `forward_inference`
functionality;
- There are no effects on non-AArch64 builds.

Proposed changes are aimed to fix the specific issue with Compute Library support
for oneDNN and `forward_training_no_acl` may be remove in future, once the issue with
`BWD_D` deconvolution will be resolved, but these changes introduce a new `prop_kind_t`
therefore it seems more appropiate to raise an RFC, rather than a PR.

Any comments on how to solve the problem in a more elegant way would be appreciated.