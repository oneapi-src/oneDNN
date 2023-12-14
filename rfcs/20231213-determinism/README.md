# Proposal for deterministic mode in oneDNN

## Problem statement and proposal

Multiple applications require deterministic output to achieve some
level of certification (e.g. medical, avionics applications, ..).  As
such, PyTorch and Tensorflow are exposing deterministic mode to those
users.  Furthermore, they start relying on this deterministic mode for
their own validation processes (e.g. PyTorch torch.compile
validation).

Here, determistic execution refers to returning the
bitwise same result when a primitive is run multiple times on the same
system (so fixed hardware configuration), with the same environment
(fixed software environment).

The proposal is to add a new boolean attribute named `deterministic`,
which would prevent the dispatching of non-deterministic
implementations.

## Impact on implementations

There are two main reasons for an implementation to be non-deterministic:
- parallel reduction based on atomics. This mostly happen in GPU
  implementations as this can sometimes significantly improve
  performance. Here the recommendation is either to avoid parallel
  reduction, or to add a synchronization and reduce partial sums with
  a fixed order. Note that this reduction order requirement applies to
  both floating-point (because of roundings) and integer primitives
  (because of saturation).
- the use of a state. This is prominent in random generation
  algorithm. There currently is no random generation happening in
  oneDNN. However, if it were to happen, the library would have to
  rely on stateless algorithms (aka counter-based RNG), like Philox or
  Threefry.

## Impact on API

On API side, we will have to introduce a new attribute.

We would need to add the following symbols to the C API
```cpp
/// Returns the deterministic primitive attribute value.
///
/// @param attr Primitive attributes.
/// @param value Output deterministic attribute value
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_deterministic(
        const_dnnl_primitive_attr_t attr, int *value);

/// Sets the deterministic primitive attribute value.
///
/// @param attr Primitive attributes.
/// @param value Boolean value to set deterministic attribute
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_deterministic(
        dnnl_primitive_attr_t attr, int value);
```

And the following symbols to the C++ API
```cpp
    /// Returns the deterministic attribute value
    bool get_deterministic() const {
        int result;
        error::wrap_c_api(dnnl_primitive_attr_get_deterministic(get(), &result),
                "could not get deterministic primitive attribute");
        return result;
    }

    /// Sets deterministic attribute value
    ///
    /// @param value Specified deterministic mode.
    void set_deterministic(bool value) {
        error::wrap_c_api(dnnl_primitive_attr_set_deterministic(
                                  get(), value),
                "could not set deterministic primitive attribute");
    }
```

To align with users expectations, the default value for this attribute
will be false (current behavior, and default for both Tensorflow and Pytorch).

We are not proposing a global settting for deterministic behavior as
those can typically cause unwanted global synchronizations, and do not
seem to be needed. This can be revised if there is a usage scenario
for global setting.

Internally, we will expose a new `attr_t::skip_mask` for
implementation that support non-default behavior
(deterministic=true). Because most implementation support determinism
already, this will require a 1 line change in almost all
implementations.

## Impact on validation

To check for determinism, benchdnn will need to be adapted to
- run the same operation twice with same inputs, and check for bitwise
  similar results.
- fill tensors in a way that forces differences in output when the
  order of operations changes between runs.

That second requirement requires benchdnn to introduce a new mode, as
the current modes do the exact opposite: they fill tensors to avoid
intermediate roundings, making the output independant of the order of
operations. 

