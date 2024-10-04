# Support for Source and Weights Reduction in MatMul

## Motivation

For historical reasons oneDNN provides two primitives that significantly overlap in
functionality. It's the Matrix Multiplication (MatMul) and Inner Product (IP)
primitives. Currently, both primitives may have different implementations that have to
be maintained, which can be costly. However, MatMul is, in a sense, a building block that
can be used to implement the Inner Product primitive, and the idea is to do that inside the
library.

Unfortunately, the MatMul primitive lacks some functionality and therefore cannot be used as
is to achieve the goal. Most of the lack of functionality is due to differences in semantics
between the MatMul and Inner Product primitives, so those differences should be addressed
first.

The main differences between the primitives are:
* There might be a gap in supported formats (not related to semantics).
* Inner Product has a notion of forward and backward propagations while MatMul does not.
* Inner Product supports calculating bias gradients while MatMul does not.

The first difference can be addressed via adding support for more formats to MatMul,
which is fairly straightforward.

The second difference doesn't actually require addressing because MatMul can be
configured to calculate the forward and backward propagations.

The third difference can be addressed via using the reduction primitive to calculate
the bias gradients separately; however, this approach can be suboptimal. To get the best
performance MatMul has to support calculating the bias gradients natively.

This proposal outlines a solution for the third difference.

## Proposal

The proposal is to add support for calculating the bias gradients to the MatMul primitive.

### External vs Internal API

The feature for calculating the bias gradients can either be made available to the user or
kept internal.

#### External (recommended)
Pros:
* There are users (e.g. TensorFlow, XLA) that do not want to use the inner product
primitive, instead they use MatMul for the forward pass and their own implementation
of a MatMul + Reduction kind of operation for the backward pass. Those users can
benefit from making the feature external.
* If MatMul supports all the formats that Inner Product does, then having Inner Product in
the library would no longer be justified. Therefore, it can be removed, further decreasing
maintenance cost.

Cons:
* Making the feature external will not automatically lead users to adopt it over their
own implementations or the Inner Product primitive. Thus, the inner product may have to
remain for an indefinite period.

#### Internal
Pros:
* As with any internal API one of the pros is to be able to change it at any time
without breaking the API.
* There is some margin for error.

Cons:
* Limited availability prevents the feature from reaching its full potential.

#### Summary

The preferred option is to make the feature external, as this will pave the path to
eliminating the inefficiencies present in the API and the library.

### API

The proposal is to extend the current MatMul API to instruct it to produce an additional output
that would contain the bias gradients.

The current C++ API for MatMul provides two constructors: the first takes a bias memory
descriptor, and the second does not. Adding a third constructor that would take a memory
descriptor for the bias gradients is problematic because:
* The third constructor should not take a bias memory descriptor as it doesn't make sense to
create a MatMul that applies bias and calculates the bias gradients simultaneously.
* Taking a memory descriptor for the bias gradients instead of the one for bias creates an
overload issue due to the second and third constructors being identical.

To differentiate the memory descriptors in the second and third constructors, a new concept
called "MatMul extension" will be introduced. The extension will be represented using
enumerations.

C API:
```c
/// Extensions for matmul.
typedef enum {
    /// Undefined matmul extension.
    dnnl_matmul_extension_undef = 0,
    /// Bias extension.
    dnnl_matmul_extension_bias,
    /// Reduce src extension. The extension memory descriptor must have
    /// the same number of dimensions as src tensor. Reduction dimensions
    /// are of size 1.
    dnnl_matmul_extension_reduce_src,
    /// Reduce weights extension. The extension memory descriptor must have
    /// the same number of dimensions as weights tensor. Reduction dimensions
    /// are of size 1.
    dnnl_matmul_extension_reduce_weights,
} dnnl_matmul_extension_t;
```
C++ API:
```cpp
struct matmul : public primitive {
    enum class extension {
        /// Undefined matmul extension.
        undef = dnnl_matmul_extension_undef,
        /// Bias extension.
        bias = dnnl_matmul_extension_bias,
        /// Reduce src extension. The extension memory descriptor must have
        /// the same number of dimensions as src tensor. Reduction dimensions
        /// are of size 1.
        reduce_src = dnnl_matmul_extension_reduce_src,
        /// Reduce weights extension. The extension memory descriptor must have
        /// the same number of dimensions as weights tensor. Reduction dimensions
        /// are of size 1.
        reduce_weights = dnnl_matmul_extension_reduce_weights,
    };
    // ...
};
```

The API for creating a primitive descriptor will take a memory descriptor for the extension
and the extension kind.

C API:
```c
/// Creates a primitive descriptor for a matrix multiplication primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param src_desc Source memory descriptor (matrix A)
/// @param weights_desc Weights memory descriptor (matrix B)
/// @param dst_desc Destination memory descriptor (matrix C).
/// @param ext_desc Extension memory descriptor. Each extension defines requirements
///    to @p ext_desc. If @p extension is #dnnl_matmul_extension_undef then @p ext_desc is
///    ignored.
/// @param extension Extension kind.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_matmul_primitive_desc_create_v2(
        dnnl_primitive_desc_t *primitive_desc,
        dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t weights_desc,
        const_dnnl_memory_desc_t dst_desc,
        const_dnnl_memory_desc_t ext_desc,
        dnnl_matmul_extension_t extension,
        const_dnnl_primitive_attr_t attr);
```

C++ API:
```cpp
struct matmul : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        /// @param aengine Engine to use.
        /// @param src_desc Memory descriptor for source (matrix A).
        /// @param weights_desc Memory descriptor for weights (matrix B).
        /// @param dst_desc Memory descriptor for destination (matrix C).
        /// @param ext_desc Memory descriptor for extension. Each extension
        ///     defines requirements to @p ext_desc. If @p extension is
        ///     #dnnl::matmul::extension::undef then @p ext_desc is ignored.
        /// @param attr Primitive attributes to use. Attributes are optional
        ///     and default to empty attributes.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case an
        ///     empty object will be produced. This flag is optional and
        ///     defaults to false.
        primitive_desc(const engine &aengine, const memory::desc &src_desc,
                const memory::desc &weights_desc, const memory::desc &dst_desc,
                const memory::desc &ext_desc, matmul::extension extension,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false);
        // ...
    };
    // ...
```

When `matmul::extension::reduce_src` or `matmul::extension::reduce_weights` is specified
MatMul will produce an additional output tensor. The tensor should be passed to the primitive
using the following argument tag:
```c
/// Reduce tensor argument.
#define DNNL_ARG_REDUCE 42
```

### Additional Considerations

#### Attribute/Pre-op

At first glance it might look like the feature should be enabled using the attributes or
some sort of a pre-op mechanism. The recommendation to associate the feature with MatMul is
based on the following considerations:
* The current attributes and post-ops mechanisms instruct the library on how to process the
destination tensor. In the case of a potential pre-ops mechanism, it would indicate how to
process the input tensors. Here, "processing" refers to modifying the tensors before or after
the compute operation, and these mechanisms do not generate additional outputs.
* The feature is specific to MatMul, so there is little value in generalizing it.

#### Extension Enum vs Extension Flags

The rationale for representing the extensions with enumerations is that they are not intended to
be combined.

#### Internal GEMM vs External MatMul

The GPU runtime has an internal GEMM primitive that is used to implement MatMul, Inner
Product and RNN. The question is whether it makes sense to introduce the internal GEMM primitive
to the CPU runtime as well. For the following reasons, adding the internal GEMM primitive for
CPU runtime appears to be redundant:
* The primary reason for introducing the GEMM primitive was the absence of the MatMul primitive
that the GPU runtime required at that time.
* The MatMul primitive offers the necessary functionality to implement the Inner Product and
RNN primitives.
* The operation descriptors are now internal, enabling the addition of functionality that is not exposed to users, which means the GEMM primitive does not provide any additional flexibility.
