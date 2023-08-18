# Introducing Dropout primitive attribute

## Introduction

In many DNN and GNN models, [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) is used to improve training results. In some cases, this layer can take a significant amount of time. To enhance the performance of training, we want to optimize it and, as a result, fuse it with the previous primitive.

This idea was [proposed](https://github.com/oneapi-src/oneDNN/pull/760) some time ago.
Between post-op and attribute implementation, the primitive attribute was chosen to support complex primitives, like, RNN, where post-op semantics are not well defined.

## Proposal

Additional function to set dropout attibute in C API:

```c
/// Returns the parameters of a drop out attribute.
///
/// @param attr Primitive attributes.
/// @param mask_desc Output memory descriptor of a drop out mask.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_dropout(
        const_dnnl_primitive_attr_t attr, const_dnnl_memory_desc_t *mask_desc);

/// Set up drop-out primitive attribute.
///
/// @param attr Primitive attributes.
/// @param mask_desc Output memory descriptor of a drop out mask.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_dropout(
        dnnl_primitive_attr_t attr, const_dnnl_memory_desc_t mask_desc);
```

for C++ API:
```c++
/// Returns the parameters of a drop out attribute.
///
/// @param mask_desc Output memory descriptor of a drop out mask.
void get_dropout(memory::desc &mask_desc) const {
    const_dnnl_memory_desc_t cdesc;
    error::wrap_c_api(
            dnnl_primitive_attr_get_dropout(get(), &cdesc),
            "could not get parameters of a dropout attribute");
    dnnl_memory_desc_t cloned_md = nullptr;
    error::wrap_c_api(dnnl_memory_desc_clone(&cloned_md, cdesc),
            "could not clone a memory descriptor");
    mask_desc = memory::desc(cloned_md);
    enabled = enabled_u8;
}

/// Set up drop-out.
///
/// @param mask_desc Output memory descriptor of a drop out mask.
void set_dropout(const memory::desc &mask_desc) {
    error::wrap_c_api(dnnl_primitive_attr_set_dropout(get(), mask_desc.get()),
            "could not set dropout primitive attribute");
}
```
and runtime dropout arguments: output mask, which can be used in backward pass,
dropout probability and seed.

```c
/// Arguments for drop out output mask.
#define DNNL_ARG_ATTR_DROPOUT_MASK 16385

/// Arguments for drop out probability param.
#define DNNL_ARG_ATTR_DROPOUT_PROBABILITY 16386

/// Arguments for drop out seed.
#define DNNL_ARG_ATTR_DROPOUT_SEED 16387
```
In most frameworks, the dropout operation is enabled only for the forward training pass, while for the backward pass, the binary multiplication operation can be used. For forward inference, nothing should be done to the tensor.

