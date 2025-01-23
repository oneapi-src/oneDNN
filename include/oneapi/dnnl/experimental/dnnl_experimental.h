#ifndef ONEAPI_DNNL_EXPERIMENTAL_DNNL_EXPERIMENTAL_H
#define ONEAPI_DNNL_EXPERIMENTAL_DNNL_EXPERIMENTAL_H

#include <oneapi/dnnl/dnnl.h>

#define DNNL_ARG_SRC DNNL_ARG_SRC_0
#define DNNL_ARG_WTS_GATE DNNL_ARG_SRC_1
#define DNNL_ARG_WTS_UP DNNL_ARG_SRC_2
#define DNNL_ARG_WTS_DOWN DNNL_ARG_SRC_3

/// @addtogroup dnnl_api_gmlp
/// @{

/// Creates a primitive descriptor for a matrix multiplication primitive.
///
/// @param primitive_desc Output primitive descriptor.
/// @param engine Engine to use.
/// @param query_desc Query memory descriptor (tensor Q)
/// @param key_desc Key memory descriptor (tensor K)
/// @param value_desc Value memory descriptor (tensor V)
/// @param dst_desc Destination memory descriptor.
/// @param attn_mask_desc Attention mask memory descriptor.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.

dnnl_status_t DNNL_API dnnl_gmlp_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc_iface, dnnl_engine_t engine,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t W_gate_desc,
        const_dnnl_memory_desc_t W_up_desc, const_dnnl_memory_desc_t W_down_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_primitive_attr_t attr,
        const_dnnl_primitive_attr_t gate_attr,
        const_dnnl_primitive_attr_t up_attr,
        const_dnnl_primitive_attr_t down_attr);

/// @} dnnl_api_gmlp

#endif /* ONEAPI_DNNL_EXPERIMENTAL_DNNL_EXPERIMENTAL_H */
