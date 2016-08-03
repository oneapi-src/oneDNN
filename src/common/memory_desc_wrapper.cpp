#include <assert.h>
#include "mkl_dnn_types.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "type_helpers.hpp"

namespace mkl_dnn {
namespace impl {

namespace {
using mkl_dnn::impl::array_set;
using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::memory_format;

status_t fill_n(blocking_desc_t& blk, const tensor_desc_t& tensor) {
    const uint32_t ndims = types::ndims(tensor);
    if (ndims != 1)
        return invalid_arguments;
    array_set(blk.padding_dims, 0, ndims);
    array_set(blk.block_dims, 1, ndims);
    array_set(blk.strides[1], 1, ndims);
    blk.strides[0][0] = 1;
    return success;
}

/* TODO: improve me maybe... and put this to utils */
inline void set_default_strides(dims_t strides, const dims_t sizes,
        uint32_t ndims, const uint32_t *perm = NULL) {
    uint32_t id_perm[ndims];
    for (uint32_t i = 0; i < ndims; ++i)
        id_perm[i] = i;
    if (perm == NULL)
        perm = id_perm;

    for (uint32_t d = 0; d < ndims; ++d)
        strides[perm[ndims - 1 - d]] = d == 0 ? 1 :
            strides[perm[ndims - d]] * sizes[perm[ndims - d]];
}

status_t fill_nonblocked(blocking_desc_t& blk, const tensor_desc_t& tensor,
        bool (*constraint)(const tensor_desc_t &), const uint32_t perm[]) {
    const uint32_t ndims = types::ndims(tensor);
    if (constraint && !constraint(tensor)) return invalid_arguments;
    array_set(blk.padding_dims, 0, ndims);
    array_set(blk.block_dims, 1, ndims);
    array_set(blk.strides[1], 1, ndims);
    set_default_strides(blk.strides[0], tensor.dims, ndims, perm);
    return success;
}

status_t fill_nchw(blocking_desc_t& blk, const tensor_desc_t& tensor) {
    const uint32_t perm[4] = {0, 1, 2, 3};
    const auto constraint = [](const tensor_desc_t &t) {
        return t.ndims_batch == 1
            && t.ndims_channels == 1
            && t.ndims_spatial == 2;
    };
    return fill_nonblocked(blk, tensor, constraint, perm);
}

status_t fill_nhwc(blocking_desc_t& blk, const tensor_desc_t& tensor) {
    const uint32_t perm[4] = {0, 2, 3, 1};
    auto constraint = [](const tensor_desc_t &t) {
        return t.ndims_batch == 1
            && t.ndims_channels == 1
            && t.ndims_spatial == 2;
    };
    return fill_nonblocked(blk, tensor, constraint, perm);
}

status_t fill_oihw(blocking_desc_t& blk, const tensor_desc_t& tensor) {
    const uint32_t perm[4] = {0, 1, 2, 3};
    auto constraint = [](const tensor_desc_t &t) {
        return t.ndims_batch == 0
            && t.ndims_channels == 2
            && t.ndims_spatial == 2;
    };
    return fill_nonblocked(blk, tensor, constraint, perm);
}

status_t fill_goihw(blocking_desc_t& blk, const tensor_desc_t& tensor) {
    const uint32_t perm[5] = {0, 1, 2, 3, 4};
    auto constraint = [](const tensor_desc_t &t) {
        return t.ndims_batch == 1
            && t.ndims_channels == 2
            && t.ndims_spatial == 2;
    };
    return fill_nonblocked(blk, tensor, constraint, perm);
}

}

status_t memory_desc_wrapper::compute_blocking(memory_desc_t &memory_desc)
{
    if (types::ndims(memory_desc.tensor_desc) == 0)
        return invalid_arguments;

    const tensor_desc_t &tensor = memory_desc.tensor_desc;
    blocking_desc_t &blk = memory_desc.blocking_desc;

    switch (memory_desc.format) {
    case n: return fill_n(blk, tensor);
    case nchw: return fill_nchw(blk, tensor);
    case nhwc: return fill_nhwc(blk, tensor);
    case oihw: return fill_oihw(blk, tensor);
    case goihw: return fill_goihw(blk, tensor);
    default: break;
    }

    return invalid;
}

}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
