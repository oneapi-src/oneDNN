#ifndef TYPE_HELPERS_HPP
#define TYPE_HELPERS_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

template <mkl_dnn::impl::precision_t> struct precision2type {};
template <>
struct precision2type<mkl_dnn::impl::precision::f32> { typedef float type; };

namespace mkl_dnn {
namespace impl {
// XXX: is this the only namespace we have?
namespace types {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;

inline uint32_t ndims(const tensor_desc_t &tensor) {
    uint32_t ndims = tensor.ndims_batch + tensor.ndims_channels +
        tensor.ndims_spatial;
    assert(ndims <= TENSOR_MAX_DIMS);
    return ndims;
}

inline bool operator==(const tensor_desc_t &lhs, const tensor_desc_t &rhs) {
    return lhs.ndims_batch == rhs.ndims_batch
        && lhs.ndims_channels == rhs.ndims_channels
        && lhs.ndims_spatial == rhs.ndims_spatial
        && mkl_dnn::impl::array_cmp(lhs.dims, rhs.dims, ndims(lhs));
}

inline bool operator!=(const tensor_desc_t &lhs, const tensor_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline status_t tensor_is_ok(const tensor_desc_t &tensor) {
    return ndims(tensor) <= TENSOR_MAX_DIMS ? success : invalid;
}

inline uint32_t ndims(const memory_desc_t &memory_desc) {
    return ndims(memory_desc.tensor_desc);
}

inline size_t get_size(const memory_desc_t &memory_desc) {
    if (memory_desc.format == any) return 0;

    assert(one_of(memory_desc.format, n, nchw, nhwc, oihw, goihw, blocked));

    size_t max_size = 0;
    auto dims = memory_desc.tensor_desc.dims;
    auto blocks = memory_desc.blocking_desc.block_dims;
    auto strides = memory_desc.blocking_desc.strides;

    /* FIXME: write the correct formula */
    for (uint32_t d = 0; d < ndims(memory_desc); ++d) {
        using mkl_dnn::impl::nstl::max;
        max_size = max<size_t>(max_size, (dims[d]/blocks[d])*strides[0][d]);
        if (blocks[d] != 1)
            max_size = max<size_t>(max_size, blocks[d]*strides[1][d]);
    }

    return max_size*sizeof(float);
}

inline size_t get_size(const memory_primitive_desc_t &memory_prim_desc) {
    return get_size(memory_prim_desc.memory_desc);
}

status_t compute_blocking(memory_desc_t &memory_desc);

inline bool blocking_desc_is_equal(const blocking_desc_t &lhs,
        const blocking_desc_t &rhs, uint32_t ndims = TENSOR_MAX_DIMS) {
    using mkl_dnn::impl::array_cmp;
    return lhs.offset_padding == rhs.offset_padding
        && lhs.offset_padding_to_data == rhs.offset_padding_to_data
        && array_cmp(lhs.padding_dims, rhs.padding_dims, ndims)
        && array_cmp(lhs.block_dims, rhs.block_dims, ndims)
        && array_cmp(lhs.strides[0], rhs.strides[0], ndims)
        && array_cmp(lhs.strides[1], rhs.strides[1], ndims);
}

inline bool operator==(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    if (lhs.tensor_desc != rhs.tensor_desc || lhs.format != rhs.format)
        return false;
    if (lhs.format == blocked)
        return blocking_desc_is_equal(lhs.blocking_desc, rhs.blocking_desc,
            ndims(lhs.tensor_desc));
    return true;
}

inline bool operator!=(const memory_desc_t &lhs, const memory_desc_t &rhs) {
    return !operator==(lhs, rhs);
}

inline bool operator==(const memory_primitive_desc_t &lhs,
        const memory_primitive_desc_t &rhs) {
    return lhs.base.engine == rhs.base.engine // XXX: is it true?
        && lhs.memory_desc == rhs.memory_desc;
}

inline status_t convolution_desc_is_ok(
        const convolution_desc_t &convolution_desc) {
    // XXX: fill-in
    return success;
}

inline status_t pooling_desc_is_ok(
    const pooling_desc_t &pooling_desc) {
    // XXX: fill-in
    return success;
}

struct memory_desc_wrapper: public c_compatible {
    const memory_desc_t &_md;
    memory_desc_wrapper(const memory_desc_t &md): _md(md) {
        assert(_md.format != any);
    }

    const tensor_desc_t &tensor() const { return _md.tensor_desc; }
    uint32_t ndims() const { return types::ndims(_md.tensor_desc); }
    const dims_t &dims() const { return _md.tensor_desc.dims; }
    const memory_format_t format() const { return _md.format; }

    size_t off_v(const dims_t pos) const {
        return 0;
    }

    inline size_t off(size_t l_offset) const {
        if (one_of(_md.format, n, nchw, oihw, goihw))
            return l_offset; // tentative
        assert(false);
        // format specific
        return l_offset;
    }

    template<typename T, typename... Args>
    inline size_t off(T xn, Args... args) const {
        return off(logical_offset(xn, args...));
    }

    template<typename T>
    inline size_t logical_offset(T x0) const { return (size_t)x0; }

    template<typename T, typename... Args>
    inline size_t logical_offset(T xn, Args... args) const {
        const size_t n_args = sizeof...(Args);
        return ((size_t)xn)*array_product<n_args>(&dims()[ndims() - n_args])
            + logical_offset(args...);
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
