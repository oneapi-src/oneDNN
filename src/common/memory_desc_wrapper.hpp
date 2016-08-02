#ifndef MEMORY_DESC_WRAPPER_HPP
#define MEMORY_DESC_WRAPPER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace mkl_dnn {
namespace impl {
namespace types {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::precision;
using namespace mkl_dnn::impl::memory_format;

/** thin wrapper class over \struct memory_desc_t which allows easy
 * manipulations with underlying C structure, which is taken by refernce */
struct memory_desc_wrapper: public c_compatible {
    const memory_desc_t &_md;
    const uint32_t _ndims;

    /** constructor which takes a reference to a constant underlying C memory
     * descriptor \param md */
    memory_desc_wrapper(const memory_desc_t &md)
        : _md(md)
        , _ndims(_md.tensor_desc.ndims_batch
                + _md.tensor_desc.ndims_channels
                + _md.tensor_desc.ndims_spatial)
    {}

    /* implementing attrubutes */
    const dims_t &dims() const { return _md.tensor_desc.dims; }
    const tensor_desc_t &tensor() const { return _md.tensor_desc; }
    precision_t precision() const { return _md.precision; }
    memory_format_t format() const { return _md.format; }
    const blocking_desc_t &blocking_desc() const { return _md.blocking_desc; }
    inline uint32_t ndims() const { return _ndims; }

    /** returns physical offset by logical one
     * logical offset is represented by an array \param pos */
    inline size_t off_v(const dims_t pos) const {
        assert(format() != any);
        const blocking_desc_t &blk = blocking_desc();

        size_t phys_offset = blk.offset_padding + blk.offset_padding_to_data;
        for (uint32_t d = 0; d < ndims(); ++d) {
            const uint32_t block = blk.block_dims[d];
            const uint32_t pos_within_block = pos[d] % block;
            const uint32_t pos_block = pos[d] / block;
            phys_offset += pos_block * blk.strides[0][d];
            phys_offset += pos_within_block * blk.strides[1][d];
        }
        return phys_offset;
    }

    /** returns physical offset by logical one
     * logical offset is represented by a scalar \param l_offset */
    inline size_t off_l(size_t l_offset) const {
        dims_t pos;
        for (uint32_t d = 0; d < ndims(); ++d) {
            const uint32_t cur_dim = dims()[ndims() - 1 - d];
            pos[ndims() - 1 - d] = l_offset % cur_dim;
            l_offset /= cur_dim;
        }
        return off_v(pos);
    }

    /** returns physical offset by logical one
     * logical offset is represented by a tuple of indeces (\param xn, ...,
     * \param x1, \param x0) */
    template<typename... Args>
    inline size_t off(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos);
    }

private:
    /* TODO: put logical_offset in utils */
    template<typename T>
    inline size_t logical_offset(T x0) const { return (size_t)x0; }

    template<typename T, typename... Args>
    inline size_t logical_offset(T xn, Args... args) const {
        const size_t n_args = sizeof...(args);
        return ((size_t)xn)*array_product<n_args>(&dims()[ndims() - n_args])
            + logical_offset(args...);
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

