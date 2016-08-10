#ifndef MEMORY_DESC_WRAPPER_HPP
#define MEMORY_DESC_WRAPPER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "type_helpers.hpp"

namespace mkl_dnn {
namespace impl {

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

    memory_desc_wrapper(const memory_primitive_desc_t &mpd)
        : memory_desc_wrapper(mpd.memory_desc) {}

    /* implementing attrubutes */
    const dims_t &dims() const { return _md.tensor_desc.dims; }
    const tensor_desc_t &tensor() const { return _md.tensor_desc; }
    precision_t precision() const { return _md.precision; }
    memory_format_t format() const { return _md.format; }
    const blocking_desc_t &blocking_desc() const {
        return _md.layout_desc.blocking;
    }
    inline uint32_t ndims() const { return _ndims; }

    /** returns the number of elements including padding if \param with_padding
     * is true, and the number of data elements otherwise */
    size_t nelems(bool with_padding = false) const {
        if (is_zero()) return 0;
        return array_product(with_padding
                ? blocking_desc().padding_dims : dims(), ndims());
    }

    /** returns true if memory descriptor is zero */
    bool is_zero() const { return ndims() == 0; }

    /** returns the size required to store described memory */
    size_t size() const {
        if (is_zero() || format() == any) return 0;
        assert(one_of(format(), x, nc, nchw, nhwc, nChw8c, oi, oihw, OIhw8i8o,
                    goihw, gOIhw8i8o, blocked));

        const auto &block_dims = blocking_desc().block_dims;
        const auto &strides = blocking_desc().strides;
        const auto &padding_dims = blocking_desc().padding_dims;

        size_t max_size = 0;
        for (uint32_t d = 0; d < ndims(); ++d) {
            auto block = block_dims[d];
            max_size = nstl::max(max_size,
                    size_t(padding_dims[d]/block)*strides[0][d]);
            if (block > 1)
                max_size = nstl::max(max_size, size_t(block*strides[1][d]));
        }
        return max_size * types::precision_size(precision());
    }

    /* offset section */

    /** returns physical offset by logical one. logical offset is represented by
     * an array \param pos. if \param is_pos_padded is true \param pos
     * represents the position in already padded area */
    inline size_t off_v(const dims_t pos, bool is_pos_padded = false) const {
        assert(format() != any);
        const blocking_desc_t &blk = blocking_desc();
        const dims_t &optd = blk.offset_padding_to_data;

        size_t phys_offset = blk.offset_padding;
        for (uint32_t d = 0; d < ndims(); ++d) {
            const uint32_t block = blk.block_dims[d];

            const uint32_t p = pos[d] + (is_pos_padded ? 0 : optd[d]);
            const uint32_t pos_within_block = p % block;
            const uint32_t pos_block = p / block;

            phys_offset += pos_block * blk.strides[0][d];
            phys_offset += pos_within_block * blk.strides[1][d];
        }
        return phys_offset;
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a scalar \param l_offset. if \param is_pos_padded is true, \param
     * l_offset represents logical offset in already padded area */
    inline size_t off_l(size_t l_offset, bool is_pos_padded = false) const {
        const dims_t &padding_dims = blocking_desc().padding_dims;
        dims_t pos;
        for (uint32_t rd = 0; rd < ndims(); ++rd) {
            const uint32_t d = ndims() - 1 - rd;
            const uint32_t cur_dim = is_pos_padded ? padding_dims[d] : dims()[d];
            pos[d] = l_offset % cur_dim;
            l_offset /= cur_dim;
        }
        return off_v(pos, is_pos_padded);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indeces (\param xn, ..., \param x1, \param x0) */
    template<typename... Args> inline size_t off(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, false);
    }

    /** returns physical offset by logical one. logical offset is represented by
     * a tuple of indeces (\param xn, ..., \param x1, \param x0) in already
     * padded area */
    template<typename... Args> inline size_t off_padding(Args... args) const {
        assert(sizeof...(args) == ndims());
        dims_t pos = { args... };
        return off_v(pos, true);
    }

    /* static functions section */
    /* TODO: replace with non-static, once _md becomes non-const ref */

    static status_t compute_blocking(memory_desc_t &memory_desc);

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

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

