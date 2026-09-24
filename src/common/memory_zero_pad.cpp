/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <atomic>
#include <cassert>

#include "dnnl_thread.hpp"
#include "dnnl_traits.hpp"
#include "stream.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "memory.hpp"
#include "primitive_exec_types.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::status;

enum blk_kind_t { a, b, c, ab, ba, bc, cb };

namespace {
// A padding element only ever needs to be set to zero, so its data type is
// irrelevant beyond the number of bytes it occupies. Emitting a single
// width-sized store avoids both a runtime `memset` call and a per-byte loop.
ALWAYS_INLINE void zero_element(uint8_t *p, size_t element_size) {
    switch (element_size) {
        case 1: *reinterpret_cast<uint8_t *>(p) = 0; break;
        case 2: *reinterpret_cast<uint16_t *>(p) = 0; break;
        case 4: *reinterpret_cast<uint32_t *>(p) = 0; break;
        case 8: *reinterpret_cast<uint64_t *>(p) = 0; break;
        default: assert(!"unsupported element size"); break;
    }
}
} // namespace

template <blk_kind_t blk_kind>
void typed_zero_pad_blk(const memory_desc_wrapper &m_d, void *data_handle,
        size_t element_size, dim_t blksize) {
    auto *data = reinterpret_cast<uint8_t *>(data_handle);
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.padded_dims();
    const auto &blk = m_d.blocking_desc();
    auto dim_is_blocked = [&](int dim) {
        for (int i = 0; i < blk.inner_nblks; i++)
            if (blk.inner_idxs[i] == dim) return true;
        return false;
    };
    bool A_blocked = dim_is_blocked(0), B_blocked = dim_is_blocked(1),
         C_blocked = dim_is_blocked(2);

    assert(blk.inner_nblks < 4);
    assert((A_blocked || B_blocked || C_blocked) || (A_blocked && B_blocked)
            || (C_blocked && B_blocked));

    const dim_t a_tail_s = A_blocked ? dims[0] % blksize : 0;
    const dim_t b_tail_s = B_blocked ? dims[1] % blksize : 0;
    const dim_t c_tail_s = C_blocked ? dims[2] % blksize : 0;
    assert(a_tail_s || b_tail_s || c_tail_s);

    const int ndims = m_d.ndims();
    assert(1 <= ndims && ndims <= 6);
    const dim_t A = A_blocked ? pdims[0] / blksize : dims[0];
    const dim_t B = ndims <= 1 ? 1 : B_blocked ? pdims[1] / blksize : dims[1];
    const dim_t C = ndims <= 2 ? 1 : C_blocked ? pdims[2] / blksize : dims[2];
    const dim_t D = ndims <= 3 ? 1 : dims[3];
    const dim_t E = ndims <= 4 ? 1 : dims[4];
    const dim_t F = ndims <= 5 ? 1 : dims[5];
    const dim_t inner_blk = blk.inner_nblks == 3 ? blk.inner_blks[2] : 1;

    auto zeroize_tail = [=](uint8_t *d, const dim_t tail_s) {
        for (dim_t bb = tail_s; bb < blksize; ++bb)
            zero_element(d + (size_t)bb * element_size, element_size);
    };
    auto zeroize_tail_inner = [=](uint8_t *d, const dim_t tail_s) {
        for_(dim_t b1 = 0; b1 < blksize; ++b1)
        for (dim_t b2 = tail_s; b2 < blksize; ++b2) {
            const dim_t idx = (b1 / inner_blk) * blksize * inner_blk
                    + inner_blk * b2 + b1 % inner_blk;
            zero_element(d + (size_t)idx * element_size, element_size);
        }
    };
    auto zeroize_tail_outer = [=](uint8_t *d, const dim_t tail_s) {
        for_(dim_t b1 = tail_s; b1 < blksize; ++b1)
        for (dim_t b2 = 0; b2 < blksize; ++b2) {
            const dim_t idx = (b1 / inner_blk) * blksize * inner_blk
                    + inner_blk * b2 + b1 % inner_blk;
            zero_element(d + (size_t)idx * element_size, element_size);
        }
    };

    if (c_tail_s) {
        parallel_nd(A, B, D, E, F,
                [=](dim_t a, dim_t b, dim_t d, dim_t e, dim_t f) {
            auto *x = &data[m_d.blk_off(a, b, C - 1, d, e, f) * element_size];
            if (blk_kind == c)
                zeroize_tail(x, c_tail_s);
            else if (blk_kind == bc)
                zeroize_tail_inner(x, c_tail_s);
            else if (blk_kind == cb)
                zeroize_tail_outer(x, c_tail_s);
        });
    }

    if (b_tail_s) {
        parallel_nd(A, C, D, E, F,
                [=](dim_t a, dim_t c, dim_t d, dim_t e, dim_t f) {
            auto *x = &data[m_d.blk_off(a, B - 1, c, d, e, f) * element_size];
            if (blk_kind == b)
                zeroize_tail(x, b_tail_s);
            else if (blk_kind == ab || blk_kind == cb)
                zeroize_tail_inner(x, b_tail_s);
            else if (blk_kind == ba || blk_kind == bc)
                zeroize_tail_outer(x, b_tail_s);
        });
    }

    if (a_tail_s) {
        parallel_nd(B, C, D, E, F,
                [=](dim_t b, dim_t c, dim_t d, dim_t e, dim_t f) {
            auto *x = &data[m_d.blk_off(A - 1, b, c, d, e, f) * element_size];
            if (blk_kind == a)
                zeroize_tail(x, a_tail_s);
            else if (blk_kind == ba)
                zeroize_tail_inner(x, a_tail_s);
            else if (blk_kind == ab)
                zeroize_tail_outer(x, a_tail_s);
        });
    }
}

/*
 * all
 */
void typed_zero_pad_generic_blocked(const memory_desc_wrapper &m_d,
        void *data_handle, size_t element_size) {
    auto *data = reinterpret_cast<uint8_t *>(data_handle);
    const int ndims = m_d.ndims();
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.padded_dims();

    const ptrdiff_t nelems = (ptrdiff_t)m_d.nelems(true);

    /* [D_0] .. [D_k][D_k+1] .. [D_ndim - 1]
     *            |  \                     /
     *            |   ---------------------
     *           has        contiguous
     *         padding
     *
     * step     <-- D_k+1 * ... * D_ndims-1
     * step_dim <-- k
     */

    ptrdiff_t step = 1;
    int step_dim = ndims - 1;
    for (; step_dim >= 0; --step_dim) {
        if (dims[step_dim] != pdims[step_dim]) break;
        step *= dims[step_dim];
    }

    assert(step_dim >= 0 && "no zero padding is required");
    if (step_dim < 0) return;

    parallel_nd(nelems / step, [=](ptrdiff_t e1) {
        bool need_zero = false;

        ptrdiff_t idx = e1;
        for (int d = step_dim; d >= 0; --d) {
            if (idx % pdims[d] >= dims[d]) {
                need_zero = true;
                break;
            }
            idx /= pdims[d];
        }

        if (need_zero) {
            for (ptrdiff_t e0 = 0; e0 < step; ++e0)
                zero_element(
                        &data[m_d.off_l(e1 * step + e0, true) * element_size],
                        element_size);
        }
    });
}

// Sub-byte data types pack several elements into a shared byte or bytes, thus
// a byte-based zeroing (as in the routines above) would write past the buffer
// and corrupt neighbor elements. The physical offset of each element is
// resolved individually (via `off_l`) because for formats with multiple inner
// blocks the innermost logical run is not physically contiguous. A byte may be
// shared between a padded and a non-padded element, or between two parallel
// threads, so every byte an element occupies is updated with an atomic
// bit-clear that keeps the byte's other elements intact regardless of the
// update order.
void typed_zero_pad_sub_byte(
        const memory_desc_wrapper &m_d, void *data_handle, int bits_per_elem) {
    auto *data = reinterpret_cast<uint8_t *>(data_handle);
    const int ndims = m_d.ndims();
    const auto &dims = m_d.dims();
    const auto &pdims = m_d.padded_dims();
    const ptrdiff_t nelems = (ptrdiff_t)m_d.nelems(true);

    ptrdiff_t step = 1;
    int step_dim = ndims - 1;
    for (; step_dim >= 0; --step_dim) {
        if (dims[step_dim] != pdims[step_dim]) break;
        step *= dims[step_dim];
    }

    assert(step_dim >= 0 && "no zero padding is required");
    if (step_dim < 0) return;

    parallel_nd(nelems / step, [=](ptrdiff_t e1) {
        bool need_zero = false;

        ptrdiff_t idx = e1;
        for (int d = step_dim; d >= 0; --d) {
            if (idx % pdims[d] >= dims[d]) {
                need_zero = true;
                break;
            }
            idx /= pdims[d];
        }

        if (need_zero) {
            for (ptrdiff_t e0 = 0; e0 < step; ++e0) {
                const auto off = m_d.off_l(e1 * step + e0, true);
                const size_t start_bit
                        = static_cast<size_t>(off * bits_per_elem);
                const size_t end_bit = start_bit + bits_per_elem;
                const size_t first_byte = start_bit / 8;
                const size_t last_byte = (end_bit - 1) / 8;
                for (size_t b = first_byte; b <= last_byte; ++b) {
                    const int lo = b == first_byte
                            ? static_cast<int>(start_bit - b * 8)
                            : 0;
                    const int hi = b == last_byte
                            ? static_cast<int>(end_bit - b * 8)
                            : 8;
                    const uint8_t clear_mask = ~static_cast<uint8_t>(
                            ((1u << (hi - lo)) - 1) << lo);
                    reinterpret_cast<std::atomic<uint8_t> *>(&data[b])
                            ->fetch_and(clear_mask, std::memory_order_relaxed);
                }
            }
        }
    });
}

status_t typed_zero_pad(
        const memory_t *memory, const exec_ctx_t &ctx, size_t element_size) {
    const memory_desc_wrapper mdw(memory->md());
    memory_storage_t *memory_storage = memory->memory_storage();

    if (mdw.format_kind() != format_kind::blocked) return unimplemented;

    if (mdw.nelems(false) == mdw.nelems(true)) return success;

    const size_t map_size = mdw.size();
    assert(!is_runtime_value(map_size));

    void *mapped_ptr
            = ctx.map_memory_storage(memory_storage, ctx.stream(), map_size);

    auto *data = static_cast<uint8_t *>(mapped_ptr);
    auto blk = mdw.blocking_desc();

    auto get_blksize = [&](dim_t ind) {
        dim_t blksize = 1;
        for (int i = 0; i < blk.inner_nblks; i++) {
            if (blk.inner_idxs[i] == ind) blksize *= blk.inner_blks[i];
        }
        return blksize;
    };
    const dim_t blksize = get_blksize(blk.inner_idxs[0]);

    // Blocked tail handling is only valid for these block sizes; any other
    // value falls through to the generic implementation below.
    const bool supported_blksize = utils::one_of(blksize, 4, 8, 16);

#define CASE(blk_kind) \
    do { \
        if (supported_blksize) { \
            typed_zero_pad_blk<blk_kind>(mdw, data, element_size, blksize); \
            ctx.unmap_memory_storage( \
                    memory_storage, mapped_ptr, ctx.stream()); \
            return success; \
        } \
    } while (0)

    switch (blk.inner_nblks) {
        case 1:
            if (blk.inner_idxs[0] == 0) {
                CASE(a);
            } else if (blk.inner_idxs[0] == 1) {
                CASE(b);
            }
            break;
        case 2:
        case 3:
            if (blk.inner_nblks == 3 && blk.inner_idxs[0] != blk.inner_idxs[2])
                break;
            if (blksize != get_blksize(blk.inner_idxs[1])) break;

            if (blk.inner_idxs[0] == 0 && blk.inner_idxs[1] == 1) {
                CASE(ab);
            } else if (blk.inner_idxs[0] == 1 && blk.inner_idxs[1] == 0) {
                CASE(ba);
            }
            if (blk.inner_idxs[0] == 1 && blk.inner_idxs[1] == 2) {
                CASE(bc);
            } else if (blk.inner_idxs[0] == 2 && blk.inner_idxs[1] == 1) {
                CASE(cb);
            }
            break;
        default: break;
    }

#undef CASE

    // the last line of defence
    typed_zero_pad_generic_blocked(mdw, data, element_size);

    ctx.unmap_memory_storage(memory_storage, mapped_ptr, ctx.stream());
    return success;
}

status_t typed_zero_pad_sub_byte_entry(
        const memory_t *memory, const exec_ctx_t &ctx, int bits_per_elem) {
    const memory_desc_wrapper mdw(memory->md());
    memory_storage_t *memory_storage = memory->memory_storage();

    if (mdw.format_kind() != format_kind::blocked) return unimplemented;

    if (mdw.nelems(false) == mdw.nelems(true)) return success;

    const size_t map_size = mdw.size();
    assert(!is_runtime_value(map_size));

    void *mapped_ptr
            = ctx.map_memory_storage(memory_storage, ctx.stream(), map_size);

    typed_zero_pad_sub_byte(mdw, mapped_ptr, bits_per_elem);

    ctx.unmap_memory_storage(memory_storage, mapped_ptr, ctx.stream());
    return success;
}

static status_t zero_pad(const memory_t *memory, const exec_ctx_t &ctx) {
    memory_desc_wrapper mdw(memory->md());
    const auto dt = mdw.data_type();
    switch (dt) {
        case f4_e2m1:
        case s4:
        case u4:
        case u2:
            return typed_zero_pad_sub_byte_entry(
                    memory, ctx, types::data_type_bits(dt));
        case f16:
        case bf16:
        case e8m0:
        case f8_e5m2:
        case f8_e4m3:
        case f32:
        case s32:
        case s8:
        case u8:
        case f64: return typed_zero_pad(memory, ctx, types::data_type_size(dt));
        default: assert(!"memory is undefined"); return unimplemented;
    }
    return unimplemented;
}

status_t stream_t::zero_pad(const memory_t *memory, const exec_ctx_t &ctx) {
    return ::zero_pad(memory, ctx);
}

status_t memory_t::zero_pad(const exec_ctx_t &ctx) const {
    memory_desc_wrapper mdw(md());
    const bool skip_zeroing = false || memory_storage()->is_null()
            || mdw.is_zero() || !mdw.is_blocking_desc();
    if (skip_zeroing) return success;

    stream_t *stream = ctx.stream();
    status_t status;
    if (stream == nullptr) {
        engine_t *engine;
        engine = memory_storage()->engine();
        CHECK(engine->get_service_stream(stream));
    }

    if (stream != nullptr)
        status = stream->zero_pad(this, ctx);
    else
        status = ::zero_pad(this, ctx);

    return status;
}

extern "C" dnnl_status_t DNNL_API dnnl_impl_zero_pad(
        const memory_t *memory, stream_t *stream) {
    if (memory == nullptr || stream == nullptr)
        return status::invalid_arguments;
    memory_arg_t mem_arg = {const_cast<memory_t *>(memory), true};
    exec_args_t args = {{0, mem_arg}};
    return memory->zero_pad(exec_ctx_t(stream, std::move(args)));
}
