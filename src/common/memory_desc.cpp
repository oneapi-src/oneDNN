/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include <cctype>

#include "oneapi/dnnl/dnnl.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

namespace dnnl {
namespace impl {

status_t memory_desc_init_by_tag(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (ndims == 0 || tag == format_tag::undef) {
        memory_desc = types::zero_md();
        return success;
    }

    format_kind_t format_kind = types::format_tag_to_kind(tag);

    /* memory_desc != 0 */
    bool args_ok
            = memory_desc_sanity_check(ndims, dims, data_type, format_kind);
    VCHECK_MEMORY(args_ok, invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind;

    status_t status = success;
    if (tag == format_tag::any) {
        // nop
    } else if (format_kind == format_kind::blocked) {
        status = memory_desc_wrapper::compute_blocking(md, tag);
    } else {
        assert(!"unreachable");
        status = invalid_arguments;
    }

    if (status == success) memory_desc = md;

    return status;
}

status_t memory_desc_init_by_strides(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, const dims_t strides) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    /* memory_desc != 0 */
    bool args_ok = memory_desc_sanity_check(
            ndims, dims, data_type, format_kind::undef);
    VCHECK_MEMORY(args_ok, invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::blocked;

    dims_t default_strides = {0};
    if (strides == nullptr) {
        bool has_runtime_strides = false;
        default_strides[md.ndims - 1] = 1;
        for (int d = md.ndims - 2; d >= 0; --d) {
            if (md.padded_dims[d] == DNNL_RUNTIME_DIM_VAL)
                has_runtime_strides = true;
            default_strides[d] = has_runtime_strides
                    ? DNNL_RUNTIME_DIM_VAL
                    : default_strides[d + 1] * md.padded_dims[d + 1];
        }
        strides = default_strides;
    }
    VCHECK_MEMORY(memory_desc_strides_check(md, strides), invalid_arguments,
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    array_copy(md.format_desc.blocking.strides, strides, md.ndims);

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_csr_encoding(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt, data_type_t pointers_dt) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    // This is the only number of dims that is supported at this point.
    VCHECK_MEMORY(ndims <= 2, unimplemented, VERBOSE_BAD_NDIMS, "", ndims);

    bool args_ok = memory_desc_sanity_check(
            ndims, dims, data_type, format_kind::undef);
    VCHECK_MEMORY(args_ok, invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::csr;
    md.format_desc.sparse_desc.nnz = nnz;
    md.format_desc.sparse_desc.metadata_types[0] = indices_dt;
    md.format_desc.sparse_desc.metadata_types[1] = pointers_dt;

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_coo_encoding(memory_desc_t &memory_desc, int ndims,
        const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    // This is the only number of dims that is supported at this point.
    VCHECK_MEMORY(ndims <= 2, unimplemented, VERBOSE_BAD_NDIMS, "", ndims);

    bool args_ok = memory_desc_sanity_check(
            ndims, dims, data_type, format_kind::undef);
    VCHECK_MEMORY(args_ok, invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::coo;
    md.format_desc.sparse_desc.nnz = nnz;
    md.format_desc.sparse_desc.metadata_types[0] = indices_dt;

    memory_desc = md;

    return success;
}

status_t memory_desc_init_by_packed_encoding(memory_desc_t &memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz) {
    if (ndims == 0) {
        memory_desc = types::zero_md();
        return success;
    }

    bool args_ok = memory_desc_sanity_check(
            ndims, dims, data_type, format_kind::undef);
    VCHECK_MEMORY(args_ok, invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    auto md = memory_desc_t();
    md.ndims = ndims;
    array_copy(md.dims, dims, ndims);
    md.data_type = data_type;
    array_copy(md.padded_dims, dims, ndims);
    md.format_kind = format_kind::sparse;
    md.format_desc.sparse_desc.encoding = sparse_encoding::packed;
    md.format_desc.sparse_desc.nnz = nnz;

    memory_desc = md;

    return success;
}

status_t memory_desc_init_submemory(memory_desc_t &memory_desc,
        const memory_desc_t &parent_memory_desc, const dims_t dims,
        const dims_t offsets) {
    VCHECK_MEMORY(memory_desc_sanity_check(parent_memory_desc),
            invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);

    const memory_desc_wrapper src_d(parent_memory_desc);
    VCHECK_MEMORY((!src_d.has_runtime_dims_or_strides()), unimplemented,
            VERBOSE_UNSUPPORTED_MEM_STRIDE);

    for (int d = 0; d < src_d.ndims(); ++d) {
        VCHECK_MEMORY(
                !(utils::one_of(DNNL_RUNTIME_DIM_VAL, dims[d], offsets[d])),
                unimplemented, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

        const bool dim_offsets_oob = (dims[d] < 0 || offsets[d] < 0
                || (offsets[d] + dims[d] > src_d.dims()[d]));
        VCHECK_MEMORY(
                !dim_offsets_oob, invalid_arguments, VERBOSE_BAD_DIM, "src", d);
    }

    VCHECK_MEMORY(src_d.format_kind() == format_kind::blocked, unimplemented,
            VERBOSE_UNSUPPORTED_TAG);

    dims_t blocks;
    src_d.compute_blocks(blocks);

    memory_desc_t dst_d = parent_memory_desc;
    auto &dst_d_blk = dst_d.format_desc.blocking;

    /* TODO: put this into memory_desc_wrapper */
    for (int d = 0; d < src_d.ndims(); ++d) {
        const bool is_right_border = offsets[d] + dims[d] == src_d.dims()[d];

        /* very limited functionality for now */
        const bool ok = offsets[d] % blocks[d] == 0 /* [r1] */
                && src_d.padded_offsets()[d] == 0
                && IMPLICATION(!is_right_border,
                        (dims[d] % blocks[d] == 0 || dims[d] < blocks[d]));
        if (!ok) return unimplemented;

        dst_d.dims[d] = dims[d];
        dst_d.padded_dims[d] = is_right_border
                ? src_d.padded_dims()[d] - offsets[d]
                : dst_d.dims[d];
        dst_d.padded_offsets[d] = src_d.padded_offsets()[d];
        dst_d.offset0 += /* [r1] */
                offsets[d] / blocks[d] * dst_d_blk.strides[d];
    }

    memory_desc = dst_d;

    return success;
}

status_t memory_desc_reshape(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, int ndims, const dims_t dims) {
    auto volume = [](const dim_t *dims, int ndims) -> dim_t {
        dim_t prod = 1;
        for (int i = 0; i < ndims; ++i) {
            if (dims[i] == DNNL_RUNTIME_DIM_VAL) return DNNL_RUNTIME_DIM_VAL;
            prod *= dims[i] > 0 ? dims[i] : 1;
        }
        return prod;
    };

    VCHECK_MEMORY(memory_desc_sanity_check(in_memory_desc), invalid_arguments,
            VERBOSE_MEM_DESC_CHECK_FAIL);
    VCHECK_MEMORY(memory_desc_sanity_check(ndims, dims,
                          in_memory_desc.data_type, in_memory_desc.format_kind),
            invalid_arguments, VERBOSE_MEM_DESC_CHECK_FAIL);
    VCHECK_MEMORY(one_of(in_memory_desc.format_kind, format_kind::any,
                          format_kind::blocked),
            invalid_arguments, VERBOSE_UNSUPPORTED_TAG);
    VCHECK_MEMORY(!types::is_zero_md(&in_memory_desc), invalid_arguments,
            VERBOSE_NULL_ARG);
    VCHECK_MEMORY(volume(in_memory_desc.dims, in_memory_desc.ndims)
                    == volume(dims, ndims),
            invalid_arguments, VERBOSE_SHAPE_RESTRICTION);
    VCHECK_MEMORY(
            !memory_desc_wrapper(in_memory_desc).has_runtime_dims_or_strides(),
            invalid_arguments, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    VCHECK_MEMORY(in_memory_desc.extra.flags == 0, invalid_arguments,
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    if (in_memory_desc.format_kind == format_kind::any)
        return memory_desc_init_by_tag(out_memory_desc, ndims, dims,
                in_memory_desc.data_type, format_tag::any);

    assert(in_memory_desc.format_kind == format_kind::blocked);
    assert(in_memory_desc.extra.flags == 0);

    // temporary output
    auto md = in_memory_desc;

    md.ndims = ndims;
    array_copy(md.dims, dims, md.ndims);

    const int i_ndims = in_memory_desc.ndims;
    const int o_ndims = md.ndims;

    const auto &i_dims = in_memory_desc.dims,
               &i_pdims = in_memory_desc.padded_dims;
    const auto &o_dims = md.dims;

    const auto &i_bd = in_memory_desc.format_desc.blocking;
    auto &o_bd = md.format_desc.blocking;

    dims_t blocks = {0};
    memory_desc_wrapper(in_memory_desc).compute_blocks(blocks);

    enum class action_t { REMOVE_1, ADD_1, KEEP_DIM, REARRANGE_DIMS, FAIL };

    // Determine groups in input and output dims starting with the given
    // positions (going backwards) that satisfy one of the conditions:
    // - REMOVE_1
    //      input_group = {1}, output_group = empty
    // - ADD_1
    //      input_group = empty, output_group = {1}
    // - KEEP_DIM
    //      input_group = {x}, output_group = {x}
    // - REARRANGE_DIMS
    //      input_group = {x1, x2, .., xk}, output_group = {y1, y2, ..., ym}
    //      and product(x_i) = product(y_j), and the groups are minimal
    // - FAIL
    //      invalid configuration (return false)
    auto find_groups
            = [&](int &i_group_begin, int i_group_end, int &o_group_begin,
                      int o_group_end) -> action_t {
        // 1st step: check for `1` in the input dims
        if (i_group_end > 0 && i_dims[i_group_end - 1] == 1) {
            i_group_begin = i_group_end - 1;
            if (i_pdims[i_group_end - 1] == 1) {
                o_group_begin = o_group_end;
                return action_t::REMOVE_1;
            } else if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
                o_group_begin = o_group_end - 1;
                return action_t::KEEP_DIM;
            } else {
                return action_t::FAIL;
            }
        }

        // 2nd step: check for `1` in the output dims
        if (o_group_end > 0 && o_dims[o_group_end - 1] == 1) {
            i_group_begin = i_group_end;
            o_group_begin = o_group_end - 1;
            return action_t::ADD_1;
        }

        // at this moment both groups cannot be empty
        if (i_group_end == 0 || o_group_end == 0) return action_t::FAIL;

        // 3rd step: find the non-trivial groups of the same volume
        i_group_begin = i_group_end - 1;
        o_group_begin = o_group_end - 1;

        dim_t i_volume = i_dims[i_group_begin];
        dim_t o_volume = o_dims[o_group_begin];

        while (i_volume != o_volume) {
            if (i_volume < o_volume) {
                if (i_group_begin == 0) return action_t::FAIL;
                i_volume *= i_dims[--i_group_begin];

                // do not allow `0` axis in the middle
                if (i_volume == 0) return action_t::FAIL;
            } else {
                if (o_group_begin == 0) return action_t::FAIL;
                o_volume *= o_dims[--o_group_begin];

                // do not allow `0` axis in the middle
                if (o_volume == 0) return action_t::FAIL;
            }
        }

        assert(i_volume == o_volume);
        assert(i_group_begin >= 0);
        assert(o_group_begin >= 0);

        return (i_group_begin + 1 == i_group_end
                       && o_group_begin + 1 == o_group_end)
                ? action_t::KEEP_DIM
                : action_t::REARRANGE_DIMS;
    };

    int i_group_begin = i_ndims, i_group_end = i_ndims;
    int o_group_begin = o_ndims, o_group_end = o_ndims;

    while (i_group_end != 0 || o_group_end != 0) {
        action_t action = find_groups(
                i_group_begin, i_group_end, o_group_begin, o_group_end);

        if (action == action_t::REMOVE_1) {
            // nop, padding is already taken into account by `find_groups()`
        } else if (action == action_t::ADD_1) {
            // get the stride from the right
            dim_t current_stride = 1;
            if (i_group_begin == i_ndims) {
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    current_stride *= i_bd.inner_blks[d];
            } else {
                // Add `1` to the left from axes with index `i_group_begin`
                current_stride
                        = i_bd.strides[i_group_begin] * i_dims[i_group_begin];
                for (int d = 0; d < i_bd.inner_nblks; ++d)
                    if (i_bd.inner_idxs[d] == i_group_begin)
                        current_stride /= i_bd.inner_blks[d];
            }
            md.padded_dims[o_group_begin] = 1;
            md.padded_offsets[o_group_begin] = 0;
            o_bd.strides[o_group_begin] = current_stride;
        } else if (action == action_t::KEEP_DIM) {
            // change the axis index from `i_group_begin` to `o_group_begin`
            assert(i_group_begin + 1 == i_group_end);
            assert(o_group_begin + 1 == o_group_end);

            md.padded_dims[o_group_begin]
                    = in_memory_desc.padded_dims[i_group_begin];
            md.padded_offsets[o_group_begin]
                    = in_memory_desc.padded_offsets[i_group_begin];
            o_bd.strides[o_group_begin] = i_bd.strides[i_group_begin];
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    o_bd.inner_idxs[d] = o_group_begin;
        } else if (action == action_t::REARRANGE_DIMS) {
            // check that input group is dense, sequential, and is not blocked
            for (int d = i_group_end - 1; d > i_group_begin; --d)
                if (i_dims[d] * i_bd.strides[d] != i_bd.strides[d - 1])
                    return invalid_arguments;

            // checked (i_group_begin, i_group_end), `i_group_begin` remains
            for (int d = 0; d < i_bd.inner_nblks; ++d)
                if (i_bd.inner_idxs[d] == i_group_begin)
                    return invalid_arguments;
            if (in_memory_desc.padded_dims[i_group_begin]
                    != i_dims[i_group_begin])
                return invalid_arguments;
            if (in_memory_desc.padded_offsets[i_group_begin] != 0)
                return invalid_arguments;

            // oK, fill output md according to
            // o_dims[o_group_begin .. o_group_end]

            dim_t current_stride = i_bd.strides[i_group_end - 1];
            for (int d = o_group_end - 1; d >= o_group_begin; --d) {
                md.padded_dims[d] = o_dims[d];
                md.padded_offsets[d] = 0;
                o_bd.strides[d] = current_stride;
                current_stride *= md.padded_dims[d];
            }
        } else {
            assert(action == action_t::FAIL);
            return invalid_arguments;
        }

        i_group_end = i_group_begin;
        o_group_end = o_group_begin;
    }

    out_memory_desc = md;
    return success;
}

status_t memory_desc_permute_axes(memory_desc_t &out_memory_desc,
        const memory_desc_t &in_memory_desc, const int *perm) {
    VCHECK_MEMORY(memory_desc_sanity_check(in_memory_desc), invalid_arguments,
            VERBOSE_MEM_DESC_CHECK_FAIL);
    VCHECK_MEMORY(one_of(in_memory_desc.format_kind, format_kind::any,
                          format_kind::blocked),
            invalid_arguments, VERBOSE_UNSUPPORTED_TAG);
    VCHECK_MEMORY(!types::is_zero_md(&in_memory_desc), invalid_arguments,
            VERBOSE_NULL_ARG);
    VCHECK_MEMORY(
            !memory_desc_wrapper(in_memory_desc).has_runtime_dims_or_strides(),
            invalid_arguments, VERBOSE_UNSUPPORTED_MEM_STRIDE);
    VCHECK_MEMORY(in_memory_desc.extra.flags == 0, invalid_arguments,
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    // verify that perm is indeed a permutation of [0 .. ndims)
    unsigned occurrence_mask = 0;
    for (int d = 0; d < in_memory_desc.ndims; ++d)
        if (0 <= perm[d] && perm[d] < in_memory_desc.ndims)
            occurrence_mask |= (1u << perm[d]);
    VCHECK_MEMORY((occurrence_mask + 1 == (1u << in_memory_desc.ndims)),
            invalid_arguments, VERBOSE_BAD_NDIMS, "in_memory_desc",
            in_memory_desc.ndims);

    out_memory_desc = in_memory_desc;
    for (int d = 0; d < in_memory_desc.ndims; ++d) {
        if (perm[d] == d) continue;
        out_memory_desc.dims[perm[d]] = in_memory_desc.dims[d];
        out_memory_desc.padded_dims[perm[d]] = in_memory_desc.padded_dims[d];
        out_memory_desc.padded_offsets[perm[d]]
                = in_memory_desc.padded_offsets[d];
        if (in_memory_desc.format_kind == format_kind::blocked) {
            const auto &i_bd = in_memory_desc.format_desc.blocking;
            auto &o_bd = out_memory_desc.format_desc.blocking;

            o_bd.strides[perm[d]] = i_bd.strides[d];
            for (int blk = 0; blk < i_bd.inner_nblks; ++blk)
                if (i_bd.inner_idxs[blk] == d) o_bd.inner_idxs[blk] = perm[d];
        }
    }

    return success;
}

// This is only used by internal API that is used for testing only.
status_t memory_desc_init_by_string_tag(memory_desc_t &md, int ndims,
        const dims_t dims, data_type_t data_type, const std::string &tag) {
    // Copy to temporary to handle dims == md->dims case.
    dims_t tmp_dims;
    std::copy(dims, dims + ndims, tmp_dims);

    md.ndims = ndims;
    VCHECK_MEMORY(!(ndims < 0 || ndims > DNNL_MAX_NDIMS), invalid_arguments,
            VERBOSE_BAD_NDIMS, "", ndims);

    std::copy(tmp_dims, tmp_dims + ndims, md.dims);
    md.data_type = data_type;
    md.format_kind = format_kind::blocked;

    // Parse dimensions and their block sizes starting from the innermost one.
    std::vector<std::pair<int, int>> dim_blocks;
    int pos = (int)tag.size() - 1;
    int ndims_from_tag = -1;
    while (pos >= 0) {
        int pos0 = pos;

        --pos;
        while (pos >= 0 && std::isdigit(tag[pos]))
            pos--;

        int dim_idx = std::tolower(tag[pos0]) - 'a';
        VCHECK_MEMORY(dim_idx < ndims, invalid_arguments, VERBOSE_BAD_NDIMS, "",
                ndims);
        ndims_from_tag = std::max(dim_idx + 1, ndims_from_tag);
        int block_str_len = pos0 - pos - 1;
        bool is_blocked = block_str_len > 0;
        int block = is_blocked ? std::stoi(tag.substr(pos + 1, block_str_len))
                               : 1;
        if (is_blocked && block == 1) continue; // skip trivial blocks
        dim_blocks.emplace_back(dim_idx, block);
    }
    VCHECK_MEMORY((ndims_from_tag == ndims), invalid_arguments,
            VERBOSE_BAD_NDIMS, "", ndims);

    auto &blk = md.format_desc.blocking;

    // Compute strides and fill inner block sizes/indices.
    dim_t stride = 1;
    dims_t full_inner_blks;
    std::fill(full_inner_blks, full_inner_blks + ndims, 1);
    for (auto &p : dim_blocks) {
        int dim_idx = p.first;
        int block = p.second;
        if (block == 1) {
            assert(blk.strides[dim_idx] == 0);
            blk.strides[dim_idx] = stride;

            dim_t fib = full_inner_blks[dim_idx];
            dim_t padded_dim = md.dims[dim_idx] == DNNL_RUNTIME_DIM_VAL
                    ? DNNL_RUNTIME_DIM_VAL
                    : (md.dims[dim_idx] + fib - 1) / fib * fib;
            md.padded_dims[dim_idx] = padded_dim;
            if (one_of(DNNL_RUNTIME_DIM_VAL, padded_dim, stride))
                stride = DNNL_RUNTIME_DIM_VAL;
            else
                stride *= (padded_dim / fib);
        } else {
            full_inner_blks[dim_idx] *= block;
            blk.inner_blks[blk.inner_nblks] = block;
            blk.inner_idxs[blk.inner_nblks] = dim_idx;
            blk.inner_nblks++;
            stride *= block;
        }
    }

    // Inner block sizes/indices are stored from the outermost to the innermost
    // so need to reverse them.
    std::reverse(blk.inner_blks, blk.inner_blks + blk.inner_nblks);
    std::reverse(blk.inner_idxs, blk.inner_idxs + blk.inner_nblks);

    return success;
}

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_memory_desc_create_with_tag(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, format_tag_t tag) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_tag(*md, ndims, dims, data_type, tag));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_strides(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type,
        const dims_t strides) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_strides(*md, ndims, dims, data_type, strides));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_csr_encoding(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt, data_type_t pointers_dt) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_csr_encoding(
            *md, ndims, dims, data_type, nnz, indices_dt, pointers_dt));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_coo_encoding(memory_desc_t **memory_desc,
        int ndims, const dims_t dims, data_type_t data_type, dim_t nnz,
        data_type_t indices_dt) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_coo_encoding(
            *md, ndims, dims, data_type, nnz, indices_dt));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_with_packed_encoding(
        memory_desc_t **memory_desc, int ndims, const dims_t dims,
        data_type_t data_type, dim_t nnz) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_packed_encoding(
            *md, ndims, dims, data_type, nnz));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_create_submemory(memory_desc_t **memory_desc,
        const memory_desc_t *parent_memory_desc, const dims_t dims,
        const dims_t offsets) {
    if (any_null(memory_desc, parent_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_submemory(*md, *parent_memory_desc, dims, offsets));
    (*memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_reshape(memory_desc_t **out_memory_desc,
        const memory_desc_t *in_memory_desc, int ndims, const dims_t dims) {
    if (any_null(out_memory_desc, in_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_reshape(*md, *in_memory_desc, ndims, dims));
    (*out_memory_desc) = md.release();
    return success;
}

status_t dnnl_memory_desc_permute_axes(memory_desc_t **out_memory_desc,
        const memory_desc_t *in_memory_desc, const int *perm) {
    if (any_null(out_memory_desc, in_memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_permute_axes(*md, *in_memory_desc, perm));
    (*out_memory_desc) = md.release();
    return success;
}

int dnnl_memory_desc_equal(const memory_desc_t *lhs, const memory_desc_t *rhs) {
    if (lhs == rhs) return 1;
    if (any_null(lhs, rhs)) return 0;
    return memory_desc_wrapper(*lhs) == memory_desc_wrapper(*rhs);
}

size_t dnnl_memory_desc_get_size(const memory_desc_t *md) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size();
}

size_t dnnl_memory_desc_get_size_v2(const memory_desc_t *md, int index) {
    if (md == nullptr) return 0;
    return memory_desc_wrapper(*md).size(index);
}

size_t dnnl_data_type_size(dnnl_data_type_t data_type) {
    return types::data_type_size(data_type);
}

status_t dnnl_memory_desc_query(
        const memory_desc_t *md, query_t what, void *result) {
    if (any_null(md, result)) return invalid_arguments;

    const bool is_blocked = md->format_kind == format_kind::blocked;

    switch (what) {
        case query::ndims_s32: *(int32_t *)result = md->ndims; break;
        case query::dims: *(const dims_t **)result = &md->dims; break;
        case query::data_type: *(data_type_t *)result = md->data_type; break;
        case query::submemory_offset_s64: *(dim_t *)result = md->offset0; break;
        case query::padded_dims:
            *(const dims_t **)result = &md->padded_dims;
            break;
        case query::padded_offsets:
            *(const dims_t **)result = &md->padded_offsets;
            break;
        case query::format_kind:
            switch ((int)md->format_kind) {
                case format_kind::rnn_packed:
                case format_kind::cublaslt_blocked:
                case format_kind::wino:
                    *(format_kind_t *)result = format_kind::opaque;
                    break;
                default: *(format_kind_t *)result = md->format_kind;
            }
            break;
        case query::strides:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.strides;
            break;
        case query::inner_nblks_s32:
            if (!is_blocked) return status::invalid_arguments;
            *(int32_t *)result = md->format_desc.blocking.inner_nblks;
            break;
        case query::inner_blks:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_blks;
            break;
        case query::inner_idxs:
            if (!is_blocked) return status::invalid_arguments;
            *(const dims_t **)result = &md->format_desc.blocking.inner_idxs;
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

status_t dnnl_memory_desc_query_v2(
        const memory_desc_t *md, query_t what, int index, void *result) {
    if (any_null(md, result)) return invalid_arguments;
    const bool is_sparse = md->format_kind == format_kind::sparse;
    VCHECK_MEMORY(
            !((!is_sparse && index > 0)
                    || (is_sparse && index > 0 && what != query::data_type)),
            invalid_arguments, VERBOSE_UNSUPPORTED_SPARSE_CFG);

    switch (what) {
        case query::sparse_encoding:
            if (!is_sparse) return status::invalid_arguments;
            *(sparse_encoding_t *)result = md->format_desc.sparse_desc.encoding;
            break;
        case query::nnz_s64:
            if (!is_sparse) return status::invalid_arguments;
            *(dim_t *)result = md->format_desc.sparse_desc.nnz;
            break;
        case query::data_type:
            *(data_type_t *)result = (index == 0)
                    ? md->data_type
                    : md->format_desc.sparse_desc.metadata_types
                              [md->format_desc.sparse_desc.encoding
                                                      == sparse_encoding_t::
                                                              dnnl_coo
                                              ? 0
                                              : index - 1];
            break;
        case query::num_handles_s32:
            if (is_sparse) {
                switch (md->format_desc.sparse_desc.encoding) {
                    case sparse_encoding::csr:
                    case sparse_encoding::coo:
                        *(int *)result = md->ndims + 1;
                        break;
                    case sparse_encoding::packed: *(int *)result = 3; break;
                    default: assert(!"unknown encoding"); *(int *)result = 0;
                }
            } else
                *(int *)result = 1;
            break;
        default: return dnnl_memory_desc_query(md, what, result);
    }
    return status::success;
}

status_t dnnl_memory_desc_destroy(memory_desc_t *memory_desc) {
    delete memory_desc;
    return success;
}

status_t dnnl_memory_desc_clone(memory_desc_t **memory_desc,
        const memory_desc_t *existing_memory_desc) {
    (*memory_desc) = new memory_desc_t(*existing_memory_desc);
    return success;
}

status_t dnnl_memory_desc_get_blob(
        uint8_t *blob, size_t *size, const memory_desc_t *md) {
    if (md == nullptr || (blob == nullptr && size == nullptr))
        return invalid_arguments;
    if (blob != nullptr)
        memcpy(blob, md, *size);
    else if (size != nullptr)
        *size = sizeof(memory_desc_t);

    return success;
}

status_t dnnl_memory_desc_create_with_blob(
        memory_desc_t **md, const uint8_t *blob) {
    if (one_of(nullptr, md, blob)) return invalid_arguments;

    *md = new memory_desc_t();
    memcpy(*md, blob, sizeof(memory_desc_t));
    return success;
}

// This is an internal API that is used only for testing in benchdnn.
extern "C" status_t DNNL_API dnnl_memory_desc_create_with_string_tag(
        memory_desc_t **memory_desc, int ndims, const dims_t dims,
        data_type_t data_type, const char *tag) {
    if (any_null(memory_desc)) return invalid_arguments;

    auto md = utils::make_unique<memory_desc_t>();
    if (!md) return out_of_memory;
    CHECK(memory_desc_init_by_string_tag(*md, ndims, dims, data_type, tag));
    (*memory_desc) = md.release();
    return success;
}

extern "C" status_t DNNL_API dnnl_memory_desc_set_data_type(
        memory_desc_t *memory_desc, data_type_t data_type) {
    if (any_null(memory_desc)) return invalid_arguments;
    memory_desc->data_type = data_type;
    return success;
}
