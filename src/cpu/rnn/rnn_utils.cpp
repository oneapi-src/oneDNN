/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
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

#include <initializer_list>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/rnn.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm/gemm_pack.hpp"

#include "cpu/rnn/ref_rnn.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;
using namespace rnn_utils;
using namespace format_tag;
using namespace rnn_packed_format;
using namespace data_type;

static bool check_dims_contiguous_except_one(const memory_desc_wrapper &mdw,
        int idx_with_arbitrary_stride, std::initializer_list<int> perm) {
    if (mdw.format_kind() != format_kind::blocked) return false;
    if ((size_t)mdw.ndims() != perm.size()) return false;

    const auto &blk = mdw.blocking_desc();

    dim_t expect_stride = 1;
    for (int idx = mdw.ndims() - 1; idx >= 0; --idx) {
        const int permuted_idx = *(perm.begin() + idx);
        const bool ok = (idx == idx_with_arbitrary_stride)
                ? expect_stride <= blk.strides[permuted_idx]
                : expect_stride == blk.strides[permuted_idx];
        if (!ok) return false;
        expect_stride = mdw.dims()[permuted_idx] * blk.strides[permuted_idx];
    }

    return true;
}

bool rnn_utils::is_ldigo(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 2, 3, 4});
}

bool rnn_utils::is_ldgoi(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 3, {0, 1, 3, 4, 2});
}

bool rnn_utils::is_ldio(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 2, 3});
}

bool rnn_utils::is_ldoi(const memory_desc_wrapper &mdw) {
    return check_dims_contiguous_except_one(mdw, 2, {0, 1, 3, 2});
}

bool rnn_utils::is_ldigo_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(format_tag::ldgOi32o,
            format_tag::ldgOI32o2i, format_tag::ldgOI32o4i,
            format_tag::ldgOI16o4i, format_tag::ldgOI64o2i,
            format_tag::ldgOI64o4i, format_tag::ldgOi16o);
    return md_format_tag != format_tag::undef;
}

bool rnn_utils::is_ldgoi_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(
            format_tag::ldgIo32i, format_tag::ldgIO32i2o, format_tag::ldgIo16i);
    return md_format_tag != format_tag::undef;
}

bool rnn_utils::is_ldio_blocked(const memory_desc_wrapper &mdw) {
    format_tag_t md_format_tag = mdw.matches_one_of_tag(format_tag::ldOi32o,
            format_tag::ldOI32o4i, ldOI16o4i, format_tag::ldOi16o);
    return md_format_tag != format_tag::undef;
}

int rnn_utils::get_good_ld(int dim, int sizeof_dt) {
    // we want matrices leading dimentions to be 64-byte aligned,
    // and not divisible by 256 to avoid 4K aliasing effects
    const int ld = rnd_up(dim, 64 / sizeof_dt);
    return (ld % 256 == 0) ? ld + 64 / sizeof_dt : ld;
}

void rnn_utils::set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_ht_offset, size_t &ws_states_layer_offset,
        size_t &ws_states_iter_offset, size_t &ws_states_iter_c_offset,
        size_t &ws_diff_states_layer_offset, size_t &ws_diff_states_iter_offset,
        size_t &ws_diff_states_iter_c_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_ht_offset, size_t &scratch_diff_ht_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size) {

    const size_t page_size = 4096; // 2097152;
    size_t current_offset;
    /* Mandatory workspaces: go to workspace if use_workspace, scratchpad
     * otherwise */
    current_offset = 0; // assumes the workspace base pointer is page aligned

#define register_space(a) \
    do { \
        current_offset = utils::rnd_up(current_offset, page_size); \
        CONCAT2(a, _offset) = current_offset; \
        current_offset += rnn.CONCAT2(a, _size); \
    } while (false)

    register_space(ws_gates);
    register_space(ws_ht);
    register_space(ws_states_layer);
    register_space(ws_states_iter);
    register_space(ws_states_iter);

    // For all currently supported cells, ws_iter should not be used
    // at all since dst_iter == dst_layer
    assert(rnn.ws_states_layer_size == rnn.ws_states_iter_size);
    ws_states_iter_offset = ws_states_layer_offset;

    register_space(ws_states_iter_c);
    register_space(ws_diff_states_layer);
    register_space(ws_diff_states_iter);
    register_space(ws_diff_states_iter_c);
    register_space(ws_grid_comp);

    workspace_size = rnn.use_workspace ? current_offset : 0;

    /* Optional scratchpads */
    // Assumes the scratchpad base pointer is page aligned.
    // If use_workspace, the following goes to scratchpad alone,
    // otherwise, all goes to scratchpad and continue incrementing offset
    current_offset = rnn.use_workspace ? 0 : current_offset;

    register_space(scratch_gates);
    register_space(scratch_ht);
    register_space(scratch_diff_ht);
    register_space(scratch_cell);
    if (rnn.copy_bias)
        register_space(ws_bias);
    else
        ws_bias_offset = 0;

    scratchpad_size = current_offset;
#undef register_space
}

void rnn_utils::get_scratchpad_and_workspace_sizes(const rnn_conf_t &rnn,
        size_t &scratchpad_size, size_t &workspace_size) {
    size_t ws_gates_offset, ws_ht_offset, ws_states_layer_offset,
            ws_states_iter_offset, ws_states_iter_c_offset,
            ws_diff_states_layer_offset, ws_diff_states_iter_offset,
            ws_diff_states_iter_c_offset, ws_grid_comp_offset,
            scratch_gates_offset, scratch_ht_offset, scratch_diff_ht_offset,
            scratch_cell_offset, ws_bias_offset;
    set_offsets(rnn, ws_gates_offset, ws_ht_offset, ws_states_layer_offset,
            ws_states_iter_offset, ws_states_iter_c_offset,
            ws_diff_states_layer_offset, ws_diff_states_iter_offset,
            ws_diff_states_iter_c_offset, ws_grid_comp_offset, ws_bias_offset,
            scratch_gates_offset, scratch_ht_offset, scratch_diff_ht_offset,
            scratch_cell_offset, scratchpad_size, workspace_size);
}

status_t rnn_utils::set_good_strides(
        memory_desc_t &weights_md, format_tag_t tag) {
    auto &strides = weights_md.format_desc.blocking.strides;
    const auto dims = weights_md.dims;

    int ld_dim_idx = 0;
    switch (tag) {
        case ldio:
        case ldigo:
            strides[2] = rnn_utils::get_good_ld((int)strides[2],
                    (int)types::data_type_size(weights_md.data_type));
            ld_dim_idx = 2;
            break;
        case ldoi:
        case ldgoi:
            strides[weights_md.ndims - 1]
                    = rnn_utils::get_good_ld((int)strides[weights_md.ndims - 1],
                            (int)types::data_type_size(weights_md.data_type));
            if (tag == ldgoi) strides[3] = dims[4] * strides[4];
            ld_dim_idx = 3;
            break;
        default: return status::unimplemented;
    }
    strides[1] = dims[ld_dim_idx] * strides[ld_dim_idx];
    strides[0] = dims[1] * strides[1];

    return status::success;
}

status_t rnn_utils::set_expected_desc(rnn_conf_t &rnn,
        memory_desc_t &weights_md, rnn_utils::weights_type_t weights_type) {
    using namespace rnn_utils;
    bool use_packed_gemm = false;
    switch (weights_type) {
        case weights_type_t::layer:
            use_packed_gemm = rnn.use_layer_packed_gemm;
            break;
        case weights_type_t::iter:
            use_packed_gemm = rnn.use_iter_packed_gemm;
            break;
        case weights_type_t::projection:
            use_packed_gemm = rnn.use_projection_packed_gemm;
            break;
        default: assert(!"unsupported weights type");
    }

    if (use_packed_gemm) {
        weights_md.format_kind = format_kind::rnn_packed;
        rnn_packed_desc_t &rnn_pdata = weights_md.format_desc.rnn_packed_desc;
        switch (weights_type) {
            case weights_type_t::iter:
                rnn_pdata.format = rnn.is_fwd
                        ? rnn_packed_memory_format_t::ldigo_p
                        : rnn_packed_memory_format_t::ldgoi_p;
                rnn_pdata.ldb = rnn.ws_states_iter_ld;
                rnn_pdata.n = rnn.mb;
                rnn_pdata.n_parts = rnn.n_parts_weights_iter;
                array_copy(rnn_pdata.parts, rnn.parts_weights_iter,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_iter_pack_size, DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation = rnn.weights_iter_comp_offset;
                rnn_pdata.size = rnn.weights_iter_pack_size;
                break;
            case weights_type_t::layer:
                rnn_pdata.format = rnn.is_fwd
                        ? rnn_packed_memory_format_t::ldigo_p
                        : rnn_packed_memory_format_t::ldgoi_p;
                rnn_pdata.ldb = rnn.ws_states_layer_ld;
                rnn_pdata.n
                        = rnn.merge_gemm_layer ? rnn.n_iter * rnn.mb : rnn.mb;
                rnn_pdata.n_parts = rnn.n_parts_weights_layer;
                array_copy(rnn_pdata.parts, rnn.parts_weights_layer,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_layer_pack_size, DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation = rnn.weights_layer_comp_offset;
                rnn_pdata.size = rnn.weights_layer_pack_size;
                break;
            case weights_type_t::projection:
                // TODO: add ldoi_p for bwd?
                rnn_pdata.format = rnn_packed_memory_format_t::ldio_p;
                rnn_pdata.ldb = rnn.proj_ht_ld;
                rnn_pdata.n = rnn.mb;
                rnn_pdata.n_parts = rnn.n_parts_weights_projection;
                array_copy(rnn_pdata.parts, rnn.parts_weights_projection,
                        DNNL_RNN_MAX_N_PARTS);
                array_copy(rnn_pdata.part_pack_size,
                        rnn.part_weights_projection_pack_size,
                        DNNL_RNN_MAX_N_PARTS);
                rnn_pdata.offset_compensation
                        = rnn.weights_projection_comp_offset;
                rnn_pdata.size = rnn.weights_projection_pack_size;
                break;
            default: assert(!"unsupported weights type");
        }
        if (rnn.is_signed_int8_conf()) {
            weights_md.extra.flags
                    = 0 | memory_extra_flags::rnn_s8s8_compensation;
            weights_md.extra.compensation_mask = 0;
        }
    } else {
        using namespace format_tag;
        if (rnn.is_brgemm) {
            const int n_block = rnn.n_block;
            format_tag_t tag = format_tag::undef;

            if (weights_type == weights_type_t::projection) {
                if (rnn.is_int8_conf())
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldOI32o4i, 16, format_tag::ldOI16o4i);
                else
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldOi32o, 16, format_tag::ldOi16o);
            } else if (rnn.is_fwd) {
                if (rnn.is_int8_conf())
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOI64o4i, 32, ldgOI32o4i, 16,
                            ldgOI16o4i);
                else if (rnn.is_xf16_conf())
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOI64o2i, 32, ldgOI32o2i);
                else {
                    assert(IMPLICATION(n_block == 64, rnn.is_bf32()));
                    tag = utils::map(n_block, format_tag::undef, 64,
                            format_tag::ldgOi32o /*bf32*/, 32,
                            format_tag::ldgOi32o, 16, ldgOi16o);
                }
            } else {
                if (rnn.is_xf16_conf())
                    tag = format_tag::ldgIO32i2o;
                else
                    tag = utils::map(n_block, format_tag::undef, 32,
                            format_tag::ldgIo32i, 16, format_tag::ldgIo16i);
            }

            if (tag != format_tag::undef) {
                CHECK(memory_desc_init_by_tag(weights_md, tag));
                if (rnn.is_unsigned_int8_conf()) {
                    weights_md.extra.flags
                            = 0 | memory_extra_flags::rnn_u8s8_compensation;
                    weights_md.extra.compensation_mask
                            = (weights_type == weights_type_t::projection)
                            ? 13 // 1101
                            : 27; // 11011
                } else if (rnn.is_signed_int8_conf()) {
                    weights_md.extra.flags
                            = 0 | memory_extra_flags::rnn_s8s8_compensation;
                    weights_md.extra.compensation_mask = 0;
                }
                return status::success;
            } else {
                return status::unimplemented;
            }
        } else {
            const format_tag_t tag = weights_type == weights_type_t::projection
                    ? rnn.is_fwd ? ldio : ldoi
                    : rnn.is_fwd ? ldigo
                                 : ldgoi;
            CHECK(memory_desc_init_by_tag(weights_md, tag));
            // Adjust strides for good leading dimension in GEMM
            CHECK(set_good_strides(weights_md, tag));
        }
    }
    return status::success;
}

float rnn_utils::to_float(const void *data, const data_type_t dt) {
    if (dt == data_type::f32)
        return *static_cast<const float *>(data);
    else if (dt == data_type::bf16)
        return float(*static_cast<const bfloat16_t *>(data));
    else if (dt == data_type::f16)
        return float(*static_cast<const float16_t *>(data));
    return 0.0;
}

const void *rnn_utils::inc_ptr(
        const void *data, data_type_t data_type, int offset) {
    if (data_type == data_type::f32)
        return static_cast<const float *>(data) + offset;
    else if (data_type == data_type::bf16)
        return static_cast<const bfloat16_t *>(data) + offset;
    else if (data_type == data_type::f16)
        return static_cast<const float16_t *>(data) + offset;
    else
        return data;
}

void *rnn_utils::inc_ptr(void *data, data_type_t data_type, int offset) {
    return const_cast<void *>(
            inc_ptr(static_cast<const void *>(data), data_type, offset));
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
