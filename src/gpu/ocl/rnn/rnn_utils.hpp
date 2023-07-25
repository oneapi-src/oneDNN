/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_OCL_RNN_RNN_UTILS_HPP
#define GPU_OCL_RNN_RNN_UTILS_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "gpu/primitive_conf.hpp"
#include "gpu/serialization.hpp"

#define OFF6(i0, d0, i1, d1, i2, d2, i3, d3, i4, d4, i5, d5) \
    ((((((i0) * (d1) + (i1)) * (d2) + (i2)) * (d3) + (i3)) * (d4) + (i4)) \
                    * (d5) \
            + (i5))
#define OFF5(i0, d0, i1, d1, i2, d2, i3, d3, i4, d4) \
    (((((i0) * (d1) + (i1)) * (d2) + (i2)) * (d3) + (i3)) * (d4) + (i4))
#define OFF4(i0, d0, i1, d1, i2, d2, i3, d3) \
    ((((i0) * (d1) + (i1)) * (d2) + (i2)) * (d3) + (i3))
#define OFF3(i0, d0, i1, d1, i2, d2) (((i0) * (d1) + (i1)) * (d2) + (i2))
#define OFF2(i0, d0, i1, d1) ((i0) * (d1) + (i1))

#define elemwise_sig(f) \
    status_t f(const exec_ctx_t &ctx, dim_t dir, dim_t lay, dim_t iter, \
            dim_t dhc, dim_t batch, dim_t bwd_batch_block, \
            const workspace_t &workspace, \
            const memory_storage_t &scratch_gates, \
            const memory_storage_t &scratch_diff_states, \
            const memory_storage_t *scales, const memory_storage_t &bias, \
            const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias) const

#define elemwise_sig_gru_lbr(f) \
    status_t f(const exec_ctx_t &ctx, dim_t dir, dim_t lay, dim_t iter, \
            dim_t dhc, dim_t batch, dim_t bwd_batch_block, \
            const workspace_t &workspace, \
            const memory_storage_t &scratch_gates, \
            const memory_storage_t &scratch_cell, \
            const memory_storage_t &scratch_diff_states, \
            const memory_storage_t &bias, const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias) const

#define elemwise_sig_gru(f) \
    status_t f(const exec_ctx_t &ctx, dim_t dir, dim_t lay, dim_t iter, \
            dim_t dhc, dim_t batch, dim_t bwd_batch_block, \
            const workspace_t &workspace, \
            const memory_storage_t &scratch_gates, \
            const memory_storage_t &scratch_cell, \
            const memory_storage_t &scratch_diff_states, \
            const memory_storage_t &scratch_dhG1, \
            const memory_storage_t &bias, const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias, int part) const

#define cell_execution_sig(f) \
    status_t f(engine_t *engine, const exec_ctx_t &ctx, dim_t dir, dim_t lay, \
            dim_t iter, dim_t *wei_layer_offset, dim_t *wei_iter_offset, \
            const memory_storage_t &bias, const workspace_t &workspace, \
            const memory_storage_t &scratch_gates, \
            const memory_storage_t &scratch_cell, \
            const memory_storage_t &scratch_diff_states, \
            const memory_storage_t &scratch_dhG1, \
            const memory_storage_t &wei_layer, \
            const memory_storage_t &wei_iter, \
            const memory_storage_t &diff_weights_layer, \
            const memory_storage_t &diff_weights_iter, \
            const memory_storage_t &diff_bias, const memory_storage_t *scales, \
            const memory_storage_t *tm_scales) const

#define grid_execution_sig(f) \
    status_t f(engine_t *engine, const exec_ctx_t &ctx, \
            const memory_storage_t &bias, const workspace_t &workspace, \
            const memory_storage_t &scratch_gates, \
            const memory_storage_t &scratch_cell, \
            const memory_storage_t &scratch_diff_states, \
            const memory_storage_t &scratch_dhG1, \
            const memory_storage_t &wei_layer, \
            const memory_storage_t &wei_iter, \
            const memory_storage_t &diff_weights_layer, \
            const memory_storage_t &diff_weights_iter, \
            const memory_storage_t &diff_bias, const memory_storage_t *scales, \
            const memory_storage_t *tm_scales) const

#define gemm_sig(f) \
    status_t f(engine_t *engine, const exec_ctx_t &ctx, \
            const memory_storage_t &a, dim_t off_a, const memory_storage_t &b, \
            dim_t off_b, const memory_storage_t &c, dim_t off_c, \
            gemm_kind_t gemm_kind) const

#define weights_assign_sig(f) \
    void f(const rnn_utils::conf_t &rnn, const memory_desc_t *md, \
            dim_t *weights_, dim_t n_parts, const dim_t *gates_per_part, \
            const memory_storage_t &w_, dim_t ld, dim_t nld, \
            data_type_t wei_t) const

static inline bool is_ws_print_enabled() {
    return get_verbose_dev_mode(dnnl::impl::verbose_t::debuginfo) >= 5;
}

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace rnn_utils {

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum data_type_conf_t {
    all_f32,
    all_f16,
    all_bf16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8
};

enum ws_part_t {
    gates,
    states,
    c_states,
    diff_states,
    dhG1_gru,
    cell,
    grid,
    bias
};

struct ocl_conf_t {
    status_t create_generator(
            engine_t *engine, compute::compiled_bundle_t &generator) const {

        compute::kernel_ctx_t kernel_ctx;
        CHECK(init_kernel_ctx(kernel_ctx));
        auto status = compute::compiled_bundle_t::create(
                generator, engine, get_kernel_names(), kernel_ctx);
        return status;
    }
    const std::vector<const char *> &get_kernel_names() const {
        if (!is_ws_print_enabled()) {
            static const std::vector<const char *> names
                    = {"ref_rnn_bias_prepare", "ref_rnn_copy_init_layer",
                            "ref_rnn_copy_init_iter", "ref_rnn_copy_res_layer",
                            "ref_rnn_copy_res_iter", "ref_rnn_ws_set",
                            "ref_rnn_elemwise_fwd", "ref_rnn_elemwise_bwd"};
            return names;
        } else {
            static const std::vector<const char *> names
                    = {"ref_rnn_bias_prepare", "ref_rnn_copy_init_layer",
                            "ref_rnn_copy_init_iter", "ref_rnn_copy_res_layer",
                            "ref_rnn_copy_res_iter", "ref_rnn_ws_set",
                            "ref_rnn_elemwise_fwd", "ref_rnn_elemwise_bwd",
                            "ref_rnn_ws_print"};
            return names;
        }
    }

#if __cplusplus >= 202002L
    bool operator==(const ocl_conf_t &) const = default;
#endif
    serialized_t serialize() const {
        assert_trivially_serializable(ocl_conf_t);
        serialized_t s {};
        // Explicitly maintain zero padding to keep the implementation simple and
        // robust
        s.append(*this);
        return s;
    }

    static ocl_conf_t deserialize(const serialized_t &s) {
        ocl_conf_t t {};
        deserializer_t d(s);
        d.pop(t);
        return t;
    }

    status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

    int threads_per_eu = 0;
    int subgroup_size = 0;
    int cell_kind = 0;
    int activation_kind = 0;
    int direction_kind = 0;

    data_type_t src_dt = data_type::undef;
    data_type_t wei_dt = data_type::undef;
    data_type_t bia_dt = data_type::undef;
    data_type_t dst_dt = data_type::undef;
    data_type_t acc_dt = data_type::undef;
    data_type_t aux_dt = data_type::undef;
    data_type_t input_dt = data_type::undef;
    data_type_t output_dt = data_type::undef;
    data_type_t diff_dt = data_type::undef;

    struct inner_layouts_t {
#if __cplusplus >= 202002L
        bool operator==(const inner_layouts_t &) const = default;
#endif
        block_layout_t src_layer;
        block_layout_t src_iter;
        block_layout_t src_iter_c;
        block_layout_t weights_layer;
        block_layout_t weights_iter;
        block_layout_t bias;
        block_layout_t dst_layer;
        block_layout_t dst_iter;
        block_layout_t dst_iter_c;
        block_layout_t diff_src_layer;
        block_layout_t diff_src_iter;
        block_layout_t diff_src_iter_c;
        block_layout_t diff_weights_layer;
        block_layout_t diff_weights_iter;
        block_layout_t diff_bias;
        block_layout_t diff_dst_layer;
        block_layout_t diff_dst_iter;
        block_layout_t diff_dst_iter_c;
    };

    inner_layouts_t inner_layouts = {};

    int n_bias = 0;

    int wei_qparam_mask = 0;

    int elemwise_bwd_batch_block = 0;
    bool need_bias_atomic_reduce = false;
    bool with_bias = false;
    bool with_src_iter = false;
    bool with_src_iter_c = false;
    bool with_dst_iter = false;
    bool with_dst_iter_c = false;
    bool is_fwd = false;
    bool copy_bias = false;
    bool is_int8 = false;
    bool is_testmode = false;
    bool is_training = false;
    uint8_t pad0[1] = {};
};

struct conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    dim_t n_layer, n_iter, n_dir, n_gates, n_states;
    dim_t mb;
    dim_t slc, sic, dhc, dlc, wic;

    dim_t gates_ld, gates_nld, gates_ws_ld, arch_ld;

    dim_t n_parts_weights_layer, parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    dim_t n_parts_weights_iter, parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    dim_t n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    dim_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS],
            part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    // Size of packed data in bytes
    dim_t weights_layer_comp_offset, weights_layer_pack_size,
            weights_iter_comp_offset, weights_iter_pack_size;

    bool copy_bias;
    dim_t weights_layer_ld, weights_layer_nld;
    dim_t diff_weights_layer_ld, diff_weights_layer_nld;
    dim_t weights_iter_ld, weights_iter_nld;
    dim_t diff_weights_iter_ld, diff_weights_iter_nld;
    dim_t states_nld, states_ws_ld, scratch_diff_states_ld;
    bool is_fwd, is_training, is_lbr, is_int8, is_testmode, is_vanilla_gru;
    bool use_workspace;

    // for test mode (--skip_nonliner=true of benchdnn)
    float tm_cscale;
    dim_t tm_ngates;

    // Size of workspace for each tensor in bytes
    dim_t ws_states_cell_size, ws_c_states_cell_size, ws_gates_cell_size;
    dim_t ws_gates_size, ws_states_size, ws_c_states_size,
            scratch_diff_states_size, scratch_cell_size, scratch_dhG1_size,
            ws_grid_comp_size, ws_per_cell, ws_bias_size;

    dim_t ws_gates_offset;
    dim_t ws_states_offset;
    dim_t ws_grid_comp_offset;
    dim_t ws_c_state_offset;
    dim_t ws_bias_offset;

    bool merge_gemm_iter, merge_gemm_layer, use_gemm, use_layer_packed_gemm,
            use_iter_packed_gemm;

    // Element size of each workspace part in bytes
    dim_t ws_gates_elsz, ws_states_elsz, ws_grid_comp_elsz, ws_bias_elsz;

    dim_t scratch_gates_size;
    dim_t n_iter_scratch_gates;
    dim_t scratch_gates_elsz, scratch_gates_ld;

    data_type_t acc_data_type;
    dim_t acc_data_type_elsz;
    data_type_t aux_data_type;
    data_type_t input_data_type;
    data_type_t output_data_type;
    data_type_t dst_data_type;
    data_type_t diff_data_type;
};
bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);

dim_t get_good_ld(dim_t arch_ld, dim_t dim, dim_t sizeof_dt);
void init_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d, bool is_xe_hpc);
void init_test_mode(conf_t &rnn, const primitive_attr_t &attr);
void set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d);
dim_t set_workspace_offsets(const conf_t &rnn, dim_t &ws_gates_offset,
        dim_t &ws_h_state_offset, dim_t &ws_c_state_offset,
        dim_t &ws_grid_comp_onfset, dim_t &ws_bias_offset);
void set_gru_offsets_part2(const conf_t &rnn, dim_t iter, dim_t dir, dim_t lay,
        data_type_t src_t, dim_t *wei_iter_off_ptr,
        const dim_t &ws_states_offset_, dim_t &cell_wei_iter_offset,
        dim_t &cell_scratch_offset, dim_t &cell_ws_iter_offset);
void set_offsets_fwd_gemm(const conf_t &rnn, dim_t dir, dim_t lay,
        data_type_t src_t, dim_t *wei_layer_off_ptr,
        const dim_t &ws_states_offset_, dim_t &grid_ws_lay_offset,
        dim_t &grid_wei_lay_offset, dim_t &grid_ws_iter_offset);
void set_offsets_fwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir, dim_t lay,
        data_type_t src_t, dim_t *wei_iter_off_ptr,
        const dim_t &ws_states_offset_, dim_t &cell_ws_iter_offset,
        dim_t &cell_ws_lay_offset, dim_t &cell_scratch_offset,
        dim_t &cell_wei_iter_offset);
void set_offsets_bwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir, dim_t lay,
        dim_t &cell_diff_wei_iter_off, dim_t &cell_diff_wei_lay_off,
        dim_t &cell_scr_diff_lay_off, dim_t &cell_scr_diff_iter_off);
void set_offsets_bwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir, dim_t lay,
        dim_t &cell_diff_wei_iter_off, dim_t &cell_diff_wei_lay_off,
        dim_t &cell_scr_diff_lay_off, dim_t &cell_scr_diff_iter_off,
        dim_t &cell_diff_wei_iter_off2);
void set_offsets_bwd_gemm(const conf_t &rnn, dim_t iter, dim_t dir, dim_t lay,
        dim_t &cell_diff_wei_iter_off, dim_t &cell_diff_wei_lay_off,
        dim_t &cell_scr_diff_lay_off);
dim_t get_workspace_size(const conf_t &rnn);
status_t set_expected_desc(
        conf_t &rnn, memory_desc_t &weights_md, bool is_iter);
status_t set_good_strides(
        dim_t ld_, memory_desc_t &weights_md, format_tag_t tag);
memory_storage_t &get_storage(const std::unique_ptr<memory_storage_t> &storage);

inline void append_strides(compute::kernel_arg_list_t &arg_list,
        const dim_t offs[4][MAX_NDIMS], int ocl_nparams, int ndims) {
    for (int d = 0; d < ocl_nparams; d++) {
        arg_list.append((d < ndims) ? (cl_int)offs[1][d] : 0);
    }
}

inline void append_strides(compute::kernel_arg_list_t &arg_list,
        const strides_t &strides, int ocl_nparams) {
    for (int d = 0; d < ocl_nparams; d++) {
        assert(strides[d] < INT_MAX);
        arg_list.append((cl_int)strides[d]);
    }
}

} // namespace rnn_utils

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
