/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_RNN_RNN_UTILS_HPP
#define GPU_INTEL_OCL_RNN_RNN_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "gpu/intel/compute/compute_engine.hpp"
#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/serialization.hpp"

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
            const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::sub_buffer_t &scratch_gates, \
            const rnn_utils::sub_buffer_t &scratch_diff_gates, \
            const rnn_utils::sub_buffer_t &scratch_diff_states, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_s1, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_iter, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_iter_s1, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_layer, \
            dim_t diff_states_layer_ld, const memory_storage_t *scales, \
            const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias) const

#define elemwise_sig_gru_lbr(f) \
    status_t f(const exec_ctx_t &ctx, dim_t dir, dim_t lay, dim_t iter, \
            dim_t dhc, dim_t batch, dim_t bwd_batch_block, \
            const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::sub_buffer_t &scratch_gates, \
            const rnn_utils::sub_buffer_t &scratch_diff_gates, \
            const memory_storage_t &scratch_cell, \
            const rnn_utils::sub_buffer_t &scratch_diff_states, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_iter, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_layer, \
            dim_t diff_states_layer_ld, const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias) const

#define elemwise_sig_gru(f) \
    status_t f(const exec_ctx_t &ctx, dim_t dir, dim_t lay, dim_t iter, \
            dim_t dhc, dim_t batch, dim_t bwd_batch_block, \
            const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::sub_buffer_t &scratch_gates, \
            const rnn_utils::sub_buffer_t &scratch_diff_gates, \
            const memory_storage_t &scratch_cell, \
            const rnn_utils::sub_buffer_t &scratch_diff_states, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_iter, \
            const rnn_utils::sub_buffer_t &scratch_diff_states_layer, \
            dim_t diff_states_layer_ld, \
            const rnn_utils::sub_buffer_t &scratch_dhG1, \
            const memory_storage_t *tm_scales, \
            const memory_storage_t &diff_bias, int part) const

#define cell_execution_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, dim_t dir, \
            dim_t lay, dim_t iter, const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::scratch_t &scratch, \
            const memory_storage_t &diff_bias, const memory_storage_t *scales, \
            const memory_storage_t *tm_scales) const

#define grid_execution_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, \
            const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::scratch_t &scratch, \
            const memory_storage_t &diff_bias, const memory_storage_t *scales, \
            const memory_storage_t *tm_scales) const

#define gemm_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, \
            const rnn_utils::sub_buffer_t &a, \
            const rnn_utils::sub_buffer_t &b, \
            const rnn_utils::sub_buffer_t &c, gemm_kind_t gemm_kind) const

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
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

namespace kernel_id {
constexpr size_t bias_prepare = 0;
constexpr size_t copy_init_layer = 1;
constexpr size_t copy_init_iter = 2;
constexpr size_t copy_res_layer = 3;
constexpr size_t copy_res_iter = 4;
constexpr size_t elemwise_fwd = 5;
constexpr size_t elemwise_bwd = 6;
constexpr size_t cell_fwd = 7;
} // namespace kernel_id

struct ocl_conf_t {
    status_t create_generator(const compute::compute_engine_t &engine,
            compute::kernel_bundle_t &bundle) const {

        compute::kernel_ctx_t kernel_ctx;
        CHECK(init_kernel_ctx(kernel_ctx));
        return engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
    }
    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> names = {"rnn_bias_prepare",
                "simple_rnn_copy_init_layer", "simple_rnn_copy_init_iter",
                "simple_rnn_copy_res_layer", "simple_rnn_copy_res_iter",
                "simple_rnn_elemwise_fwd", "simple_rnn_elemwise_bwd",
                "simple_rnn_cell_fwd"};
        return names;
    }

#if __cplusplus >= 202002L
    bool operator==(const ocl_conf_t &) const = default;
#endif
    serialized_t serialize() const {
        assert_trivially_serializable(ocl_conf_t);
        return serialized_t(*this);
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
    data_type_t src_c_dt = data_type::undef;
    data_type_t wei_dt = data_type::undef;
    data_type_t bia_dt = data_type::undef;
    data_type_t dst_dt = data_type::undef;
    data_type_t dst_c_dt = data_type::undef;
    data_type_t acc_dt = data_type::undef;
    data_type_t aux_dt = data_type::undef;
    data_type_t ws_state_dt = data_type::undef;
    data_type_t input_dt = data_type::undef;
    data_type_t output_dt = data_type::undef;
    data_type_t diff_dt = data_type::undef;
    uint8_t pad[4] = {};

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
    bool recompute_gates = false;
    bool copy_src_layer = false;
    bool copy_diff_dst_layer = false;
    bool copy_diff_src_layer;
    bool deterministic = false;
    struct comp_conf_t {
        bool is_enabled = false;
        bool compute_gemm_layer = false;
        bool gemm_layer_k_tail = false;
        bool compute_gemm_iter = false;
        bool gemm_iter_k_tail = false;
        bool dhc_tail = false;
        bool mb_tail = false;
        bool enable_iter_block = false;
        int dhc_thr = 0;
        int dhc_tg = 0;
        int mb_thr = 0;
        int mb_tg = 0;
#if __cplusplus >= 202002L
        bool operator==(const comp_conf_t &) const = default;
#endif
    } cell_comp;
};

struct conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    dim_t n_layer, n_iter, n_dir, n_gates, n_states;
    dim_t mb;
    dim_t slc, sic, dhc, dlc, wic;

    dim_t gates_ld, gates_ws_ld, arch_ld;

    dim_t n_parts_weights_layer, parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    dim_t n_parts_weights_iter, parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    dim_t n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    dim_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS],
            part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];

    // Size of packed data in bytes
    dim_t weights_layer_comp_offset, weights_layer_pack_size,
            weights_iter_comp_offset, weights_iter_pack_size;

    struct {
        bool gemm_layer;
        bool gemm_iter;
    } cell_fusion;

    dim_t dhc_loop;
    dim_t iter_loop;

    bool copy_bias;
    dim_t states_ws_ld, scratch_diff_states_ld;
    bool is_fwd, is_training, is_lbr, is_int8, is_testmode, is_vanilla_gru;
    bool use_workspace;
    bool recompute_gates;
    bool copy_src_layer;
    bool copy_diff_dst_layer;
    bool copy_diff_src_layer;

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

    dim_t n_iter_scratch_gates;
    dim_t scratch_gates_size, scratch_gates_elsz, scratch_gates_ld;
    dim_t scratch_diff_gates_size, scratch_diff_gates_elsz,
            scratch_diff_gates_ld;

    data_type_t acc_data_type;
    dim_t acc_data_type_elsz;
    data_type_t aux_data_type;
    data_type_t input_data_type;
    data_type_t output_data_type;
    data_type_t src_data_type;
    data_type_t dst_data_type;
    data_type_t diff_data_type;
    data_type_t wei_layer_type;
    data_type_t wei_iter_type;
    data_type_t bias_data_type;
};
bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);

dim_t get_good_ld(
        dim_t arch_ld, dim_t dim, dim_t sizeof_dt, bool ignore_assoc = false);
void init_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &bias_d, data_type_t acc_data_type,
        const compute::device_info_t &device_info);
void init_test_mode(conf_t &rnn, const primitive_attr_t &attr);
void set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &diff_src_layer_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d);
dim_t set_workspace_offsets(const conf_t &rnn, dim_t &ws_gates_offset,
        dim_t &ws_h_state_offset, dim_t &ws_c_state_offset,
        dim_t &ws_grid_comp_onfset, dim_t &ws_bias_offset);
dim_t get_workspace_size(const conf_t &rnn);
status_t set_weights_desc(memory_desc_t &weights_md, const conf_t &rnn);
status_t set_good_strides(
        dim_t ld_, memory_desc_t &weights_md, format_tag_t tag);
const memory_storage_t &get_storage(const memory_storage_t *storage);
const memory_storage_t &get_storage(
        const std::unique_ptr<memory_storage_t> &storage);

struct sub_buffer_t {
    static constexpr dim_t unset = 0;

    sub_buffer_t() : buffer_(nullptr) {}
    sub_buffer_t(const memory_storage_t &buffer, dim_t offset = 0,
            dim_t size = unset)
        : buffer_(buffer.is_null() ? nullptr : buffer.clone()) {
        if (buffer_) buffer_->set_offset(static_cast<size_t>(offset));
    }

    sub_buffer_t(const sub_buffer_t &buffer, dim_t offset = 0)
        : buffer_(buffer ? buffer.buffer_->clone() : nullptr) {
        if (buffer_) buffer_->set_offset(buffer.offset() + offset);
    }

    ~sub_buffer_t() = default;

    // Aligns with memory_storage_t
    sub_buffer_t &operator=(const sub_buffer_t &) = delete;

    operator bool() const { return buffer_ != nullptr && !buffer_->is_null(); }

    const memory_storage_t *get() const { return buffer_.get(); }
    const memory_storage_t &get_storage() const {
        return buffer_ ? *buffer_ : memory_storage_t::empty_storage();
    }
    size_t offset(data_type_t dt = data_type::undef) const {
        if (buffer_ == nullptr) return 0;
        if (dt == data_type::undef) return buffer_->offset();
        gpu_assert(buffer_->offset() % types::data_type_size(dt) == 0);
        return buffer_->offset() / types::data_type_size(dt);
    }

private:
    std::unique_ptr<memory_storage_t> buffer_;
};

struct data_helper_t {
    static uint32_t type_size(data_type_t d) {
        return into<uint32_t>(types::data_type_size(d));
    }
};

struct user_data_t : public data_helper_t {
    using mst = memory_storage_t;
    user_data_t(const mst &src_layer, const mst &wei_layer, const mst &wei_iter,
            const mst &bias, const mst &diff_src_layer,
            const mst &diff_dst_layer, const mst &diff_wei_layer,
            const mst &diff_wei_iter, const conf_t &conf,
            const rnn_offsets_t &offsets)
        : src_layer_(src_layer)
        , wei_layer_(wei_layer)
        , wei_iter_(wei_iter)
        , bias_(bias)
        , diff_src_layer_(diff_src_layer)
        , diff_dst_layer_(diff_dst_layer)
        , diff_wei_layer_(diff_wei_layer)
        , diff_wei_iter_(diff_wei_iter)
        , conf_(conf)
        , offsets_(offsets) {
        // The packed restriction could be removed by using batched GEMM with
        // appropriate strides.
        gpu_assert(IMPLICATION(conf_.merge_gemm_layer && !conf_.copy_src_layer
                        && conf.n_iter > 1 && conf.mb > 1,
                offsets_.src_layer[0] == offsets_.src_layer[1] * conf_.mb
                        && conf_.exec_dir == l2r))
                << "[ERROR]: GEMM dimensions must be packed in order to "
                   "perform merge_gemm_layer";

        gpu_assert(IMPLICATION(!conf.copy_src_layer && conf.n_iter > 1,
                (offsets_.src_layer[0] * type_size(conf_.src_data_type)) % 8
                        == 0))
                << "[ERROR]: GEMM interface assumes inputs buffers are well "
                   "aligned";
        gpu_assert(offsets_.src_layer[0] < INT_MAX
                && offsets_.src_layer[1] < INT_MAX
                && offsets_.src_layer[2] < INT_MAX)
                << "[UNIMPLEMENTED]: src offsets larger than INT_MAX are not "
                   "currently supported in rnn_grid.cl";

        if (!conf.is_fwd) {
            gpu_assert(IMPLICATION(!conf.copy_diff_dst_layer && conf.n_iter > 1,
                    (offsets_.diff_dst_layer[0]
                            * type_size(conf_.diff_data_type))
                                    % 8
                            == 0))
                    << "[ERROR]: GEMM interface assumes inputs buffers are "
                       "well aligned";
            gpu_assert(offsets_.diff_dst_layer[0] < INT_MAX
                    && offsets_.diff_dst_layer[1] < INT_MAX
                    && offsets_.diff_dst_layer[2] < INT_MAX)
                    << "[UNIMPLEMENTED]: diff_dst offsets larger than INT_MAX "
                       "are not currently supported in rnn_grid.cl";
        }
    }

    dim_t normalized_iter(dim_t dir, dim_t iter_) const {
        if (conf_.exec_dir == l2r)
            return iter_;
        else if (conf_.exec_dir == r2l)
            return conf_.n_iter - iter_ - 1;
        else if (dir == 1)
            return conf_.n_iter - iter_ - 1;
        else
            return iter_;
    }

    const mst &src_layer() const { return src_layer_; }
    sub_buffer_t src_layer(
            dim_t dir, dim_t iter_, bool all_iter = false) const {
        auto iter = normalized_iter(dir, iter_);

        // src_layer dimension order: iter, mini-batch, channel
        const auto iter_stride
                = offsets_.src_layer[0] * type_size(conf_.src_data_type);
        dim_t offset = iter * iter_stride;
        auto cell_size = iter_stride;
        auto n_cells = all_iter ? conf_.n_iter - iter : 1;
        return {src_layer(), offset, cell_size * n_cells};
    }
    strides_t<3> src_layer_strides(dim_t dir) const {
        auto ret = offsets_.src_layer;

        // Use negative iterations stride for backwards iteration
        ret[0] *= (0 == normalized_iter(dir, 0)) ? 1 : -1;
        return ret;
    }

    const mst &wei_layer() const { return wei_layer_; }
    sub_buffer_t wei_layer(
            dim_t lay, dim_t dir, bool is_multi_cell = false) const {

        // wei_layer dimension order: layer, dir, src c, gate, dst c
        dim_t t = type_size(conf_.wei_layer_type);
        dim_t lay_stride = offsets_.weights_layer[0];
        dim_t dir_stride = offsets_.weights_layer[1];
        dim_t offset = (lay * lay_stride + dir * dir_stride) * t;
        dim_t cell_size = dir_stride * t;

        if (is_multi_cell) return {wei_layer(), offset};

        return {wei_layer(), offset, cell_size};
    }

    const mst &wei_iter() const { return wei_iter_; }
    sub_buffer_t wei_iter(dim_t lay, dim_t dir) const {
        // wei_iter dimension order: layer, dir, src c, gate, dst c
        dim_t t = type_size(conf_.wei_iter_type);
        dim_t lay_stride = offsets_.weights_iter[0];
        dim_t dir_stride = offsets_.weights_iter[1];
        dim_t offset = (lay * lay_stride + dir * dir_stride) * t;
        dim_t cell_size = dir_stride * t;
        return {wei_iter(), offset, cell_size};
    }

    const mst &diff_wei_layer() const { return diff_wei_layer_; }
    sub_buffer_t diff_wei_layer(
            dim_t lay, dim_t dir, bool is_multi_cell = false) const {

        // diff_wei_layer dimension order: layer, dir, src c, gate, dst c
        dim_t t = sizeof(float);
        dim_t lay_stride = offsets_.diff_weights_layer[0];
        dim_t dir_stride = offsets_.diff_weights_layer[1];
        dim_t offset = (lay * lay_stride + dir * dir_stride) * t;
        dim_t cell_size = dir_stride * t;

        if (is_multi_cell) return {diff_wei_layer(), offset};

        return {diff_wei_layer(), offset, cell_size};
    }

    const mst &diff_wei_iter() const { return diff_wei_iter_; }
    sub_buffer_t diff_wei_iter(
            dim_t lay, dim_t dir, bool is_multi_cell = false) const {
        // diff_wei_iter dimension order: layer, dir, src c, gate, dst c
        dim_t t = sizeof(float);
        dim_t lay_stride = offsets_.diff_weights_iter[0];
        dim_t dir_stride = offsets_.diff_weights_iter[1];
        dim_t offset = (lay * lay_stride + dir * dir_stride) * t;
        dim_t cell_size = dir_stride * t;
        if (is_multi_cell) return {diff_wei_iter(), offset};
        return {diff_wei_iter(), offset, cell_size};
    }

    const mst &bias() const { return bias_; }
    sub_buffer_t bias(dim_t lay, dim_t dir) const {
        if (bias().data_handle() == nullptr) return {};

        // bia dimension order: lay, dir, gates, dhc
        auto t_size = type_size(conf_.bias_data_type);
        auto layer_stride = offsets_.bias[0] * t_size;
        auto dir_stride = offsets_.bias[1] * t_size;
        auto cell_size = dir_stride;
        auto offset = layer_stride * lay + dir_stride * dir;
        return {bias(), offset, cell_size};
    }

    const mst &diff_src_layer() const { return diff_src_layer_; }
    sub_buffer_t diff_src_layer(
            dim_t dir, dim_t iter_, bool all_iter = false) const {
        auto iter = normalized_iter(dir, iter_);

        // diff_src_layer dimension order: iter, mini-batch, channel
        const auto iter_stride
                = offsets_.diff_src_layer[0] * type_size(conf_.diff_data_type);
        auto offset = iter * iter_stride;
        auto cell_size = iter_stride;
        auto n_cells = all_iter ? conf_.n_iter - iter : 1;
        return {diff_src_layer(), offset, cell_size * n_cells};
    }

    const mst &diff_dst_layer() const { return diff_dst_layer_; }
    sub_buffer_t diff_dst_layer(
            dim_t dir, dim_t iter_, bool all_iter = false) const {
        auto iter = normalized_iter(dir, iter_);

        // diff_dst_layer dimension order: iter, mini-batch, channel
        const auto iter_stride
                = offsets_.diff_dst_layer[0] * type_size(conf_.diff_data_type);
        const auto dir_offset
                = (conf_.exec_dir == execution_direction_t::bi_concat && dir)
                ? offsets_.diff_dst_layer[1] * type_size(conf_.diff_data_type)
                        / 2
                : 0;
        auto offset = iter * iter_stride + dir_offset;
        auto cell_size = iter_stride;
        auto n_cells = all_iter ? conf_.n_iter - iter : 1;
        return {diff_dst_layer(), offset, cell_size * n_cells};
    }

    const mst &src_layer_;
    const mst &wei_layer_;
    const mst &wei_iter_;
    const mst &bias_;
    const mst &diff_src_layer_;
    const mst &diff_dst_layer_;
    const mst &diff_wei_layer_;
    const mst &diff_wei_iter_;
    const conf_t &conf_;
    const rnn_offsets_t &offsets_;
};

struct workspace_t : public data_helper_t {
    using mst = memory_storage_t;
    workspace_t(const mst &ws, const conf_t &conf)
        : ws_(ws)
        , conf_(conf)
        , gates_(conf.ws_gates_size > 0 ? ws.get_sub_storage(
                         conf.ws_gates_offset, conf.ws_gates_size)
                                        : nullptr)
        , gates_strides_ {0}
        , states_(conf.ws_states_size > 0 ? ws.get_sub_storage(
                          conf.ws_states_offset, conf.ws_states_size)
                                          : nullptr)
        , states_strides_ {0}
        , c_states_(conf.ws_c_states_size > 0 ? ws.get_sub_storage(
                            conf.ws_c_state_offset, conf.ws_c_states_size)
                                              : nullptr)
        , bias_(conf.ws_bias_size > 0 ? ws.get_sub_storage(
                        conf.ws_bias_offset, conf.ws_bias_size)
                                      : nullptr)
        , grid_comp_(conf.ws_grid_comp_size > 0 ? ws.get_sub_storage(
                             conf.ws_grid_comp_offset, conf.ws_grid_comp_size)
                                                : nullptr) {
        if (gates_) {
            const dim_t n_b = conf_.mb;
            const dim_t n_tb = conf_.n_iter * n_b;
            const dim_t n_dtb = conf_.n_dir * n_tb;
            gates_strides_
                    = {n_dtb * conf_.gates_ws_ld, n_tb * conf_.gates_ws_ld,
                            n_b * conf_.gates_ws_ld, conf_.gates_ws_ld};
        }
        if (states_) {
            const dim_t n_b = conf_.mb;
            const dim_t n_tb = (conf_.n_iter + 1) * n_b;
            const dim_t n_dtb = conf_.n_dir * n_tb;
            states_strides_
                    = {n_dtb * conf_.states_ws_ld, n_tb * conf_.states_ws_ld,
                            n_b * conf_.states_ws_ld, conf_.states_ws_ld};
        }
    }

    template <size_t ndims>
    static dim_t get_offset(const strides_t<ndims> &strides,
            const std::array<dim_t, ndims> &dims) {
        dim_t offset = 0;
        for (size_t i = 0; i < ndims; i++) {
            offset += strides[i] * dims[i];
        }
        return offset;
    }

    dim_t calc_off_ws_state(
            dim_t i0_, dim_t i1, dim_t i2_, dim_t i3, dim_t i4) const {
        // Logical index into workspace grid
        auto i0 = conf_.copy_src_layer ? i0_ + 1 : i0_;
        auto i0_size = conf_.copy_src_layer ? conf_.n_layer + 1 : conf_.n_layer;
        auto i2 = i2_ + 1;

        gpu_assert(i0 >= 0) << "Logical index must be larger than 0";

        MAYBE_UNUSED(i0_size);
        return OFF5(i0, i0_size, i1, conf_.n_dir, i2, conf_.n_iter + 1, i3,
                conf_.mb, i4, conf_.states_ws_ld);
    }

    dim_t calc_off_ws_c_state(
            dim_t i0_, dim_t i1, dim_t i2_, dim_t i3, dim_t i4) const {
        // Logical index into workspace grid
        auto i0 = i0_;
        auto i0_size = conf_.n_layer;
        auto i2 = i2_ + 1;

        gpu_assert(i0 >= 0) << "Logical index must be larger than 0";

        MAYBE_UNUSED(i0_size);
        return OFF5(i0, i0_size, i1, conf_.n_dir, i2, conf_.n_iter + 1, i3,
                conf_.mb, i4, conf_.states_ws_ld);
    }

    dim_t calc_off_ws_grid_offset(
            dim_t i0, dim_t i1, dim_t i2, dim_t i3, dim_t i4) const {
        return OFF5(i0, conf_.n_layer, i1, conf_.n_dir, i2, conf_.n_iter, i3,
                conf_.mb, i4, conf_.dhc);
    }

    const mst &ws() const { return ws_; }
    const mst &gates() const { return get_storage(gates_); }
    const mst &states() const { return get_storage(states_); }

    sub_buffer_t states(dim_t layer, dim_t dir, dim_t time) const {
        if (!states_) return {};

        auto i0 = conf_.copy_src_layer ? layer + 1 : layer;
        auto i2 = time + 1;
        auto off_ = get_offset(states_strides(), {i0, dir, i2, 0})
                * conf_.ws_states_elsz;
        return {states(), off_, conf_.ws_states_cell_size};
    }

    const strides_t<4> &states_strides() const { return states_strides_; }

    sub_buffer_t states_range(dim_t layer_start, dim_t layer_end,
            dim_t dir_start, dim_t dir_end, dim_t time_start,
            dim_t time_end) const {
        if (!states_) return {};
        auto off_start
                = calc_off_ws_state(layer_start, dir_start, time_start, 0, 0)
                * conf_.ws_states_elsz;
        auto off_end = calc_off_ws_state(layer_end, dir_end, time_end, 0, 0)
                * conf_.ws_states_elsz;
        return {states(), off_start, off_end - off_start};
    }

    sub_buffer_t c_states(dim_t layer, dim_t dir, dim_t time) const {
        if (!c_states_) return {};
        // conf_.aux_data_type is float for all datatypes except f16
        // so can be used for lstm_elemwise_u8s8 case as well
        auto off = calc_off_ws_c_state(layer, dir, time, 0, 0)
                * type_size(conf_.aux_data_type);
        return {c_states(), off, conf_.ws_c_states_cell_size};
    }

    sub_buffer_t gates(dim_t layer, dim_t dir, dim_t time, dim_t mb = 0) const {
        if (!gates_) return {};

        auto off = get_offset(gates_strides(), {layer, dir, time, mb})
                * type_size(conf_.aux_data_type);
        return {gates(), off, conf_.ws_gates_cell_size};
    }
    const strides_t<4> &gates_strides() const { return gates_strides_; }

    sub_buffer_t grid_comp(dim_t layer, dim_t dir, dim_t time) const {
        if (!grid_comp_) return {};

        auto off = calc_off_ws_grid_offset(layer, dir, time, 0, 0)
                * type_size(conf_.aux_data_type);
        return {grid_comp(), off, conf_.ws_per_cell};
    }

    const mst &c_states() const { return get_storage(c_states_); }
    const mst &bias() const { return get_storage(bias_); }
    const mst &grid_comp() const { return get_storage(grid_comp_); }

private:
    const mst &ws_;
    const conf_t &conf_;
    std::unique_ptr<mst> gates_;
    strides_t<4> gates_strides_;
    std::unique_ptr<mst> states_;
    strides_t<4> states_strides_;
    std::unique_ptr<mst> c_states_;
    std::unique_ptr<mst> bias_;
    std::unique_ptr<mst> grid_comp_;
};

struct scratch_t : public data_helper_t {
    using mst = memory_storage_t;

    enum {
        key_gemm_iter_fwd = memory_tracking::names::key_nested_multiple,
        key_gemm_iter_fwd_2,
        key_gemm_layer_fwd,
        key_gemm_layer_fwd_src,
        key_gemm_iter_bwd,
        key_gemm_iter_bwd_2,
        key_gemm_layer_bwd,
        key_gemm_layer_bwd_src,
        key_gemm_diff_wei_layer,
        key_gemm_diff_wei_layer_src,
        key_gemm_diff_wei_iter,
        key_gemm_diff_wei_iter_2,
    };

    scratch_t(const conf_t &conf, const memory_tracking::grantor_t &scratchpad)
        : conf_(conf) {
        using namespace memory_tracking::names;
        gates_ = scratchpad.get_memory_storage(key_rnn_gates);
        diff_gates_ = scratchpad.get_memory_storage(key_rnn_diff_gates);
        cell_ = scratchpad.get_memory_storage(key_rnn_cell);
        diff_states_ = scratchpad.get_memory_storage(key_rnn_diff_states);
        diff_ht_ = scratchpad.get_memory_storage(key_rnn_diff_ht);
    }

    struct gemm_pds_t {
        const primitive_desc_t *iter_fwd_pd;
        const primitive_desc_t *iter_fwd_2_pd;
        const primitive_desc_t *layer_fwd_pd;
        const primitive_desc_t *layer_fwd_src_pd;
        const primitive_desc_t *iter_bwd_pd;
        const primitive_desc_t *iter_bwd_2_pd;
        const primitive_desc_t *layer_bwd_pd;
        const primitive_desc_t *layer_bwd_src_pd;
        const primitive_desc_t *diff_wei_layer_pd;
        const primitive_desc_t *diff_wei_layer_src_pd;
        const primitive_desc_t *diff_wei_iter_pd;
        const primitive_desc_t *diff_wei_iter_2_pd;
    };

    static void book(memory_tracking::registrar_t &scratchpad,
            const conf_t &rnn_conf, const gemm_pds_t &gemms) {
        using namespace memory_tracking::names;
        if (rnn_conf.scratch_gates_size > 0)
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1,
                    OCL_BUFFER_ALIGNMENT, 4096);
        scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1,
                OCL_BUFFER_ALIGNMENT, 4096);
        scratchpad.book(key_rnn_diff_states, rnn_conf.scratch_diff_states_size,
                1, OCL_BUFFER_ALIGNMENT, 4096);
        scratchpad.book(key_rnn_diff_ht, rnn_conf.scratch_dhG1_size, 1,
                OCL_BUFFER_ALIGNMENT, 4096);
        // book scratchpad for nested primitives
        if (gemms.layer_fwd_pd) {
            scratchpad.book(key_gemm_layer_fwd,
                    gemms.layer_fwd_pd->scratchpad_registry());
        }
        if (gemms.layer_fwd_src_pd) {
            scratchpad.book(key_gemm_layer_fwd_src,
                    gemms.layer_fwd_src_pd->scratchpad_registry());
        }
        if (gemms.iter_fwd_pd) {
            scratchpad.book(key_gemm_iter_fwd,
                    gemms.iter_fwd_pd->scratchpad_registry());
        }

        if (rnn_conf.is_fwd) {
            if (rnn_conf.is_vanilla_gru)
                scratchpad.book(key_gemm_iter_fwd_2,
                        gemms.iter_fwd_2_pd->scratchpad_registry());
        } else {
            scratchpad.book(key_rnn_diff_gates,
                    rnn_conf.scratch_diff_gates_size, 1, OCL_BUFFER_ALIGNMENT,
                    4096);
            scratchpad.book(key_gemm_iter_bwd,
                    gemms.iter_bwd_pd->scratchpad_registry());
            scratchpad.book(key_gemm_layer_bwd,
                    gemms.layer_bwd_pd->scratchpad_registry());
            if (gemms.layer_bwd_src_pd)
                scratchpad.book(key_gemm_layer_bwd_src,
                        gemms.layer_bwd_src_pd->scratchpad_registry());
            scratchpad.book(key_gemm_diff_wei_layer,
                    gemms.diff_wei_layer_pd->scratchpad_registry());
            if (gemms.diff_wei_layer_src_pd)
                scratchpad.book(key_gemm_diff_wei_layer_src,
                        gemms.diff_wei_layer_src_pd->scratchpad_registry());
            scratchpad.book(key_gemm_diff_wei_iter,
                    gemms.diff_wei_iter_pd->scratchpad_registry());
            if (rnn_conf.is_vanilla_gru) {
                scratchpad.book(key_gemm_iter_bwd_2,
                        gemms.iter_bwd_2_pd->scratchpad_registry());
                scratchpad.book(key_gemm_diff_wei_iter_2,
                        gemms.diff_wei_iter_2_pd->scratchpad_registry());
            }
        }
    }

    dim_t calc_off_gates(dim_t iter) const {
        return conf_.n_iter_scratch_gates != 1
                ? iter * conf_.mb * conf_.scratch_gates_ld
                : 0;
    };

    const mst *gates() const {
        // Reuse diff_gates_ when possible to reduce memory consumption
        gpu_assert(gates_ || diff_gates_);
        return (conf_.is_fwd || conf_.recompute_gates)
                ? (gates_ ? gates_.get() : diff_gates_.get())
                : nullptr;
    }
    sub_buffer_t gates(dim_t iter) const {
        auto g = gates();
        if (g == nullptr) return {};

        auto off = calc_off_gates(iter) * conf_.scratch_gates_elsz;
        auto cell_size
                = conf_.mb * conf_.scratch_gates_ld * conf_.scratch_gates_elsz;
        return {*g, off, cell_size};
    }

    dim_t calc_off_diff_gates(dim_t iter) const {
        return conf_.n_iter_scratch_gates != 1
                ? iter * conf_.mb * conf_.scratch_diff_gates_ld
                : 0;
    };
    const mst *diff_gates() const { return diff_gates_.get(); }

    sub_buffer_t diff_gates(dim_t iter) const {
        auto g = diff_gates();
        if (g == nullptr) return {};

        auto off = calc_off_diff_gates(iter) * conf_.scratch_diff_gates_elsz;
        auto cell_size = conf_.mb * conf_.scratch_diff_gates_ld
                * conf_.scratch_diff_gates_elsz;
        return {*g, off, cell_size};
    }

    const mst *cell() const { return cell_.get(); }

    dim_t calc_off_diff_state(
            dim_t i0, dim_t i1, dim_t i2, dim_t i3, dim_t i4, dim_t i5) const {
        // Logical index into workspace grid
        bool have_result_layer = conf_.copy_diff_src_layer || conf_.n_layer > 1;
        auto i0_size = conf_.n_layer - 1 + conf_.copy_diff_dst_layer
                + have_result_layer;
        if (!have_result_layer) { i0--; }
        auto i2_size = conf_.copy_diff_dst_layer || conf_.copy_diff_src_layer
                        || conf_.n_layer != 1
                ? conf_.n_states + 1
                : conf_.n_states;
        auto i3_size = conf_.n_iter + 1;
        if (i0_size <= 1) {
            if (i0_size <= 0) {
                i3_size = 2;
                i3 %= i3_size;
            }
            return OFF5(i1, conf_.n_dir, i2, i2_size, i3, i3_size, i4, conf_.mb,
                    i5, conf_.scratch_diff_states_ld);
        }
        gpu_assert(i0 < i0_size && i0 >= 0)
                << "Logical index " << i0 << " should be in [0, " << i0_size
                << ")";
        gpu_assert(i2 < i2_size && i2 >= 0)
                << "Logical index " << i2 << " should be in [0, " << i2_size
                << ")" << i2_size;
        MAYBE_UNUSED(i0_size);

        return OFF6(i0, i0_size, i1, conf_.n_dir, i2, i2_size, i3, i3_size, i4,
                conf_.mb, i5, conf_.scratch_diff_states_ld);
    }

    const mst *diff_states() const { return diff_states_.get(); }

    sub_buffer_t diff_states(
            dim_t layer, dim_t dir, dim_t state, dim_t iter) const {
        int aux_elsz = type_size(conf_.aux_data_type);

        if (!diff_states_) return {};
        auto off
                = calc_off_diff_state(layer, dir, state, iter, 0, 0) * aux_elsz;
        auto cell_size = conf_.mb * conf_.scratch_diff_states_ld * aux_elsz;
        return {*diff_states(), off, cell_size};
    }

    const mst *diff_ht() const { return diff_ht_.get(); }

private:
    const conf_t &conf_;

    std::unique_ptr<mst> gates_;
    std::unique_ptr<mst> diff_gates_;
    std::unique_ptr<mst> cell_;
    std::unique_ptr<mst> diff_states_;
    std::unique_ptr<mst> diff_ht_;
};

struct arg_list_t {
    template <typename T>
    void append(const T &t) {
        args.append(t);
    }
    void append(const rnn_utils::sub_buffer_t &buffer, data_type_t dt) {
        args.append(buffer.get_storage());
        args.append(into<dim_t>(buffer.offset(dt)));
    }
    compute::kernel_arg_list_t args;
};

static_assert(sizeof(arg_list_t) == sizeof(compute::kernel_arg_list_t),
        "The arg_list_t is a helper for injecting RNN specific helper "
        "functions structures into kernel_args_list_t.");

} // namespace rnn_utils

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
