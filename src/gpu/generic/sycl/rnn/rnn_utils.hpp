/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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
#include "common/memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive_desc.hpp"

#define OFF5(i0, d0, i1, d1, i2, d2, i3, d3, i4, d4) \
    (((((i0) * (d1) + (i1)) * (d2) + (i2)) * (d3) + (i3)) * (d4) + (i4))

#define cell_execution_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, dim_t dir, \
            dim_t lay, dim_t iter, const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::scratch_t &scratch) const

#define grid_execution_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, \
            const rnn_utils::user_data_t &user_data, \
            const rnn_utils::workspace_t &workspace, \
            const rnn_utils::scratch_t &scratch) const

#define gemm_sig(f) \
    status_t f(impl::engine_t *engine, const exec_ctx_t &ctx, \
            const rnn_utils::sub_buffer_t &a, \
            const rnn_utils::sub_buffer_t &b, \
            const rnn_utils::sub_buffer_t &c, gemm_kind_t gemm_kind) const

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

template <size_t ndims>
using strides_t = std::array<dim_t, ndims>;

struct rnn_offsets_t {
    strides_t<3> src_layer;
    strides_t<4> src_iter;
    strides_t<5> weights_layer;
    strides_t<5> weights_iter;
    strides_t<4> bias;
    strides_t<3> dst_layer;
    strides_t<4> dst_iter;
};

namespace rnn_utils {

enum ws_part_t { gates, states, cell, grid, bias };

namespace kernel_id {
constexpr size_t copy_init_layer = 0;
constexpr size_t copy_init_iter = 1;
constexpr size_t copy_res_layer = 2;
constexpr size_t copy_res_iter = 3;
constexpr size_t bias_fwd = 4;
constexpr size_t cell_fwd = 5;
} // namespace kernel_id

struct conf_t {
    dim_t n_layer, n_iter, n_dir, n_gates, n_states;
    dim_t mb;
    dim_t slc, sic, dhc, dlc;

    dim_t gates_ld, gates_ws_ld;

    dim_t n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];

    dim_t iter_loop;

    dim_t states_ws_ld;
    bool is_fwd, is_training;
    bool use_workspace;

    // Size of workspace for each tensor in bytes
    dim_t ws_states_cell_size, ws_gates_cell_size;
    dim_t ws_gates_size, ws_states_size, scratch_cell_size, ws_per_cell,
            ws_bias_size;

    dim_t ws_gates_offset;
    dim_t ws_states_offset;
    dim_t ws_bias_offset;

    // Element size of each workspace part in bytes
    dim_t ws_gates_elsz, ws_states_elsz, ws_bias_elsz;

    dim_t n_iter_scratch_gates;
    dim_t scratch_gates_size, scratch_gates_elsz, scratch_gates_ld;

    data_type_t acc_data_type;
    data_type_t aux_data_type;
    data_type_t input_data_type;
    data_type_t output_data_type;
    data_type_t src_data_type;
    data_type_t dst_data_type;
    data_type_t wei_layer_type;
    data_type_t wei_iter_type;
    data_type_t bias_data_type;
};

dim_t get_good_ld(
        dim_t arch_ld, dim_t dim, dim_t sizeof_dt, bool ignore_assoc = false);
void init_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &bias_d, data_type_t acc_data_type);
void set_rnn_conf(conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d);
dim_t set_workspace_offsets(
        const conf_t &rnn, dim_t &ws_gates_offset, dim_t &ws_h_state_offset);
dim_t get_workspace_size(const conf_t &rnn);
status_t set_weights_desc(memory_desc_t &weights_md, const conf_t &rnn);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);
const memory_storage_t &get_storage(const memory_storage_t *storage);
const memory_storage_t &get_storage(
        const std::unique_ptr<memory_storage_t> &storage);

struct sub_buffer_t {
    static constexpr dim_t unset = 0;

    sub_buffer_t() : buffer_(nullptr) {}
    sub_buffer_t(const memory_storage_t &buffer, dim_t offset = 0,
            dim_t size = unset)
        : buffer_(buffer.is_null() ? nullptr : buffer.clone()) {
        if (buffer_) { buffer_->set_offset(static_cast<size_t>(offset)); }
    }

    sub_buffer_t(const sub_buffer_t &buffer, dim_t offset = 0)
        : buffer_(buffer ? buffer.buffer_->clone() : nullptr) {
        if (buffer_) { buffer_->set_offset(buffer.offset() + offset); }
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
        assert(buffer_->offset() % types::data_type_size(dt) == 0);
        return buffer_->offset() / types::data_type_size(dt);
    }

    const std::unique_ptr<memory_storage_t> &get_ptr() const { return buffer_; }

private:
    std::unique_ptr<memory_storage_t> buffer_;
};

struct data_helper_t {
    static dim_t type_size(data_type_t d) {
        return static_cast<dim_t>(types::data_type_size(d));
    }
};

struct user_data_t : public data_helper_t {
    using mst = memory_storage_t;
    user_data_t(const mst &src_layer, const mst &wei_layer, const mst &wei_iter,
            const mst &bias, const conf_t &conf, const rnn_offsets_t &offsets)
        : src_layer_(src_layer)
        , wei_layer_(wei_layer)
        , wei_iter_(wei_iter)
        , bias_(bias)
        , conf_(conf)
        , offsets_(offsets) {
        // The packed restriction could be removed by using batched GEMM with
        // appropriate strides.
        assert(offsets_.src_layer[0] < INT_MAX
                && offsets_.src_layer[1] < INT_MAX
                && offsets_.src_layer[2] < INT_MAX);
    }

    dim_t normalized_iter(dim_t dir, dim_t iter_) const { return iter_; }

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

    std::unique_ptr<mst> sub_bias_ptr(dim_t offset, dim_t size) const {
        return bias().get_sub_storage(offset, size);
    }

    sub_buffer_t sub_bias(dim_t lay, dim_t dir) const {
        if (bias().data_handle() == nullptr) return {};
        // bia dimension order: lay, dir, gates, dhc
        auto t_size = type_size(conf_.bias_data_type);
        auto layer_stride = offsets_.bias[0] * t_size;
        auto dir_stride = offsets_.bias[1] * t_size;
        auto cell_size = dir_stride;
        auto offset = layer_stride * lay + dir_stride * dir;

        return {*sub_bias_ptr(offset, cell_size), offset, cell_size};
    }

    const mst &src_layer_;
    const mst &wei_layer_;
    const mst &wei_iter_;
    const mst &bias_;
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
        , bias_(conf.ws_bias_size > 0 ? ws.get_sub_storage(
                        conf.ws_bias_offset, conf.ws_bias_size)
                                      : nullptr) {
        if (gates_) {
            const int n_b = conf_.mb;
            const int n_tb = conf_.n_iter * n_b;
            const int n_dtb = conf_.n_dir * n_tb;
            gates_strides_
                    = {n_dtb * conf_.gates_ws_ld, n_tb * conf_.gates_ws_ld,
                            n_b * conf_.gates_ws_ld, conf_.gates_ws_ld};
        }
        if (states_) {
            const int n_b = conf_.mb;
            const int n_tb = (conf_.n_iter + 1) * n_b;
            const int n_dtb = conf_.n_dir * n_tb;
            states_strides_ = {n_dtb * conf_.states_ws_ld,
                    n_tb * conf_.states_ws_ld, n_b * conf_.states_ws_ld, 1};
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
        //lay,dir,time
        // Logical index into workspace grid
        auto i0 = i0_ + 1;
        auto i0_size = conf_.n_layer + 1;
        auto i2 = i2_ + 1;

        assert(i0 >= 0);

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

        assert(i0 >= 0);

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

        auto i0 = layer + 1;
        auto i2 = time + 1;
        auto off_ = get_offset(states_strides(), {i0, dir, i2, 0})
                * conf_.ws_states_elsz;
        return {states(), off_, conf_.ws_states_cell_size};
    }

    std::unique_ptr<mst> sub_state(dim_t offset, dim_t size) const {
        return states_->get_sub_storage(offset, size);
    }

    sub_buffer_t sub_state(dim_t layer, dim_t dir, dim_t time) const {
        if (!states_) return {};

        auto i0 = layer + 1;
        auto i2 = time + 1;
        auto off_ = get_offset(states_strides(), {i0, dir, i2, 0})
                * conf_.ws_states_elsz;
        return {*sub_state(off_, conf_.ws_states_cell_size), off_,
                conf_.ws_states_cell_size};
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
        key_gemm_layer_fwd,
    };

    scratch_t(const conf_t &conf, const memory_tracking::grantor_t &scratchpad)
        : conf_(conf) {
        using namespace memory_tracking::names;
        gates_ = scratchpad.get_memory_storage(key_rnn_gates);
        cell_ = scratchpad.get_memory_storage(key_rnn_cell);
    }

    struct gemm_pds {
        const primitive_desc_t *iter_fwd_pd;
        const primitive_desc_t *layer_fwd_pd;
    };

    static void book(memory_tracking::registrar_t &scratchpad,
            const conf_t &rnn_conf, const gemm_pds &gemms) {
        using namespace memory_tracking::names;
        if (rnn_conf.scratch_gates_size > 0)
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 1);
        scratchpad.book(key_rnn_cell, rnn_conf.scratch_cell_size, 1);
        // book scratchpad for nested primitives
        if (gemms.layer_fwd_pd) {
            scratchpad.book(key_gemm_layer_fwd,
                    gemms.layer_fwd_pd->scratchpad_registry());
        }
        if (gemms.iter_fwd_pd) {
            scratchpad.book(key_gemm_iter_fwd,
                    gemms.iter_fwd_pd->scratchpad_registry());
        }
    }

    dim_t calc_off_gates(dim_t iter) const {
        return conf_.n_iter_scratch_gates != 1
                ? iter * conf_.mb * conf_.scratch_gates_ld
                : 0;
    };

    const mst *gates() const {
        assert(gates_);
        return (conf_.is_fwd) ? (gates_ ? gates_.get() : diff_gates_.get())
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

    std::unique_ptr<mst> sub_gate(dim_t offset, dim_t size) const {
        return gates_->get_sub_storage(offset, size);
    }

    sub_buffer_t sub_gates(dim_t iter) const {
        auto g = gates();
        if (g == nullptr) return {};

        auto off = calc_off_gates(iter) * conf_.scratch_gates_elsz;
        auto cell_size
                = conf_.mb * conf_.scratch_gates_ld * conf_.scratch_gates_elsz;
        return {*sub_gate(off, cell_size), off, cell_size};
    }

    const mst *cell() const { return cell_.get(); }

    const mst *diff_ht() const { return diff_ht_.get(); }

private:
    const conf_t &conf_;

    std::unique_ptr<mst> gates_;
    std::unique_ptr<mst> diff_gates_;
    std::unique_ptr<mst> cell_;
    std::unique_ptr<mst> diff_states_;
    std::unique_ptr<mst> diff_ht_;
};

inline size_t calc_global_range(size_t gl_range) {
    size_t lc_range = 4;
    return ((gl_range + (lc_range - 1)) / lc_range) * lc_range;
}

} // namespace rnn_utils

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
