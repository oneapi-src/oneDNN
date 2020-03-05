/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef RNN_UTILS_HPP
#define RNN_UTILS_HPP

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

#define rnn_postgemm_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, src_data_t *ws_gates_, \
            scratch_data_t *scratch_gates_, src_data_t *states_t_l_, \
            float *c_states_t_l_, const src_data_t *states_tm1_l_, \
            const float *c_states_tm1_l_, acc_data_t *diff_states_t_l_, \
            acc_data_t *diff_states_t_lp1_, acc_data_t *diff_states_tp1_l_, \
            const float *weights_peephole_, float *bias_, \
            src_data_t *ws_grid_, scratch_data_t *scratch_cell_, \
            src_data_t *states_t_l_copy_) const

#define rnn_cell_execution_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, \
            rnn_utils::cell_position_t cell_position, src_data_t *states_t_l_, \
            float *c_states_t_l_, acc_data_t *diff_states_t_l_, \
            weights_data_t **w_layer_, weights_data_t **w_iter_, \
            const float *weights_peephole_, float **bias_, \
            const src_data_t *states_t_lm1_, const src_data_t *states_tm1_l_, \
            const float *c_states_tm1_l_, acc_data_t *diff_states_t_lp1_, \
            acc_data_t *diff_states_tp1_l_, acc_data_t *diff_w_layer_, \
            acc_data_t *diff_w_iter_, float *diff_weights_peephole_, \
            acc_data_t *diff_bias_, src_data_t *ws_gates_, \
            scratch_data_t *scratch_gates_, src_data_t *ws_grid_, \
            scratch_data_t *scratch_cell_, src_data_t *states_t_l_copy_) const

#define rnn_grid_execution_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, weights_data_t **weights_layer_, \
            weights_data_t **weights_iter_, const float *weights_peephole_, \
            float **bias_, const src_data_t *src_layer_, \
            const src_data_t *src_iter_, const float *src_iter_c_, \
            src_data_t *dst_layer_, src_data_t *dst_iter_, float *dst_iter_c_, \
            src_data_t *ws_states_, float *ws_c_states_, \
            acc_data_t *ws_diff_states_, src_data_t *ws_gates_, \
            src_data_t *ws_grid_, scratch_data_t *scratch_gates_, \
            scratch_data_t *scratch_cell_, acc_data_t *diff_weights_layer_, \
            acc_data_t *diff_weights_iter_, float *diff_weights_peephole_, \
            acc_data_t *diff_bias_) const

#define rnn_gemm_sig(f) \
    void f(const char transA, const char transB, int m, int n, int k, \
            const float alpha, const weights_data_t *a_, const int ldA, \
            const src_data_t *b_, const int ldB, const float beta, \
            acc_data_t *c_, const int ldC) const

#define rnn_bias_prepare_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, float **bias_, const float *b_, \
            float *scratch_bias_) const

#define rnn_bias_finalize_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, float *scratch_bias_, \
            const float *w_iter_comp, const float *w_layer_comp) const

#define rnn_weights_assign_sig(f) \
    void f(const rnn_utils::rnn_conf_t &rnn, const memory_desc_t *md, int nld, \
            int ld, int OC_size, int IC_size, const int n_parts, \
            const int *gates_per_part, const size_t *part_weights_pack_size, \
            weights_data_t **weights_, const weights_data_t *w_, \
            float **bias_, const float *b_, float *scratch_bias_) const

namespace dnnl {
namespace impl {
namespace cpu {

namespace rnn_utils {

enum execution_direction_t {
    l2r,
    r2l,
    bi_concat,
    bi_sum,
};

enum cell_position_t {
    middle_cell = 0x0,
    first_layer = 0x1,
    first_iter = 0x2,
    last_layer = 0x4,
    last_iter = 0x8,
    c_state_first_iter = 0x10,
    c_state_last_iter = 0x20
};

inline cell_position_t &operator|=(cell_position_t &lhs, cell_position_t rhs) {
    lhs = static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
    return lhs;
}

inline cell_position_t operator|(cell_position_t lhs, cell_position_t rhs) {
    return static_cast<cell_position_t>(
            static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

enum data_type_conf_t {
    all_f32,
    all_bf16,
    u8u8u8f32,
    f32u8f32f32,
    u8u8u8u8,
    f32u8f32u8
};

struct rnn_conf_t {
    execution_direction_t exec_dir;
    data_type_conf_t dt_conf;
    int n_layer, n_iter, n_dir, n_gates, n_states;
    int mb;
    int slc, sic, dhc, dlc;
    int gates_ld, gates_nld, gates_ws_ld;
    int n_parts_weights_layer, parts_weights_layer[DNNL_RNN_MAX_N_PARTS];
    int n_parts_weights_iter, parts_weights_iter[DNNL_RNN_MAX_N_PARTS];
    int n_bias, n_parts_bias, parts_bias[DNNL_RNN_MAX_N_PARTS];
    size_t part_weights_iter_pack_size[DNNL_RNN_MAX_N_PARTS],
            part_weights_layer_pack_size[DNNL_RNN_MAX_N_PARTS];
    /* Size of packed data in bytes */
    size_t weights_layer_comp_offset, weights_layer_pack_size,
            weights_iter_comp_offset, weights_iter_pack_size;

    bool copy_bias;
    int weights_layer_ld, weights_layer_nld;
    int diff_weights_layer_ld, diff_weights_layer_nld;
    int weights_iter_ld, weights_iter_nld;
    int diff_weights_iter_ld, diff_weights_iter_nld;
    int states_nld, states_ws_ld, src_layer_ld_, src_iter_ld_, src_iter_c_ld_,
            dst_layer_ld_, dst_iter_ld_, dst_iter_c_ld_;
    int weights_iter_compensation_size, weights_layer_compensation_size;
    bool is_fwd, is_training, is_lbr, is_lstm_peephole;
    bool use_workspace;

    /* Size of workspace for each tensor in bytes */
    size_t ws_gates_size, ws_states_size, ws_c_states_size, ws_diff_states_size,
            scratch_gates_size, scratch_cell_size, ws_grid_comp_size,
            ws_per_cell, ws_bias_size;
    bool merge_gemm_iter, merge_gemm_layer, force_nocopy, use_layer_packed_gemm,
            use_iter_packed_gemm;
    int n_iter_scratch_gates;

    inline bool is_int8() const {
        return utils::one_of(
                dt_conf, u8u8u8f32, f32u8f32f32, u8u8u8u8, f32u8f32u8);
    }

    inline bool skip_src_layer_copy() const {
        // Note: this currently always returns true
        return (exec_dir == l2r)
                && utils::one_of(dt_conf, u8u8u8u8, u8u8u8f32, f32u8f32u8,
                        f32u8f32f32, all_f32, all_bf16);
    }
    inline bool skip_src_iter_copy() const {
        return (exec_dir == l2r) && (src_iter_ld_ > 0)
                && utils::one_of(
                        dt_conf, u8u8u8u8, u8u8u8f32, all_f32, all_bf16);
    }
    inline bool skip_dst_layer_copy() const {
        return (exec_dir == l2r)
                && utils::one_of(
                        dt_conf, u8u8u8u8, f32u8f32u8, all_f32, all_bf16);
    }
    inline bool skip_dst_iter_copy() const {
        return (exec_dir == l2r) && (dst_iter_ld_ > 0)
                && utils::one_of(
                        dt_conf, u8u8u8u8, u8u8u8f32, all_f32, all_bf16);
    }

    inline dim_t src_layer_ld(cell_position_t cell_position) const {
        return (cell_position & first_layer) && skip_src_layer_copy()
                ? src_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                        ? dst_iter_ld_
                        : states_ws_ld;
    }

    inline dim_t src_iter_ld(cell_position_t cell_position) const {
        return (cell_position & first_iter) && skip_src_iter_copy()
                ? src_iter_ld_
                : ((cell_position & last_layer) && skip_dst_layer_copy()
                                        && !(cell_position & first_iter)
                                ? dst_layer_ld_
                                : states_ws_ld);
    }

    inline dim_t src_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_first_iter) ? src_iter_c_ld_
                                                    : states_ws_ld;
    }

    inline dim_t dst_layer_ld(cell_position_t cell_position) const {
        return (cell_position & last_layer) && skip_dst_layer_copy()
                ? dst_layer_ld_
                : (cell_position & last_iter) && skip_dst_iter_copy()
                        ? dst_iter_ld_
                        : states_ws_ld;
    }

    inline dim_t dst_iter_ld(cell_position_t cell_position) const {
        return (cell_position & last_iter) && skip_dst_iter_copy()
                ? dst_iter_ld_
                : states_ws_ld;
    }

    inline dim_t dst_iter_c_ld(cell_position_t cell_position) const {
        return (cell_position & c_state_last_iter) ? dst_iter_c_ld_
                                                   : states_ws_ld;
    }

    // when skipping copy, the output ld can be states_ws_ld,
    // dst_iter_ld or dst_layer_ld depending on the cell position
    inline dim_t dst_ld(cell_position_t cell_position) const {
        return (cell_position & last_layer) ? dst_layer_ld(cell_position)
                                            : dst_iter_ld(cell_position);
    }
    inline dim_t dst_copy_ld(cell_position_t cell_position) const {
        return dst_iter_ld(cell_position);
    }

    inline bool need_gemm_layer(cell_position_t cell_position) const {
        // In case of merge_gemm_layer we might still need a layer gemm if we store
        // the states of the last iteration in the destination memory. The
        // exception of this rule is the first layer though, in which case all
        // states are kept in user's src_layer, hence making full merged gemm
        // possible.
        return IMPLICATION(merge_gemm_layer,
                skip_dst_iter_copy() && (cell_position & last_iter)
                        && !(cell_position & first_layer));
    }
};

bool is_ldigo(const memory_desc_wrapper &md);
bool is_ldgoi(const memory_desc_wrapper &md);

int get_good_ld(int dim, int sizeof_dt);

bool init_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d);

void set_conf(rnn_conf_t &rnn, const rnn_desc_t &rd,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d);

void set_offsets(const rnn_conf_t &rnn, size_t &ws_gates_offset,
        size_t &ws_h_state_offset, size_t &ws_c_state_offset,
        size_t &ws_diff_states_offset, size_t &ws_grid_comp_offset,
        size_t &ws_bias_offset, size_t &scratch_gates_offset,
        size_t &scratch_cell_offset, size_t &scratchpad_size,
        size_t &workspace_size);

void get_scratchpad_and_workspace_sizes(
        const rnn_conf_t &rnn, size_t &scratchpad_size, size_t &workspace_size);
status_t set_expected_desc(
        rnn_conf_t &rnn, memory_desc_t &weights_md, bool is_iter);
status_t set_good_strides(memory_desc_t &weights_md, format_tag_t tag);

template <typename T>
struct ws_gates_aoc {
    ws_gates_aoc(const rnn_conf_t &rnn, T *data)
        : gates_(data, rnn.gates_nld, rnn.gates_ws_ld), DHC_(rnn.dhc) {}
    T &operator()(int batch, int gate, int dhc) {
        return gates_(batch, gate * DHC_ + dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> gates_;
    int DHC_;
};
using ws_gates_aoc_t = ws_gates_aoc<float>;
using ws_gates_aoc_s32_t = ws_gates_aoc<int32_t>;

template <typename T>
struct weights_peephole_aoc_t {
    weights_peephole_aoc_t(const rnn_conf_t &rnn, T *data)
        : weights_peephole_(data, 3, rnn.dhc) {}
    T &operator()(int g, int dhc) { return weights_peephole_(g, dhc); }

private:
    utils::array_offset_calculator<T, 2> weights_peephole_;
};

struct bias_aoc_t {
    bias_aoc_t(const rnn_conf_t &rnn, const float *data)
        : bias_(data, rnn.n_bias, rnn.dhc) {}
    const float &operator()(int bias_n, int dhc) { return bias_(bias_n, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<const float, 2> bias_;
};

template <typename T>
struct ws_states_aoc {
    ws_states_aoc(const rnn_conf_t &rnn, T *data, int leading_dim)
        : state_(data, rnn.states_nld, leading_dim) {}
    ws_states_aoc(const rnn_conf_t &rnn, T *data)
        : state_(data, rnn.states_nld, rnn.states_ws_ld) {}
    T &operator()(int batch, int dhc) { return state_(batch, dhc); }

private:
    dnnl::impl::utils::array_offset_calculator<T, 2> state_;
};

template <typename T>
struct ws_diff_states_aoc {
    ws_diff_states_aoc(const rnn_conf_t &rnn, T *data)
        : diff_states_(data, rnn.n_states + 1, rnn.n_iter + 1, rnn.states_nld,
                rnn.states_ws_ld) {}
    T &operator()(int state_n, int batch, int dhc) {
        return diff_states_(state_n, 0, batch, dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<T, 4> diff_states_;
};

struct ws_diff_w_iter_aoc_t {
    ws_diff_w_iter_aoc_t(const rnn_conf_t &rnn, float *data)
        : diff_weights_iter_(
                data, rnn.diff_weights_iter_nld, rnn.diff_weights_iter_ld)
        , DHC_(rnn.dhc) {}
    float &operator()(int sic, int gate, int dhc) {
        return diff_weights_iter_(sic, gate * DHC_ + dhc);
    }

private:
    dnnl::impl::utils::array_offset_calculator<float, 2> diff_weights_iter_;
    int DHC_;
};
} // namespace rnn_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
