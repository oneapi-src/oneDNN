/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include "matmul_core.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "templates/matmul_core.hpp"
#include "templates/utils.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/graph_map.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/shape_of_tensor.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
blocking_axis_t get_mm_blocking_axis(const logical_tensor_t &inp,
        const logical_tensor_t &wei, const logical_tensor_t &out) {
    auto generate_axis_by_num = [](int num) {
        std::vector<int> ret;
        ret.reserve(num);
        for (int i = 0; i < num; i++)
            ret.emplace_back(i);
        return ret;
    };
    int A_dim_size = inp.get_plain_dims().size(),
        B_dim_size = wei.get_plain_dims().size(),
        C_dim_size = out.get_plain_dims().size();

    blocking_axis_t blocking_axis;
    // add init function here
    blocking_axis.A_bs = transform_axis_plain2blocking(
            inp, generate_axis_by_num(A_dim_size - 2));
    blocking_axis.A_m = transform_axis_plain2blocking(
            inp, std::vector<int> {A_dim_size - 2});
    blocking_axis.A_k = transform_axis_plain2blocking(
            inp, std::vector<int> {A_dim_size - 1});
    blocking_axis.B_bs = transform_axis_plain2blocking(
            wei, generate_axis_by_num(B_dim_size - 2));
    blocking_axis.B_k = transform_axis_plain2blocking(
            wei, std::vector<int> {B_dim_size - 2});
    blocking_axis.B_n = transform_axis_plain2blocking(
            wei, std::vector<int> {B_dim_size - 1});
    blocking_axis.C_bs = transform_axis_plain2blocking(
            out, generate_axis_by_num(C_dim_size - 2));
    blocking_axis.C_m = transform_axis_plain2blocking(
            out, std::vector<int> {C_dim_size - 2});
    blocking_axis.C_n = transform_axis_plain2blocking(
            out, std::vector<int> {C_dim_size - 1});

    return blocking_axis;
}

matmul_core_op_t::matmul_core_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("matmul_core", ins, outs, attrs) {
    COMPILE_ASSERT(info_.inputs_.size() == 2, "matmul_core expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() >= 2 && B_dims.size() >= 2,
            "matmul_core expects each input size equal or bigger than 2 , but "
            "got " << A_dims.size());
    batch_dims_ = get_batch_dims();
    sc_dims expected_out_shape = {merge_vec(batch_dims_,
            {A_dims[A_dims.size() - 2], B_dims[B_dims.size() - 1]})};

    if (info_.outputs_.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                sc_data_format_t(), expected_out_shape, infer_out_dtype(ins)));
    } else {
        COMPILE_ASSERT(
                info_.outputs_.size() == 1, "matmul_core expects 1 output");
        if (!is_dynamic()) {
            COMPILE_ASSERT(info_.outputs_[0]->details_.get_plain_dims()
                            == expected_out_shape,
                    "Bad out dims");
        }
    }
    // record padded_K of input A for matmul_core
    attrs_["temp.padded_A_K"] = std::make_shared<VConst>();
}

body_generator_ptr matmul_core_op_t::create_generator() {
    auto mat_gen = utils::make_unique<gen_matmul_core_t>(this,
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
    mat_gen->bwise_fusion_ = attrs_.get_or_else(op_attr_key::bwise_fuse, false);
    return std::move(mat_gen);
}

float matmul_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

sc_dims matmul_core_op_t::get_batch_dims() const {
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    if (is_dynamic()) {
        return get_batch_dims_impl(A_dims, B_dims);
    } else {
        return get_batch_dims_with_bc_impl(A_dims, B_dims);
    }
}

// directly use batch dims from A or B.
sc_dims matmul_core_op_t::get_batch_dims_impl(
        const sc_dims &A_dims, const sc_dims &B_dims) {
    return A_dims.size() > B_dims.size()
            ? sc_dims {A_dims.begin(), A_dims.end() - 2}
            : sc_dims {B_dims.begin(), B_dims.end() - 2};
}

sc_dims matmul_core_op_t::get_batch_dims_with_bc_impl(
        const sc_dims &A_plain_dims, const sc_dims &B_plain_dims) {
    sc_dims C_batch_dims;
    bool is_A_dims_long = A_plain_dims.size() >= B_plain_dims.size();
    auto long_dims = is_A_dims_long ? A_plain_dims : B_plain_dims;
    auto short_dims = is_A_dims_long ? B_plain_dims : A_plain_dims;
    int num_dims = long_dims.size();
    // In 2D case, no batch axis, C_batch_dims are empty, no broadcast
    if (num_dims < 3) { return C_batch_dims; }

    int dims_diff = long_dims.size() - short_dims.size();
    C_batch_dims.insert(C_batch_dims.end(), long_dims.begin(),
            long_dims.begin() + dims_diff);

    if (std::equal(long_dims.begin() + dims_diff, long_dims.end() - 2,
                short_dims.begin())) {
        // all batch dims are equal, no broadcast
        C_batch_dims.insert(C_batch_dims.end(), long_dims.begin() + dims_diff,
                long_dims.end() - 2);
        return C_batch_dims;
    }

    for (int i = dims_diff; i <= num_dims - 3; ++i) {
        if (long_dims[i] == short_dims[i - dims_diff]) {
            // no broadcasting at this dim
            C_batch_dims.push_back(long_dims[i]);
        } else { // long_dims[i] != short_dims[i - dims_diff]
            if (long_dims[i] == 1 && short_dims[i - dims_diff] != 1) {
                C_batch_dims.push_back(short_dims[i - dims_diff]);
            } else if (long_dims[i] != 1 && short_dims[i - dims_diff] == 1) {
                C_batch_dims.push_back(long_dims[i]);
            } else { // both are not 1, invalid broadcast
                COMPILE_ASSERT(
                        false, "Can not broadcast in batch dims properly");
            }
        }
    }
    return C_batch_dims;
}

sc_data_type_t matmul_core_op_t::infer_out_dtype(
        const std::vector<graph_tensor_ptr> &ins) {
    if (ins.at(0)->details_.dtype_ == datatypes::u8
            || ins.at(0)->details_.dtype_ == datatypes::s8) {
        assert(ins.at(1)->details_.dtype_ == datatypes::s8);
        return datatypes::s32;
    }
    return datatypes::f32;
}

void matmul_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    bool dynamic = is_dynamic();
    if (!config_data_) {
        config_data_ = create_generator()->get_default_config(ctx);
    }
    const bool is_A_not_blocking
            = !info_.inputs_[0]->details_.get_format().is_blocking();
    const sc_dims &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &A_blocking_dims
            = info_.inputs_[0]->details_.get_blocking_dims();
    const bool is_B_not_blocking
            = !info_.inputs_[1]->details_.get_format().is_blocking();
    const sc_dims &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &B_blocking_dims
            = info_.inputs_[1]->details_.get_blocking_dims();
    const sc_dims &C_dims = info_.outputs_[0]->details_.get_plain_dims();
    const sc_dim M = A_dims[A_dims.size() - 2];
    const sc_dim K = A_dims.back();
    const sc_dim N = B_dims.back();
    auto &graph = get_owner_graph();
    matmul_core_config_t &tcfg = *config_data_.get_as<matmul_core_config_t>();
    int M_block = tcfg.M_block, N_block = tcfg.N_block, K_block = tcfg.K_block;

    in_formats.resize(2);
    out_formats.resize(1);
    sc_data_type_t B_dtype = info_.inputs_[1]->details_.dtype_;
    sc_data_format_t A_format = info_.inputs_[0]->details_.get_format();
    sc_data_format_t B_format = info_.inputs_[1]->details_.get_format();
    bool is_B_vnni_low_fp = ops::is_vnni_low_fp(ctx, B_dtype);
    // constant check
    bool constant_A = false, constant_B = false;
    bool block_A = attrs_.get_or_else("block_A", false);
    bool block_B = attrs_.get_or_else("block_B", false);
    bool transposed_a = attrs_.get_or_else("transposed_a", false);
    bool transposed_b = attrs_.get_or_else("transposed_b", false);
    if (info_.inputs_[0]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[0]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)) {
        constant_A = true;
    } else {
        bool constant_A_parents = true;
        for (const auto &input :
                info_.inputs_[0]->producer_owner_->get_inputs()) {
            auto parent_node = input->producer_owner_;
            constant_A_parents &= (parent_node->attrs_.get_or_else(
                                           "constant", const_kind::not_const)
                    || parent_node->isa<constant_op_t>());
        }
        constant_A = constant_A_parents
                && !info_.inputs_[0]->producer_owner_->get_inputs().empty();
    }

    if (info_.inputs_[1]->producer_owner_->isa<constant_op_t>()
            || info_.inputs_[1]->producer_owner_->attrs_.get_or_else(
                    "constant", const_kind::not_const)) {
        constant_B = true;
    } else {
        bool constant_B_parents = true;
        for (const auto &input :
                info_.inputs_[1]->producer_owner_->get_inputs()) {
            auto parent_node = input->producer_owner_;
            constant_B_parents &= (parent_node->attrs_.get_or_else(
                                           "constant", const_kind::not_const)
                    || parent_node->isa<constant_op_t>());
        }
        constant_B = constant_B_parents
                && !info_.inputs_[1]->producer_owner_->get_inputs().empty();
    }
    assert(in_formats.size() == 2);
    std::vector<int> blk_candidates = get_dynamic_block_candidates();
    std::vector<int> m_blk_candidates = get_dynamic_batch_block_candidates();
    auto A_m_blk = M_block, B_n_blk = N_block, A_k_blk = K_block;
    auto C_m_blk = M_block, C_n_blk = N_block, B_k_blk = K_block;
    sc_data_format_t ret_A_format, ret_B_format, ret_C_format;
    auto cur_format_set = std::unordered_set<std::vector<sc_data_format_t>>();
    auto cur_dispatch_key_set = dispatch_key_set_t();
    bool first = true;
    std::vector<bool> is_padding = {false, true};
    std::vector<bool> is_output_plain = {false, true};
    for (auto &m_b : m_blk_candidates) { // M
        for (auto &n_b : blk_candidates) { // N
            for (auto &k_b : blk_candidates) { // K
                for (auto A_isp : is_padding) { // A is_padding
                    for (auto B_isp : is_padding) { // B is_padding
                        for (auto out_plain :
                                is_output_plain) { // output plain, always
                            // false in dynamic
                            if (is_dynamic_dim(M_block)
                                    || is_dynamic_dim(N_block)) {
                                if (is_dynamic_dim(M_block)) {
                                    A_m_blk = C_m_blk = m_b;
                                    COMPILE_ASSERT(!constant_A,
                                            "if M is dynamic, input A should "
                                            "not "
                                            "be constant!");
                                } else {
                                    A_m_blk = C_m_blk = M_block;
                                }
                                if (is_dynamic_dim(N_block)) {
                                    B_n_blk = C_n_blk = n_b;
                                    COMPILE_ASSERT(!constant_B,
                                            "if N is dynamic, input B should "
                                            "not "
                                            "be constant!");
                                } else {
                                    B_n_blk = C_n_blk = N_block;
                                }
                            }
                            if (is_dynamic_dim(K_block)) {
                                A_k_blk = B_k_blk = k_b;
                                COMPILE_ASSERT(!constant_A && !constant_B,
                                        "if K is dynamic, input A and B should "
                                        "not "
                                        "be constant!");
                            }
                            // process A
                            if (constant_A || block_A || A_isp
                                    || (A_dims.size() > 2
                                            && !A_format.is_plain()) // follow
                                    // original
                                    // logic
                                    || (!dynamic && A_format.is_blocking())
                                    || (!is_dynamic_dim(M) && M % M_block)
                                    || (!is_dynamic_dim(K) && K % K_block)) {
                                // should be blocking
                                if (A_dims.size() == 2) {
                                    ret_A_format = sc_data_format_t::MKmk(
                                            A_m_blk, A_k_blk);
                                } else {
                                    // regular ND*ND matmul (non-batch
                                    // format) whether constant and no
                                    // transA
                                    auto A_formt_m_blk = transposed_a
                                            ? A_format.blocks_[1]
                                            : A_format.blocks_[0];
                                    auto A_formt_k_blk = transposed_a
                                            ? A_format.blocks_[0]
                                            : A_format.blocks_[1];
                                    if (A_formt_m_blk != A_m_blk
                                            || A_formt_k_blk != A_k_blk
                                            || A_format.format_code_.get(
                                                       A_blocking_dims.size()
                                                       - 1)
                                                    == static_cast<int>(
                                                            A_dims.size()
                                                            - 2)) {
                                        ret_A_format = sc_data_format_t(
                                                sc_data_format_kind_t::
                                                        get_2dblocking_by_dims(
                                                                A_dims.size()),
                                                {A_m_blk, A_k_blk});
                                    } else {
                                        ret_A_format = A_format;
                                    }
                                }
                            } else {
                                // static or dynamic with no padding
                                if (A_dims.size() == 2) {
                                    ret_A_format = sc_data_format_t::MK();
                                } else {
                                    // regular ND*ND matmul (non-batch
                                    // format)
                                    ret_A_format = A_format;
                                }
                            }
                            // if it is dynamic, follows the layout of last
                            // layer.
                            if (!is_A_not_blocking && dynamic) {
                                ret_A_format = A_format;
                                // follow last layer's config
                                if (A_format.blocks_[0]) {
                                    A_m_blk = C_m_blk = transposed_a
                                            ? A_format.blocks_[1]
                                            : A_format.blocks_[0];
                                }
                                if (A_format.blocks_[1]) {
                                    A_k_blk = B_k_blk = transposed_a
                                            ? A_format.blocks_[0]
                                            : A_format.blocks_[1];
                                }
                            }
                            // process B
                            if (utils::is_one_of(B_dtype, datatypes::u8,
                                        datatypes::s8)) {
                                if (B_dims.size() == 2) {
                                    ret_B_format = sc_data_format_t::NKkn4k(
                                            B_k_blk, B_n_blk);
                                } else {
                                    ret_B_format = sc_data_format_t(
                                            sc_data_format_kind_t::
                                                    get_2dblocking_by_dims(
                                                            B_dims.size(), true,
                                                            true),
                                            {B_k_blk, B_n_blk, 4});
                                }
                            } else if (is_B_vnni_low_fp) {
                                if (B_dims.size() == 2) {
                                    ret_B_format = sc_data_format_t::NKkn2k(
                                            B_k_blk, B_n_blk);
                                } else {
                                    ret_B_format = sc_data_format_t(
                                            sc_data_format_kind_t::
                                                    get_2dblocking_by_dims(
                                                            B_dims.size(), true,
                                                            true),
                                            {B_k_blk, B_n_blk, 2});
                                }
                            } else {
                                auto B_format_kind = sc_data_format_kind_t::
                                        get_2dblocking_by_dims(
                                                B_dims.size(), true);
                                if (constant_B || block_B || B_isp
                                        || (B_dims.size() > 2
                                                && !B_format.is_plain())
                                        || (!dynamic && B_format.is_blocking())
                                        || (!is_dynamic_dim(K) && K % K_block)
                                        || (!is_dynamic_dim(N)
                                                && N % N_block)) {
                                    // should be blocking
                                    if (B_dims.size() == 2) {
                                        ret_B_format = sc_data_format_t::NKkn(
                                                B_k_blk, B_n_blk);
                                    } else {
                                        // regular ND*ND matmul (non-batch
                                        // format) whether constant and no
                                        // transA
                                        auto B_formt_k_blk = transposed_b
                                                ? B_format.blocks_[1]
                                                : B_format.blocks_[0];
                                        auto B_formt_n_blk = transposed_b
                                                ? B_format.blocks_[0]
                                                : B_format.blocks_[1];
                                        if (B_formt_k_blk != B_k_blk
                                                || B_formt_n_blk != B_n_blk
                                                || B_format.format_code_.get(
                                                           B_dims.size() - 1)
                                                        == static_cast<int>(
                                                                B_dims.size()
                                                                - 2)) {
                                            ret_B_format = sc_data_format_t(
                                                    B_format_kind,
                                                    {B_k_blk, B_n_blk});
                                        } else {
                                            ret_B_format = B_format;
                                        }
                                    }
                                } else {
                                    // static or dynamic with no padding
                                    if (B_dims.size() == 2) {
                                        ret_B_format = sc_data_format_t::KN();
                                    } else {
                                        // regular ND*ND matmul (non-batch
                                        // format)
                                        ret_B_format = B_format;
                                    }
                                }
                            }
                            // process C
                            if (((!constant_A && !constant_B && !dynamic
                                         && M % M_block == 0
                                         && N % N_block == 0)
                                        || (dynamic && !A_isp && !B_isp))
                                    && out_plain) {
                                if (C_dims.size() == 2) {
                                    ret_C_format = sc_data_format_t::MK();
                                } else {
                                    // regular ND*ND matmul (non-batch
                                    // format)
                                    ret_C_format = sc_data_format_t::
                                            get_plain_by_dims(C_dims.size());
                                }
                            } else {
                                if (C_dims.size() == 2) {
                                    ret_C_format = sc_data_format_t::MKmk(
                                            C_m_blk, C_n_blk);
                                } else {
                                    // regular ND*ND matmul (non-batch
                                    // format)
                                    ret_C_format = sc_data_format_t(
                                            sc_data_format_kind_t::
                                                    get_2dblocking_by_dims(
                                                            C_dims.size()),
                                            {C_m_blk, C_n_blk});
                                }
                            }
                            std::vector<std::vector<sc_dim>> var_block
                                    = {{A_m_blk, A_k_blk}, {B_k_blk, B_n_blk},
                                            {C_m_blk, C_n_blk}};
                            std::vector<sc_data_format_t> ret_formats = {
                                    ret_A_format, ret_B_format, ret_C_format};
                            if (dynamic) {
                                op_dispatch_key_t ret_key(
                                        var_block, ret_formats);
                                cur_dispatch_key_set.set_.insert(ret_key);
                            }
                            if (cur_format_set.find(ret_formats)
                                    == cur_format_set.end()) {
                                in_formats[0].emplace_back(ret_A_format);
                                in_formats[1].emplace_back(ret_B_format);
                                out_formats[0].emplace_back(ret_C_format);
                                cur_format_set.insert(ret_formats);
                            }
                            // reset default cfg to first candidate in
                            // dynamic for try mode in fuse op pass.
                            if (first && dynamic) {
                                tcfg.M_block = C_m_blk;
                                tcfg.N_block = C_n_blk;
                                tcfg.K_block = std::min(A_k_blk, B_k_blk);
                                first = false;
                            }

                            // break output plain if op is dynamic
                            if (dynamic) { break; }
                        }
                        // break is B padding loop if it is static
                        if (!is_dynamic_dim(K) && !is_dynamic_dim(N)) { break; }
                    }
                    // break is A padding loop if it is static
                    if (!is_dynamic_dim(M) && !is_dynamic_dim(K)) { break; }
                }
                // break the k loop if it is static
                if (!dynamic) { break; }
            }
            // break the n loop if it is static
            if (!dynamic) { break; }
        }
        // break the m loop if it is static
        if (!dynamic) { break; }
    }
    if (dynamic) {
        auto &dispatch_key_set = get_dispatch_key_set();
        dispatch_key_set->get_inner_set().insert(
                cur_dispatch_key_set.set_.begin(),
                cur_dispatch_key_set.set_.end());
    }
    // To calculate padded K of input A
    auto pad_K_num = utils::divide_and_ceil(K, K_block);
    attrs_["temp.padded_A_K"].get<std::shared_ptr<VConst>>()->var_
            = pad_K_num * K_block;
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void matmul_core_op_t::set_config_by_key(
        const op_dispatch_key_t &key, const context_ptr &ctx) {
    assert(key.var_block_.size() == 3);
    config_data_ = create_generator()->get_default_config(ctx);
    matmul_core_config_t &tcfg = *config_data_.get_as<matmul_core_config_t>();
    tcfg.M_block = key.var_block_[2][0];
    tcfg.N_block = key.var_block_[2][1];
    tcfg.K_block = std::min(key.var_block_[0][1], key.var_block_[1][0]);
}

std::vector<int> matmul_core_op_t::get_impl_dispatch_candidates(
        const context_ptr &ctx) {
    return {};
}

sc_op_ptr matmul_core_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    // whether we need special compensation for microkernel.
    bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
            && info_.inputs_[0]->details_.dtype_ == datatypes::s8
            && (!ctx->machine_.brgemm_use_amx_
                    || (ctx->machine_.brgemm_use_amx_
                            && !ctx->machine_.cpu_flags_.fAVX512AMXINT8));

    auto cur_node = shared_from_this();

    auto data_com = get_data_compensation(mgr);
    auto s8s8_weight_com
            = get_s8s8_and_weight_compensation(mgr, s8s8_compensation);
    auto const_com = get_constant_compensation(mgr);

    if (data_com) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0], data_com->get_outputs()[0]}, {},
                {});
    }

    if (s8s8_weight_com[0]) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0],
                        s8s8_weight_com[0]->get_outputs()[0]},
                {}, {});
    }
    if (s8s8_weight_com[1]) {
        cur_node = mgr.make("sub",
                {cur_node->get_outputs()[0],
                        s8s8_weight_com[1]->get_outputs()[0]},
                {}, {});
    }
    if (const_com) {
        cur_node = mgr.make("add",
                {cur_node->get_outputs()[0], const_com->get_outputs()[0]}, {},
                {});
    }

    return cur_node;
}

sc_op_ptr matmul_core_op_t::get_data_compensation(sc_graph_t &mgr) {
    std::string weight_zp_key = attr_keys::weight_zero_points;
    std::string dyn_weight_zp_key = attr_keys::dyn_weight_zero_points;
    bool is_dyn_quan = attrs_.has_key(dyn_weight_zp_key);
    auto weight_zero_points
            = attrs_.get_or_else(weight_zp_key, std::vector<int> {0});
    auto dyn_weight_zero_points
            = attrs_.get_or_else(dyn_weight_zp_key, graph_tensor_ptr());
    if (!is_dyn_quan
            && (weight_zero_points.empty()
                    || (std::all_of(weight_zero_points.begin(),
                            weight_zero_points.end(),
                            [](int i) { return i == 0; })))) {
        return nullptr;
    }
    if (is_dyn_quan && !dyn_weight_zero_points) { return nullptr; }
    auto data = info_.inputs_[0];
    auto cast_node = mgr.make("cast", {data}, {}, {{"dtype", datatypes::s32}});

    // K is reduce axis
    std::vector<int> rdaxis
            = {static_cast<int>(data->details_.get_plain_dims().size()) - 1};

    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});
    sc_op_ptr mul_node;
    if (is_dyn_quan) {
        COMPILE_ASSERT(dyn_weight_zero_points->details_.get_plain_dims()
                        == sc_dims {1},
                "matmul_core does not support per channel weight zero "
                "points compensation yet");
        mul_node = mgr.make("mul",
                {reduce_node->get_outputs()[0], dyn_weight_zero_points}, {},
                {});
    } else {
        std::shared_ptr<static_data_t> weight_zero_points_ptr
                = std::make_shared<static_data_t>(weight_zero_points);
        sc_dims const_plain_dims;
        sc_data_format_t const_format;
        if (weight_zero_points.size() == 1) {
            // per tensor
            const_plain_dims = {1};
        } else {
            // per channel
            COMPILE_ASSERT(0,
                    "matmul_core does not support per channel weight zero "
                    "points "
                    "compensation yet");
            auto weight = info_.inputs_[1];
            auto weight_plain_dims = weight->details_.get_plain_dims();
            assert(weight_plain_dims.back()
                    == static_cast<int64_t>(weight_zero_points.size()));
            const_plain_dims = {1, weight_plain_dims.back()};
            const_format = info_.inputs_[1]->details_.get_format();
        }
        auto constant_node = mgr.make("constant", {}, {},
                {{"values", weight_zero_points_ptr}, {"dtype", datatypes::s32},
                        {"plain_dims", const_plain_dims},
                        {"format", const_format}});
        mul_node = mgr.make("mul",
                {reduce_node->get_outputs()[0],
                        constant_node->get_outputs()[0]},
                {}, {});
    }
    if (data->details_.get_plain_dims().size() < batch_dims_.size() + 2) {
        sc_dims unsqueeze_shape(
                batch_dims_.size() + 2 - data->details_.get_plain_dims().size(),
                1);
        sc_dims reshape_dest
                = merge_vec(unsqueeze_shape, data->details_.get_plain_dims());
        reshape_dest.at(reshape_dest.size() - 1) = 1;
        auto reshape_fmt = info_.outputs_[0]->details_.get_format();
        auto reshape_node = mgr.make("tensor_view", mul_node->get_outputs(),
                {graph_tensor::make(reshape_dest, sc_data_format_t(),
                        mul_node->get_outputs()[0]->details_.dtype_)},
                {{"shape", reshape_dest}, {"format", reshape_fmt}});
        return reshape_node;
    }
    return mul_node;
}

std::vector<sc_op_ptr> matmul_core_op_t::get_s8s8_and_weight_compensation(
        sc_graph_t &mgr, bool s8s8_compensation) {
    std::string data_zp_key = attr_keys::data_zero_points;
    std::string dyn_data_zp_key = attr_keys::dyn_data_zero_points;
    bool is_dyn_quan = attrs_.has_key(dyn_data_zp_key);
    auto data_zero_points
            = attrs_.get_or_else(data_zp_key, std::vector<int> {0});
    auto dyn_data_zero_points
            = attrs_.get_or_else(dyn_data_zp_key, graph_tensor_ptr());
    bool weight_compensation = (is_dyn_quan && dyn_data_zero_points)
            || (!data_zero_points.empty()
                    && !(std::all_of(data_zero_points.begin(),
                            data_zero_points.end(),
                            [](int i) { return i == 0; })));
    std::vector<sc_op_ptr> nodes = {nullptr, nullptr};
    if (!s8s8_compensation && !weight_compensation) { return nodes; }

    auto weight = info_.inputs_[1];
    auto cast_node
            = mgr.make("cast", {weight}, {}, {{"dtype", datatypes::s32}});

    // K is reduce axis
    std::vector<int> rdaxis
            = {static_cast<int>(weight->details_.get_plain_dims().size()) - 2};
    auto reduce_node = mgr.make("reduce", cast_node->get_outputs(), {},
            {{"rd_axis", rdaxis}, {"rd_op", 0}, {"keep_dims", true}});

    if (weight_compensation) {
        if (is_dyn_quan) {
            COMPILE_ASSERT(dyn_data_zero_points->details_.get_plain_dims()
                            == sc_dims {1},
                    "matmul_core does not support per channel data zero "
                    "points compensation yet");
            nodes[0] = mgr.make("mul",
                    {reduce_node->get_outputs()[0], dyn_data_zero_points}, {},
                    {});
        } else {
            std::shared_ptr<static_data_t> data_zero_points_ptr
                    = std::make_shared<static_data_t>(data_zero_points);
            sc_dims const_plain_dims;
            sc_data_format_t const_format;
            if (data_zero_points.size() == 1) {
                // per tensor
                const_plain_dims = {1};
            } else {
                // per channel
                COMPILE_ASSERT(0,
                        "matmul_core does not support per channel data zero "
                        "points "
                        "compensation yet");
                auto data = info_.inputs_[0];
                auto data_plain_dims = data->details_.get_plain_dims();
                size_t bds = batch_dims_.size();
                assert(data_plain_dims[bds]
                        == static_cast<int64_t>(data_zero_points.size()));
                const_plain_dims = {data_plain_dims[bds], 1};
                const_format = info_.inputs_[0]->details_.get_format();
            }
            auto constant_node = mgr.make("constant", {}, {},
                    {{"values", data_zero_points_ptr},
                            {"dtype", datatypes::s32},
                            {"plain_dims", const_plain_dims},
                            {"format", const_format}});
            nodes[0] = mgr.make("mul",
                    {reduce_node->get_outputs()[0],
                            constant_node->get_outputs()[0]},
                    {}, {});
        }
        if (weight->details_.get_plain_dims().size() < batch_dims_.size() + 2) {
            sc_dims unsqueeze_shape(batch_dims_.size() + 2
                            - weight->details_.get_plain_dims().size(),
                    1);
            sc_dims reshape_dest = merge_vec(
                    unsqueeze_shape, weight->details_.get_plain_dims());
            reshape_dest.at(reshape_dest.size() - 2) = 1;
            auto reshape_fmt = info_.outputs_[0]->details_.get_format();
            nodes[0] = mgr.make("tensor_view", nodes[0]->get_outputs(),
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            nodes[0]->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt}});
        }
    }

    if (s8s8_compensation) {
        auto s8_constant_node = mgr.make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<int> {128})},
                        {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        nodes[1] = mgr.make("mul",
                {reduce_node->get_outputs()[0],
                        s8_constant_node->get_outputs()[0]},
                {}, {});
        if (weight->details_.get_plain_dims().size() < batch_dims_.size() + 2) {
            sc_dims unsqueeze_shape(batch_dims_.size() + 2
                            - weight->details_.get_plain_dims().size(),
                    1);
            sc_dims reshape_dest = merge_vec(
                    unsqueeze_shape, weight->details_.get_plain_dims());
            reshape_dest.at(reshape_dest.size() - 2) = 1;
            auto reshape_fmt = info_.outputs_[0]->details_.get_format();
            nodes[1] = mgr.make("tensor_view", nodes[1]->get_outputs(),
                    {graph_tensor::make(reshape_dest, sc_data_format_t(),
                            nodes[1]->get_outputs()[0]->details_.dtype_)},
                    {{"shape", reshape_dest}, {"format", reshape_fmt}});
        }
    }
    return nodes;
}

sc_op_ptr matmul_core_op_t::get_constant_compensation(sc_graph_t &mgr) {
    bool is_dyn_quan = attrs_.has_key(attr_keys::dyn_data_zero_points);
    auto data_zero_points = attrs_.get_or_else(
            attr_keys::data_zero_points, std::vector<int> {0});
    auto weight_zero_points = attrs_.get_or_else(
            attr_keys::weight_zero_points, std::vector<int> {0});
    auto dyn_data_zero_points = attrs_.get_or_else(
            attr_keys::dyn_data_zero_points, graph_tensor_ptr());
    auto dyn_weight_zero_points = attrs_.get_or_else(
            attr_keys::dyn_weight_zero_points, graph_tensor_ptr());
    auto K_orig = info_.inputs_[0]->details_.get_plain_dims().at(
            info_.inputs_[0]->details_.get_plain_dims().size() - 1);
    int K = static_cast<int>(K_orig);
    COMPILE_ASSERT(attrs_.has_key("temp.padded_A_K"),
            "No related VConst set, which maybe cause correctness error")
    sc_op_ptr ret_node;
    if (is_dyn_quan) {
        if (!dyn_data_zero_points || !dyn_weight_zero_points) {
            return nullptr;
        }
    } else {
        if (data_zero_points.empty() || weight_zero_points.empty()) {
            return nullptr;
        }
        if ((std::all_of(data_zero_points.begin(), data_zero_points.end(),
                    [](int i) { return i == 0; }))
                || (std::all_of(weight_zero_points.begin(),
                        weight_zero_points.end(),
                        [](int i) { return i == 0; }))) {
            return nullptr;
        }
    }
    if (is_dynamic_dim(K_orig)) {
        COMPILE_ASSERT(!is_dyn_quan,
                "Currently dynamic shape hasn't integrated with dynamic "
                "quantize.");
        ret_node = mgr.make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(std::vector<int> {
                                 data_zero_points[0] * weight_zero_points[0]})},
                        {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}});
        auto weight = info_.inputs_[1];
        int shape_idx = static_cast<int>(
                weight->details_.get_plain_dims().size() - 2);
        auto reduce_shape = mgr.make("shape_of_tensor", {weight}, {},
                {{"shape_idx", shape_idx},
                        {attr_keys::padding_shape_type,
                                static_cast<int>(padding_shape_etype_t::
                                                matmul_padding)}});
        ret_node = mgr.make("mul",
                {ret_node->get_outputs()[0], reduce_shape->get_outputs()[0]},
                {}, {});
    } else {
        if (is_dyn_quan) {
            COMPILE_ASSERT(dyn_data_zero_points->details_.get_plain_dims()
                                    == sc_dims {1}
                            && dyn_weight_zero_points->details_.get_plain_dims()
                                    == sc_dims {1},
                    "matmul_core does not support per channel data/weight zero "
                    "points compensation yet");
            ret_node = mgr.make("mul",
                    {dyn_data_zero_points, dyn_weight_zero_points}, {}, {});
            auto const_reduce = mgr.make("constant", {}, {},
                    {{"dtype", datatypes::s32},
                            {"values",
                                    std::make_shared<static_data_t>(
                                            &K, sizeof(int))},
                            {"plain_dims", sc_dims {1}},
                            {"format", sc_data_format_t()}, {"temp.val/var", 1},
                            {"temp.var", attrs_["temp.padded_A_K"]}});
            ret_node = mgr.make("mul",
                    {ret_node->get_outputs()[0],
                            const_reduce->get_outputs()[0]},
                    {}, {});
        } else {
            COMPILE_ASSERT(data_zero_points.size() == 1
                            && weight_zero_points.size() == 1,
                    "matmul_core does not support per channel data/weight zero "
                    "points "
                    "compensation yet");
            ret_node = mgr.make("constant", {}, {},
                    {{"values",
                             std::make_shared<static_data_t>(
                                     std::vector<int> {data_zero_points[0]
                                             * weight_zero_points[0] * K})},
                            {"dtype", datatypes::s32},
                            {"plain_dims", sc_dims {1}},
                            {"format", sc_data_format_t()},
                            {"temp.val/var",
                                    data_zero_points[0]
                                            * weight_zero_points[0]},
                            {"temp.var", attrs_["temp.padded_A_K"]}});
        }
    }
    return ret_node;
}

shape_rl_vec matmul_core_op_t::get_dynamic_shape_relations() const {
    return get_shape_relations_impl(get_inputs()[0]->details_.get_plain_dims(),
            get_inputs()[1]->details_.get_plain_dims(),
            get_outputs()[0]->details_.get_plain_dims());
}

shape_rl_vec matmul_core_op_t::get_shape_relations_impl(
        const std::vector<sc_dim> &data_plain_dims,
        const std::vector<sc_dim> &weight_plain_dims,
        const std::vector<sc_dim> &out_plain_dims) {
    assert(data_plain_dims.size() == weight_plain_dims.size()
            || data_plain_dims.size() == 2 || weight_plain_dims.size() == 2);
    shape_rl_vec ret;
    auto data_M = data_plain_dims[data_plain_dims.size() - 2];
    auto data_K = data_plain_dims[data_plain_dims.size() - 1];
    auto weight_K = weight_plain_dims[weight_plain_dims.size() - 2];
    auto weight_N = weight_plain_dims[weight_plain_dims.size() - 1];
    auto out_M = out_plain_dims[out_plain_dims.size() - 2];
    auto out_N = out_plain_dims[out_plain_dims.size() - 1];
    if (is_dynamic_dim(data_K) || is_dynamic_dim(weight_K)) {
        ret.emplace_back(data_K, weight_K);
    }
    if (is_dynamic_dim(data_M) || is_dynamic_dim(out_M)) {
        ret.emplace_back(data_M, out_M);
    }
    if (is_dynamic_dim(weight_N) || is_dynamic_dim(out_N)) {
        ret.emplace_back(weight_N, out_N);
    }
    if (data_plain_dims.size() == weight_plain_dims.size()
            && data_plain_dims.size() > 2) {
        for (size_t i = 0; i < data_plain_dims.size() - 2; i++) {
            if (is_dynamic_dim(data_plain_dims[i])
                    || is_dynamic_dim(weight_plain_dims[i])) {
                ret.emplace_back(data_plain_dims[i], weight_plain_dims[i]);
                if (is_dynamic_dim(data_plain_dims[i])) {
                    ret.emplace_back(data_plain_dims[i], out_plain_dims[i]);
                } else {
                    ret.emplace_back(weight_plain_dims[i], out_plain_dims[i]);
                }
            }
        }
    }
    return ret;
}

sc_dims matmul_core_op_t::get_bwise_fuse_shrink_dims() {
    // Currently fordbid N-axis fuse, skip check weight
    auto out_fmt = info_.outputs_[0]->details_.get_format(),
         inp_fmt = info_.inputs_[0]->details_.get_format();
    auto output_dims = info_.outputs_[0]->details_.get_blocking_dims();
    int bs_size = batch_dims_.size();

    auto out_p2b_map = out_fmt.format_code_.collect_p2b_mapping(),
         inp_p2b_map = inp_fmt.format_code_.collect_p2b_mapping();

    COMPILE_ASSERT(out_p2b_map.size() >= 2,
            "Matmul core output should at least have MN dimension")
    // validate input according shrinked output graph tensor
    int cnt = 0;
    for (; cnt < bs_size; cnt++) {
        auto plain_pos = out_fmt.format_code_.get(cnt);
        if (out_p2b_map[plain_pos].front() != cnt
                || inp_p2b_map[plain_pos].front() != cnt)
            break;
    }
    return {output_dims.begin(), output_dims.begin() + cnt};
}

void matmul_core_op_t::collect_shrinked_lt_map(
        int bw_size, gt2gt_map &bw_lt_map) {
    // set output
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_outputs()[0], bw_size);
    auto &out_plain_dims
            = bw_lt_map.get(get_outputs()[0])->details_.get_plain_dims();
    auto old_inp_dims = get_inputs()[0]->details_.get_plain_dims();
    auto old_wei_dims = get_inputs()[1]->details_.get_plain_dims();
    // MK
    sc_dims inp_plain_dims = {
            out_plain_dims.at(out_plain_dims.size() - 2), old_inp_dims.back()};
    // KN
    sc_dims wei_plain_dims
            = {old_wei_dims.at(old_wei_dims.size() - 2), out_plain_dims.back()};

    int bs_out = out_plain_dims.size() - 2;
    int bs_inp = old_inp_dims.size() - 2;
    int bs_wei = old_wei_dims.size() - 2;

    for (int i = 1; i <= bs_out; i++) {
        if (i <= bs_inp) {
            inp_plain_dims.insert(inp_plain_dims.begin(),
                    out_plain_dims.at(out_plain_dims.size() - 2 - i));
        }
        if (i <= bs_wei) {
            wei_plain_dims.insert(wei_plain_dims.begin(),
                    out_plain_dims.at(out_plain_dims.size() - 2 - i));
        }
    }
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[0], inp_plain_dims);
    op_traits::batchwise_shrinkable_t::record_shrinked_gt(
            bw_lt_map, get_inputs()[1], wei_plain_dims);
}

void matmul_core_op_t::collect_shrinked_axis_map(
        int bw_size, gt2axis_map &bw_axis_map) {
    auto ins = get_inputs()[0], wei = get_inputs()[1], out = get_outputs()[0];
    int bs_inp = get_inputs()[0]->details_.get_plain_dims().size() - 2;
    int bs_wei = get_inputs()[1]->details_.get_plain_dims().size() - 2;
    int bs_out = get_outputs()[0]->details_.get_plain_dims().size() - 2;
    auto get_idx = [](const graph_tensor_ptr &gt) {
        std::vector<int> batch;
        auto fmt = gt->details_.get_format();
        auto p2b_map = fmt.format_code_.collect_p2b_mapping();
        for (size_t i = 0; i < p2b_map.size() - 2; i++) {
            batch.insert(batch.end(), p2b_map[i].begin(), p2b_map[i].end());
        }
        std::vector<std::vector<int>> ret;
        ret.emplace_back(batch);
        ret.emplace_back(p2b_map[p2b_map.size() - 2]);
        ret.emplace_back(p2b_map[p2b_map.size() - 1]);
        return ret;
    };

    auto BMK = get_idx(ins), BKN = get_idx(wei), BMN = get_idx(out);

    auto get_idx_type = [](const std::vector<std::vector<int>> &map, int idx) {
        for (size_t i = 0; i < map.size(); i++) {
            if (std::find(map[i].begin(), map[i].end(), idx) != map[i].end())
                return static_cast<int>(i);
        }
        assert(0); // should never goto here
        return -1;
    };
    std::vector<int> BMK_idx, BKN_idx;
    for (int i = 0; i < bw_size; i++) {
        int idx_type = get_idx_type(BMN, i);
        if (idx_type == 0) {
            auto find_iter = std::find(BMN[0].begin(), BMN[0].end(), i);
            int batch_idx = std::distance(BMN[0].begin(), find_iter);
            // reversed position
            int batch_idx_rev = bs_out - batch_idx;
            if (batch_idx_rev <= bs_inp) {
                BMK_idx.emplace_back(BMK[idx_type][bs_inp - batch_idx_rev]);
            } else {
                BMK_idx.emplace_back(-1);
            }
            if (batch_idx_rev <= bs_wei) {
                BKN_idx.emplace_back(BKN[idx_type][bs_wei - batch_idx_rev]);
            } else {
                BKN_idx.emplace_back(-1);
            }
        } else if (idx_type == 1) {
            BMK_idx.emplace_back(BMK[idx_type][0]);
            BKN_idx.emplace_back(-1);
        } else if (idx_type == 2) {
            BMK_idx.emplace_back(-1);
            BKN_idx.emplace_back(BKN[idx_type][0]);
        }
    }

    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, ins, BMK_idx);
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, wei, BKN_idx);
    op_traits::batchwise_shrinkable_t::record_shrinked_axis(
            bw_axis_map, out, bw_size);
}

void matmul_core_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);

    // assume input is known
    if (known_ranges_map[0].empty() && known_ranges_map[1].empty()) {
        stat_map.append_ops_by_status(this, infer_status_code::RETRY);
        return;
    }

    auto inp_plain_size = get_inputs()[0]->details_.get_plain_dims().size(),
         wei_plain_size = get_inputs()[1]->details_.get_plain_dims().size();
    auto &graph = get_owner_graph();
    auto inp_dims = get_inputs()[0]->details_.get_blocking_dims();
    auto wei_dims = get_inputs()[1]->details_.get_blocking_dims();
    auto inp_dims_expr
            = get_inputs()[0]->details_.get_blocking_dims_expr(graph);
    auto wei_dims_expr
            = get_inputs()[1]->details_.get_blocking_dims_expr(graph);
    auto out_dims_expr
            = get_outputs()[0]->details_.get_blocking_dims_expr(graph);

    auto batch_dims_size = batch_dims_.size();

    slice_range inp_slice, wei_slice, out_slice;
    blocking_axis_t blocking_axis
            = get_mm_blocking_axis(get_inputs()[0]->details_,
                    get_inputs()[1]->details_, get_outputs()[0]->details_);
    if (!known_ranges_map[0].empty()) {
        if (known_ranges_map[0].size() > 1) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        inp_slice = known_ranges_map[0][0];
        // check whether do M-axis fusion
        bool M_axis_fuse = false;
        if (get_inputs()[0]->details_.get_format().is_blocking()
                && (blocking_axis.A_m.size() == blocking_axis.C_m.size())) {
            auto ctx = stat_map.get_context();
            if (ctx && ctx->flags_.use_cost_model_ && batch_dims_.empty()) {
                const int run_threads
                        = runtime_config_t::get().get_num_threads();
                int prod = 1;
                for (int i = 0; i < std::min(blocking_axis.A_k.front(),
                                        (*(blocking_axis.A_m.begin() + 1)));
                        i++) {
                    prod *= inp_dims[i];
                }
                if (prod != 1 && run_threads != 1) {
                    M_axis_fuse = (prod / run_threads > 8
                            || (prod % run_threads == 0
                                    && prod >= run_threads));
                }
            } else
                M_axis_fuse = true;
        }
        std::vector<int> required_axis = M_axis_fuse
                ? std::vector<int> {blocking_axis.A_m.begin() + 1,
                        blocking_axis.A_m.end()}
                : blocking_axis.A_m;
        required_axis.insert(required_axis.end(), blocking_axis.A_k.begin(),
                blocking_axis.A_k.end());
        // Currently, support fuse batch dims and outer-most m_o
        if (!slice_full_on_axis(inp_dims, inp_slice, required_axis)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }
    if (!known_ranges_map[1].empty()) {
        if (known_ranges_map[1].size() > 1) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        wei_slice = known_ranges_map[1][0];
        auto required_axis = blocking_axis.B_k;
        required_axis.insert(required_axis.end(), blocking_axis.B_n.begin(),
                blocking_axis.B_n.end());
        // Currently, only fuse batch dims
        if (!slice_full_on_axis(wei_dims, wei_slice, required_axis)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }

    if (!known_ranges_map[0].empty() && known_ranges_map[1].empty()) {
        // implicit broadcast semantic
        if (inp_plain_size < wei_plain_size) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        auto wei_size = wei_dims_expr.size();
        wei_slice.resize(wei_size);
        int bs_cnt = 0;
        for (int64_t i = static_cast<int64_t>(wei_size) - 1; i >= 0; i--) {
            if (std::find(
                        blocking_axis.B_bs.begin(), blocking_axis.B_bs.end(), i)
                    != blocking_axis.B_bs.end()) {
                int bs_idx_inp
                        = blocking_axis
                                  .A_bs[blocking_axis.A_bs.size() - 1 - bs_cnt];
                // explicit broadcast semantic
                if (inp_dims[bs_idx_inp] < wei_dims[i]) {
                    stat_map.append_ops_by_status(
                            this, infer_status_code::RETRY);
                    return;
                } else if (inp_dims[bs_idx_inp] == wei_dims[i]) {
                    wei_slice[i] = inp_slice[bs_idx_inp];
                } else {
                    COMPILE_ASSERT(
                            wei_dims[i] == 1, "broadcast weight is expected")
                    wei_slice[i] = std::make_pair(expr(0), expr(1));
                }
                bs_cnt++;
            } else {
                wei_slice[i] = std::make_pair(expr(0), wei_dims_expr[i]);
            }
        }
    }
    if (known_ranges_map[0].empty() && !known_ranges_map[1].empty()) {
        // implicit broadcast semantic
        if (inp_plain_size > wei_plain_size) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        auto inp_size = inp_dims_expr.size();
        inp_slice.resize(inp_size);
        int bs_cnt = 0;
        for (int64_t i = static_cast<int64_t>(inp_size) - 1; i >= 0; i--) {
            if (std::find(
                        blocking_axis.A_bs.begin(), blocking_axis.A_bs.end(), i)
                    != blocking_axis.A_bs.end()) {
                int bs_idx_wei
                        = blocking_axis
                                  .B_bs[blocking_axis.B_bs.size() - 1 - bs_cnt];
                // explicit broadcast semantic
                if (wei_dims[bs_idx_wei] < inp_dims[i]) {
                    stat_map.append_ops_by_status(
                            this, infer_status_code::RETRY);
                    return;
                } else if (wei_dims[bs_idx_wei] == inp_dims[i]) {
                    inp_slice[i] = wei_slice[bs_idx_wei];
                } else {
                    COMPILE_ASSERT(
                            inp_dims[i] == 1, "broadcast input is expected")
                    inp_slice[i] = std::make_pair(expr(0), expr(1));
                }
                bs_cnt++;
            } else {
                inp_slice[i] = std::make_pair(expr(0), inp_dims_expr[i]);
            }
        }
    }

    // set output slice
    auto ref_slice = (inp_plain_size >= wei_plain_size) ? inp_slice : wei_slice;
    auto ref_bs_axis = (inp_plain_size >= wei_plain_size) ? blocking_axis.A_bs
                                                          : blocking_axis.B_bs;
    auto out_size = out_dims_expr.size();
    out_slice.resize(out_size);
    int bs_cnt = 0;
    int m_cnt = 0;
    for (size_t i = 0; i < out_size; i++) {
        if (std::find(blocking_axis.C_bs.begin(), blocking_axis.C_bs.end(), i)
                != blocking_axis.C_bs.end()) {
            out_slice[i] = ref_slice[ref_bs_axis[bs_cnt]];
            bs_cnt++;
        } else if (std::find(blocking_axis.C_m.begin(), blocking_axis.C_m.end(),
                           i)
                != blocking_axis.C_m.end()) {
            if (blocking_axis.A_m.size() == blocking_axis.C_m.size()) {
                out_slice[i] = inp_slice[blocking_axis.A_m[m_cnt]];
                m_cnt++;
            } else {
                out_slice[i] = std::make_pair(expr(0), out_dims_expr[i]);
            }
        } else {
            out_slice[i] = std::make_pair(expr(0), out_dims_expr[i]);
        }
    }

    fsmap.get(get_inputs()[0]) = slice_range_list {inp_slice};
    fsmap.get(get_inputs()[1]) = slice_range_list {wei_slice};
    fsmap.get(get_outputs()[0]) = slice_range_list {out_slice};
}

void infer_matmul_binding_axis(tunable_op_t *cur, bound_axis_map &bdax_map) {
    // search known axis from any input of cur fusbile op
    auto known_axis_map = search_known_bound_axis(cur, bdax_map);

    bound_axis &inp_axis = known_axis_map[0], &wei_axis = known_axis_map[1],
               &out_axis = bdax_map.get(cur->get_outputs()[0]);
    if (!out_axis.empty()) return;
    auto inp_plain_dims = cur->get_inputs()[0]->details_.get_plain_dims(),
         wei_plain_dims = cur->get_inputs()[1]->details_.get_plain_dims(),
         out_plain_dims = cur->get_outputs()[0]->details_.get_plain_dims();
    // if input is known
    if (!inp_axis.empty()) {
        for (auto &bd_axis : inp_axis) {
            std::vector<int> ret_w, ret_o;
            for (auto &ax : bd_axis) {
                COMPILE_ASSERT(ax < static_cast<int64_t>(inp_plain_dims.size()),
                        "matmul core input binded axis could not exceed "
                        "plain dims size: "
                                << inp_plain_dims.size() << ", but got " << ax)
                int distance_from_right
                        = static_cast<int64_t>(inp_plain_dims.size()) - 1 - ax;
                // bind weight axis
                if (distance_from_right == 0) {
                    ret_w.emplace_back(wei_plain_dims.size() - 2);
                } else if (distance_from_right > 1
                        && distance_from_right
                                < static_cast<int64_t>(wei_plain_dims.size())) {
                    ret_w.emplace_back(
                            wei_plain_dims.size() - 1 - distance_from_right);
                }
                // bind output axis
                if (distance_from_right == 1) {
                    ret_o.emplace_back(out_plain_dims.size() - 2);
                } else if (distance_from_right > 1) {
                    ret_o.emplace_back(
                            out_plain_dims.size() - 1 - distance_from_right);
                }
            }
            wei_axis.emplace_back(ret_w);
            out_axis.emplace_back(ret_o);
        }
    }
    // if weight is known, input is unknown
    else {
        for (auto &bd_axis : wei_axis) {
            std::vector<int> ret_i, ret_o;
            for (auto &ax : bd_axis) {
                COMPILE_ASSERT(ax < static_cast<int64_t>(wei_plain_dims.size()),
                        "matmul core weight binded axis could not exceed "
                        "plain dims size: "
                                << wei_plain_dims.size() << ", but got " << ax)
                int distance_from_right
                        = static_cast<int64_t>(wei_plain_dims.size()) - 1 - ax;
                // bind input axis
                if (distance_from_right == 1) {
                    ret_i.emplace_back(inp_plain_dims.size() - 1);
                } else if (distance_from_right > 1
                        && distance_from_right
                                < static_cast<int64_t>(inp_plain_dims.size())) {
                    ret_i.emplace_back(
                            inp_plain_dims.size() - 1 - distance_from_right);
                }
                // bind output axis
                if (distance_from_right == 0) {
                    ret_o.emplace_back(out_plain_dims.size() - 1);
                } else if (distance_from_right > 1) {
                    ret_o.emplace_back(
                            out_plain_dims.size() - 1 - distance_from_right);
                }
            }
            inp_axis.emplace_back(ret_i);
            out_axis.emplace_back(ret_o);
        }
    }
    set_unknown_axis_binding(cur, known_axis_map, bdax_map);
}

void pre_matmul_binding_axis(tunable_op_t *cur, bound_axis_map &bdax_map) {
    auto &outaxis = bdax_map.get(cur->get_outputs()[0]);
    COMPILE_ASSERT(!outaxis.empty(),
            "Unknown output axis found, could not pre bind axis")

    auto inp_plain_dims = cur->get_inputs()[0]->details_.get_plain_dims(),
         wei_plain_dims = cur->get_inputs()[1]->details_.get_plain_dims(),
         out_plain_dims = cur->get_outputs()[0]->details_.get_plain_dims();

    for (size_t i = 0; i < cur->get_inputs().size(); i++) {
        auto &input = cur->get_inputs()[i];
        auto &inpaxis = bdax_map.get(input);
        auto in_plain_dims = (i == 0) ? inp_plain_dims : wei_plain_dims;
        if (inpaxis.empty()) {
            bound_axis in_axis;
            for (auto &bd_axis : outaxis) {
                std::vector<int> ret;
                for (auto &ax : bd_axis) {
                    COMPILE_ASSERT(
                            ax < static_cast<int64_t>(out_plain_dims.size()),
                            "matmul core output binded axis could not exceed "
                            "plain dims size: "
                                    << out_plain_dims.size() << ", but got "
                                    << ax)
                    int distance_from_right
                            = static_cast<int64_t>(out_plain_dims.size()) - 1
                            - ax;
                    // bind input/weight axis
                    if (distance_from_right == (1 - static_cast<int64_t>(i))) {
                        ret.emplace_back(
                                in_plain_dims.size() - 1 - distance_from_right);
                    } else if (distance_from_right > 1
                            && distance_from_right < static_cast<int64_t>(
                                       in_plain_dims.size())) {
                        ret.emplace_back(
                                in_plain_dims.size() - 1 - distance_from_right);
                    }
                }
                in_axis.emplace_back(ret);
            }
            inpaxis = in_axis;
            if (auto bd_op = input->producer_owner_->dyn_cast<
                             op_traits::mixed_partition_acceptable>()) {
                bd_op->pre_binding_axis(bdax_map);
            }
        }
    }
}

void matmul_core_op_t::infer_binding_axis(bound_axis_map &bdax_map) {
    infer_matmul_binding_axis(this, bdax_map);
}

void matmul_core_op_t::pre_binding_axis(bound_axis_map &bdax_map) {
    pre_matmul_binding_axis(this, bdax_map);
}

} // namespace ops
OP_REGISTER(ops::matmul_core_op_t, matmul_core)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
