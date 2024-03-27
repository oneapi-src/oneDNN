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
#include "managed_matmul_core.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "matmul_core.hpp"
#include "templates/managed_matmul_core.hpp"
#include "templates/utils.hpp"
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/graph_map.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/quantization/quantize_info.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/dead_func_eliminate.hpp>
#include <runtime/config.hpp>
#include <unordered_set>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
static sc_data_type_t infer_out_dtype(
        const std::vector<graph_tensor_ptr> &ins) {
    if (ins.at(0)->details_.dtype_ == datatypes::u8
            || ins.at(0)->details_.dtype_ == datatypes::s8) {
        assert(ins.at(1)->details_.dtype_ == datatypes::s8);
        return datatypes::s32;
    }
    return datatypes::f32;
}

managed_matmul_core_op_t::managed_matmul_core_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : tunable_op_t("managed_matmul_core", ins, outs, attrs) {
    COMPILE_ASSERT(
            info_.inputs_.size() == 2, "managed_matmul_core expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() == 2 && B_dims.size() == 2,
            "managed_matmul_core only supports 2d cases yet");
    sc_dims expected_out_shape = {merge_vec(get_batch_dims(),
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

std::vector<int> managed_matmul_core_op_t::query_prefetch(
        const context_ptr &ctx, bool is_global,
        const std::vector<tensor_slice> &ins) {
    auto gen = create_generator();
    auto gen_ptr = static_cast<gen_managed_matmul_core_t *>(gen.get());
    if (gen_ptr->is_okay_to_prefetch(ctx,
                *config_data_.get_as<managed_matmul_core_config_t>(),
                is_global)) {
        return {1};
    } else {
        return {};
    }
}

void managed_matmul_core_op_t::generate_prefetcher_body_for_tensor(
        const context_ptr &ctx, const std::vector<expr> &func_args,
        const std::vector<expr> &ins, const std::vector<int> &indices) {
    auto gen = create_generator();
    static_cast<gen_managed_matmul_core_t *>(gen.get())
            ->generate_prefetcher_body_for_tensor(ctx,
                    *config_data_.get_as<managed_matmul_core_config_t>(),
                    func_args, ins, indices);
}

body_generator_ptr managed_matmul_core_op_t::create_generator() {
    COMPILE_ASSERT(
            info_.inputs_.size() == 2, "managed_matmul_core expects 2 inputs");
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    COMPILE_ASSERT(A_dims.size() == 2 && B_dims.size() == 2,
            "managed_matmul_core only supports 2d cases yet");

    auto num_threads = runtime_config_t::get().get_num_threads();
    sc_dim M = A_dims.front(); // A is always 2D
    sc_dim K = A_dims.back(); // A is always 2D
    sc_dim N = B_dims.back(); // B is always 2D
    bool is_valid_int8
            = (info_.inputs_[0]->details_.dtype_ == datatypes::u8
                      || info_.inputs_[0]->details_.dtype_ == datatypes::s8)
            && info_.inputs_[1]->details_.dtype_ == datatypes::s8;
    if (!is_dynamic() && M <= 5 && K >= 4096
            && N >= 4096 // TODO(niuxiaoguang): K, N shapes are from gpt-j-6B
            // and llama on SPR. Change them when necessary.
            && num_threads <= 32 && is_valid_int8) {
        attrs_["dispatch_avx"] = true;
    }

    auto mat_gen = utils::make_unique<gen_managed_matmul_core_t>(this,
            graph::extract_detail_from_tensors(get_inputs()),
            graph::extract_detail_from_tensors(get_outputs()));
    auto mat_ptr = static_cast<gen_managed_matmul_core_t *>(mat_gen.get());
    if (iim_block_ != -1) { mat_ptr->iim_block_ = iim_block_; }
    if (iin_block_ != -1) { mat_ptr->iin_block_ = iin_block_; }
    if (iik_block_ != -1) { mat_ptr->iik_block_ = iik_block_; }
    if (is_dynamic()) {
        mat_ptr->is_partial_ = static_cast<bool>(info_.cur_impl_);
    }
    return std::move(mat_gen);
}

float managed_matmul_core_op_t::get_gflop() {
    return create_generator()->get_gflop();
}

sc_dims managed_matmul_core_op_t::get_batch_dims() const {
    auto &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    auto &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    return A_dims.size() > B_dims.size()
            ? sc_dims {A_dims.begin(), A_dims.end() - 2}
            : sc_dims {B_dims.begin(), B_dims.end() - 2};
}

void managed_matmul_core_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    const sc_dims &A_dims = info_.inputs_[0]->details_.get_plain_dims();
    const sc_dims &A_blocking_dims
            = info_.inputs_[0]->details_.get_blocking_dims();
    const sc_dims &B_dims = info_.inputs_[1]->details_.get_plain_dims();
    const sc_dims &B_blocking_dims
            = info_.inputs_[1]->details_.get_blocking_dims();
    const sc_dims &C_dims = info_.outputs_[0]->details_.get_plain_dims();
    const sc_dim M = A_dims[A_dims.size() - 2];
    const sc_dim K = A_dims.back();
    const sc_dim N = B_dims.back();

    bool dynamic = is_dynamic();
    in_formats.reserve(2);
    sc_data_type_t A_dtype = info_.inputs_[0]->details_.dtype_;
    sc_data_type_t B_dtype = info_.inputs_[1]->details_.dtype_;
    sc_data_format_t A_format = info_.inputs_[0]->details_.get_format();
    sc_data_format_t B_format = info_.inputs_[1]->details_.get_format();
    bool is_A_vnni_low_fp = ops::is_vnni_low_fp(ctx, A_dtype);
    bool is_B_vnni_low_fp = ops::is_vnni_low_fp(ctx, B_dtype);
    auto gen_ptr = create_generator();
    auto gen = static_cast<gen_managed_matmul_core_t *>(gen_ptr.get());
    int iim_block = gen->iim_block_;
    int iin_block = gen->iin_block_;
    int iik_block = gen->iik_block_;
    if (!config_data_) {
        if (!dynamic && attrs_.get_or_else("transposed_a", false)) {
            config_data_ = gen->get_default_transposed_a_config(ctx);
        } else {
            auto post_rd_axis
                    = attrs_.get_or_else("post_rd_axis", std::vector<int> {});
            if (post_rd_axis.size() == 1 && post_rd_axis.at(0) == 1) {
                // mmm + reduce_on_N cases
                config_data_ = gen->get_default_post_rd_config(ctx);
            } else {
                config_data_ = create_generator()->get_default_config(ctx);
            }
        }
    }

    // constant check
    bool constant_A = false, constant_B = false;
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

    std::vector<int> blk_candidates = get_dynamic_block_candidates();
    std::vector<int> m_blk_candidates = get_dynamic_batch_block_candidates();
    bool transposed_a = attrs_.get_or_else("transposed_a", false);
    bool transposed_b = attrs_.get_or_else("transposed_b", false);
    sc_data_format_t ret_A_format, ret_B_format, ret_C_format;
    auto cur_format_set = std::unordered_set<std::vector<sc_data_format_t>>();
    auto cur_dispatch_key_set = dispatch_key_set_t();
    std::vector<bool> is_padding = {false, true};
    std::vector<bool> is_output_plain = {false, true};
    bool first = true;
    // consider ND*2D case, A_format is penetrated, which is always blocking
    auto p2bmp_a = A_format.format_code_.collect_p2b_mapping();
    // consider 2D*ND case, B_format is penetrated, which is always blocking
    auto p2bmp_b = B_format.format_code_.collect_p2b_mapping();
    bool treat_as_static
            = !get_owner_graph().attrs_.get_or_else("insert_reorder", true);
    for (auto &m_b : m_blk_candidates) { // M
        for (auto &n_b : blk_candidates) { // N
            for (auto &k_b : blk_candidates) { // K
                for (auto A_isp : is_padding) { // A is_padding
                    for (auto B_isp : is_padding) { // B is_padding
                        if (is_dynamic_dim(M)) {
                            iim_block = m_b;
                            if (treat_as_static) { iim_block = iim_block_; }
                        }
                        if (is_dynamic_dim(N)) {
                            iin_block = n_b;
                            if (treat_as_static) { iin_block = iin_block_; }
                        }
                        if (is_dynamic_dim(K)) {
                            iik_block = k_b;
                            if (treat_as_static) { iik_block = iik_block_; }
                        }
                        if (A_dims.size() == 2) {
                            if (constant_A
                                    || (!dynamic && A_format.is_blocking()
                                            && p2bmp_a.at(0).size() > 1
                                            && p2bmp_a.at(1).size() > 1)
                                    || A_isp || (!dynamic && M % iim_block)
                                    || (!dynamic && K % iik_block)
                                    || (!dynamic && is_A_vnni_low_fp
                                            && A_format
                                                    == sc_data_format_t::
                                                            NK())) {
                                ret_A_format = sc_data_format_t::MKmk(
                                        iim_block, iik_block);
                            } else {
                                ret_A_format = sc_data_format_t::MK();
                            }
                            if (dynamic && A_format.is_blocking()
                                    && p2bmp_a.at(0).size() > 1
                                    && p2bmp_a.at(1).size() > 1) {
                                ret_A_format = A_format;
                                iim_block = transposed_a ? A_format.blocks_[1]
                                                         : A_format.blocks_[0];
                                iik_block = transposed_a ? A_format.blocks_[0]
                                                         : A_format.blocks_[1];
                            }
                            if (!dynamic) {
                                in_formats.push_back({ret_A_format});
                            }
                        } else {
                            COMPILE_ASSERT(0,
                                    "managed_matmul_core only supports 2d "
                                    "yet");
                        }
                        if (B_dims.size() == 2) {
                            if (utils::is_one_of(B_dtype, datatypes::u8,
                                        datatypes::s8)) {
                                ret_B_format = sc_data_format_t::NKkn4k(
                                        iik_block, iin_block);
                            } else if (is_B_vnni_low_fp) {
                                // do vnni reorder in template for
                                // transposed matmul
                                if (!dynamic) {
                                    ret_B_format = sc_data_format_t::NKkn2k(
                                            iik_block, iin_block);
                                }
                            } else {
                                if (constant_B || B_isp
                                        || (!dynamic && B_format.is_blocking()
                                                && p2bmp_b.at(0).size() > 1
                                                && p2bmp_b.at(1).size() > 1)
                                        || (!is_dynamic_dim(K) && K % iik_block)
                                        || (!is_dynamic_dim(N)
                                                && N % iin_block)) {
                                    ret_B_format = sc_data_format_t::NKkn(
                                            iik_block, iin_block);
                                } else {
                                    ret_B_format = sc_data_format_t::KN();
                                }
                            }
                            if (!dynamic) {
                                in_formats.push_back({ret_B_format});
                            }
                        } else {
                            COMPILE_ASSERT(0,
                                    "managed_matmul_core only supports 2d "
                                    "yet");
                        }
                        if (M == iim_block && M >= 32 && N % iin_block == 0
                                && !dynamic) {
                            out_formats.push_back(
                                    {sc_data_format_t::get_plain_by_dims(
                                            C_dims.size())});
                        } else if (constant_B || dynamic || M % iim_block
                                || N % iin_block) {
                            ret_C_format = sc_data_format_t(
                                    sc_data_format_kind_t::
                                            get_2dblocking_by_dims(
                                                    C_dims.size(), false),
                                    {iim_block, iin_block});
                            if (!dynamic) {
                                out_formats.push_back({ret_C_format});
                            }
                        } else {
                            if (attrs_.get_or_else("transposed_b", false)) {
                                out_formats.push_back(
                                        {sc_data_format_t::get_plain_by_dims(
                                                C_dims.size())});
                            } else {
                                sc_data_format_t out_fmt1(
                                        sc_data_format_kind_t::
                                                get_2dblocking_by_dims(
                                                        C_dims.size(), false),
                                        {iim_block, iin_block});
                                auto out_fmt2
                                        = sc_data_format_t::get_plain_by_dims(
                                                C_dims.size());
                                out_formats.push_back({out_fmt1, out_fmt2});
                                in_formats[0].push_back(in_formats[0][0]);
                                in_formats[1].push_back(in_formats[1][0]);
                            }
                        }
                        std::vector<sc_data_format_t> ret_formats
                                = {ret_A_format, ret_B_format, ret_C_format};
                        if (dynamic) {
                            std::vector<std::vector<sc_dim>> var_block
                                    = {{iim_block, iik_block},
                                            {iik_block, iin_block},
                                            {iim_block, iin_block}};
                            op_dispatch_key_t ret_key(var_block, ret_formats);
                            cur_dispatch_key_set.set_.insert(ret_key);
                            if (cur_format_set.find(ret_formats)
                                    == cur_format_set.end()) {
                                if (in_formats.empty()) {
                                    in_formats.resize(2);
                                }
                                if (out_formats.empty()) {
                                    out_formats.resize(1);
                                }
                                in_formats[0].emplace_back(ret_A_format);
                                in_formats[1].emplace_back(ret_B_format);
                                out_formats[0].emplace_back(ret_C_format);
                                cur_format_set.insert(ret_formats);
                            }
                            if (first) {
                                // reset default cfg to first candidate in
                                // dynamic for try mode in fuse op pass.
                                iim_block_ = iim_block;
                                iin_block_ = iin_block;
                                iik_block_ = iik_block;
                                first = false;
                            }
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
    auto pad_K_num = utils::divide_and_ceil(
            info_.inputs_[0]->details_.get_plain_dims().back(), iik_block);
    attrs_["temp.padded_A_K"].get<std::shared_ptr<VConst>>()->var_
            = pad_K_num * iik_block;
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void managed_matmul_core_op_t::set_config_by_key(
        const op_dispatch_key_t &key, const context_ptr &ctx) {
    iim_block_ = key.var_block_[0][0];
    iin_block_ = key.var_block_[1][1];
    iik_block_ = key.var_block_[0][1];
}

void managed_matmul_core_op_t::set_internal_config_by_key(
        const impl_op_dispatch_key_t &key, const context_ptr &ctx) {
    config_data_ = dyn_config_candidates_[key.impl_];
    auto mmm_config = config_data_.get_as<managed_matmul_core_config_t>();
    if (mmm_config->M_split_num * mmm_config->N_split_num
            < runtime_config_t::get().get_num_threads()) {
        info_.cur_impl_ = mmm_impl_kind_t::is_partial;
    } else {
        info_.cur_impl_ = mmm_impl_kind_t::full_k;
    }
}

ir_module_ptr managed_matmul_core_op_t::get_internal_func(
        const context_ptr &ctx) {
    assert(is_dynamic());
    if (!need_dynamic_internal_query()) { return nullptr; }
    // query binding axis
    query_binding_axis(get_owner_graph());
    auto ret = std::make_shared<ir_module_t>(ctx);
    auto gen_ptr = create_generator();
    std::vector<expr> ins;
    std::vector<expr> outs;
    auto func = graph::create_func_decl_for_op(this, ins, outs);
    COMPILE_ASSERT(!info_.internal_info_->parti_in_ltsrs_.empty()
                    && !info_.internal_info_->parti_out_ltsrs_.empty(),
            "Need in/out buffer args first");
    const auto &out_details = info_.cur_impl_ == mmm_impl_kind_t::is_partial
            ? graph::extract_detail_from_tensors(get_outputs())
            : info_.internal_info_->parti_out_ltsrs_;
    const auto &in_details = info_.cur_impl_ == mmm_impl_kind_t::is_partial
            ? graph::extract_detail_from_tensors(get_inputs())
            : info_.internal_info_->parti_in_ltsrs_;
    auto pouts = graph::tensor_detail_to_ir_tensor(
            get_owner_graph(), "__outs_", out_details);
    auto pins = graph::tensor_detail_to_ir_tensor(
            get_owner_graph(), "__ins_", in_details);
    auto buffer_args = pouts;
    buffer_args.insert(buffer_args.end(), pins.begin(), pins.end());
    func->params_ = buffer_args;
    func->params_.emplace_back(
            builder::make_var(datatypes::pointer, "single_core_func"));
    builder::ir_builder_t bld;
    bld.push_scope();
    std::vector<for_loop> loops;
    gen_ptr->set_single_core_func_param(func->params_.back());
    func_t single_core_func = gen_ptr->get_single_core_func(
            ctx, config_data_.data_.get(), nullptr, pins, pouts, loops);
    single_core_func->body_ = builder::make_returns_unattached(true);
    single_core_func->attr().set(attr_keys::keep_func, true);
    auto extra_args = gen_ptr->get_extra_args_from_func(single_core_func);
    single_core_func->params_ = buffer_args;
    single_core_func->params_.insert(single_core_func->params_.end(),
            extra_args.begin(), extra_args.end());
    func->params_.back()->attr().set("prototype", single_core_func);
    bool status = gen_ptr->generate(
            ctx, config_data_.data_.get(), nullptr, pins, pouts, loops);
    assert(status);
    bld.push_returns(true);
    auto body = bld.pop_scope();
    gen_ptr->schedule_loops(ctx, config_data_.data_.get(), body, loops);
    func->body_ = std::move(body);
    ret->add_func({func, single_core_func});
    ret->set_entry_func_idx(0);
    return ret;
}

sc_op_ptr managed_matmul_core_op_t::copy(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ret = tunable_op_t::copy(ins, outs, mgr);
    auto mmm = ret->dyn_cast<managed_matmul_core_op_t>();
    mmm->iim_block_ = iim_block_;
    mmm->iin_block_ = iin_block_;
    mmm->iik_block_ = iik_block_;
    return ret;
}

sc_op_ptr managed_matmul_core_op_t::do_compensations(
        sc_graph_t &mgr, const context_ptr &ctx) {
    need_compensation_ = false;
    // whether we need special compensation for
    // microkernel.
    bool s8s8_compensation = ctx->machine_.cpu_flags_.fAVX512VNNI
            && info_.inputs_[0]->details_.dtype_ == datatypes::s8
            && (!ctx->machine_.brgemm_use_amx_
                    || (ctx->machine_.brgemm_use_amx_
                            && !ctx->machine_.cpu_flags_.fAVX512AMXINT8)
                    || attrs_.get_or_else("dispatch_avx", false));
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

sc_op_ptr managed_matmul_core_op_t::get_data_compensation(sc_graph_t &mgr) {
    bool is_dyn_quan = attrs_.has_key(attr_keys::dyn_weight_zero_points);
    auto weight_zero_points = attrs_.get_or_else(
            attr_keys::weight_zero_points, std::vector<int> {0});
    auto dyn_weight_zero_points = attrs_.get_or_else(
            attr_keys::dyn_weight_zero_points, graph_tensor_ptr());
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

    if (data->details_.get_plain_dims().size() < get_batch_dims().size() + 2) {
        sc_dims unsqueeze_shape(get_batch_dims().size() + 2
                        - data->details_.get_plain_dims().size(),
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

std::vector<sc_op_ptr>
managed_matmul_core_op_t::get_s8s8_and_weight_compensation(
        sc_graph_t &mgr, bool s8s8_compensation) {
    bool is_dyn_quan = attrs_.has_key(attr_keys::dyn_data_zero_points);
    auto data_zero_points = attrs_.get_or_else(
            attr_keys::data_zero_points, std::vector<int> {0});
    auto dyn_data_zero_points = attrs_.get_or_else(
            attr_keys::dyn_data_zero_points, graph_tensor_ptr());
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
                size_t bds = get_batch_dims().size();
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
        if (weight->details_.get_plain_dims().size()
                < get_batch_dims().size() + 2) {
            sc_dims unsqueeze_shape(get_batch_dims().size() + 2
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
        if (weight->details_.get_plain_dims().size()
                < get_batch_dims().size() + 2) {
            sc_dims unsqueeze_shape(get_batch_dims().size() + 2
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

sc_op_ptr managed_matmul_core_op_t::get_constant_compensation(sc_graph_t &mgr) {
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

    if (is_dyn_quan) {
        COMPILE_ASSERT(
                dyn_data_zero_points->details_.get_plain_dims() == sc_dims {1}
                        && dyn_weight_zero_points->details_.get_plain_dims()
                                == sc_dims {1},
                "matmul_core does not support per channel data/weight zero "
                "points compensation yet");
        ret_node = mgr.make(
                "mul", {dyn_data_zero_points, dyn_weight_zero_points}, {}, {});
        auto const_reduce = mgr.make("constant", {}, {},
                {{"dtype", datatypes::s32},
                        {"values",
                                std::make_shared<static_data_t>(
                                        &K, sizeof(int))},
                        {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()}, {"temp.val/var", 1},
                        {"temp.var", attrs_["temp.padded_A_K"]}});
        ret_node = mgr.make("mul",
                {ret_node->get_outputs()[0], const_reduce->get_outputs()[0]},
                {}, {});
    } else {
        COMPILE_ASSERT(
                data_zero_points.size() == 1 && weight_zero_points.size() == 1,
                "matmul_core does not support per channel data/weight zero "
                "points "
                "compensation yet");
        ret_node = mgr.make("constant", {}, {},
                {{"values",
                         std::make_shared<static_data_t>(
                                 std::vector<int> {data_zero_points[0]
                                         * weight_zero_points[0] * K})},
                        {"dtype", datatypes::s32}, {"plain_dims", sc_dims {1}},
                        {"format", sc_data_format_t()},
                        {"temp.val/var",
                                data_zero_points[0] * weight_zero_points[0]},
                        {"temp.var", attrs_["temp.padded_A_K"]}});
    }

    return ret_node;
}

shape_rl_vec managed_matmul_core_op_t::get_dynamic_shape_relations() const {
    return matmul_core_op_t::get_shape_relations_impl(
            get_inputs()[0]->details_.get_plain_dims(),
            get_inputs()[1]->details_.get_plain_dims(),
            get_outputs()[0]->details_.get_plain_dims());
}

bool managed_matmul_core_op_t::need_dynamic_internal_query_impl() const {
    return is_dynamic();
}

void managed_matmul_core_op_t::infer_binding_axis(binding_axis_map &bdax_map) {
    infer_matmul_binding_axis(this, bdax_map);
}
void managed_matmul_core_op_t::pre_infer_binding_axis(
        binding_axis_map &bdax_map) {
    pre_matmul_binding_axis(this, bdax_map);
}

std::vector<int> managed_matmul_core_op_t::get_impl_dispatch_candidates(
        const context_ptr &ctx) {
    return std::vector<int> {
            mmm_impl_kind_t::full_k, mmm_impl_kind_t::is_partial};
}

dispatch_set_ptr managed_matmul_core_op_t::get_internal_dispatch_key_set(
        const context_ptr &ctx) {
    auto ret = std::make_shared<impl_dispatch_key_set_t>();
    auto impls = get_dynamic_impl_dispatch_candidates(this, ctx);
    for (auto &impl : impls) {
        ret->set_.insert(impl_op_dispatch_key_t(impl, 3));
    }
    return ret;
}
} // namespace ops
OP_REGISTER(ops::managed_matmul_core_op_t, managed_matmul_core)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
