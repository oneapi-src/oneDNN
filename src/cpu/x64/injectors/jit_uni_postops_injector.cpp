/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace injector {

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(jit_generator *host,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : post_ops_(post_ops)
    , host_(host)
    , binary_injector_(nullptr)
    , lambda_jit_injectors_(lambda_jit_injectors) {

    const auto &esp = eltwise_static_params;
    bool is_binary = false;

    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.emplace(post_op.eltwise.alg,
                    jit_uni_eltwise_injector_f32<isa>(host_, post_op.eltwise,
                            esp.save_state, esp.p_table, esp.k_mask, esp.is_fwd,
                            esp.use_dst));
        } else if (post_op.is_binary()) {
            is_binary = true;
        }
    }

    if (is_binary)
        binary_injector_ = utils::make_unique<
                binary_injector::jit_uni_binary_injector_t<isa>>(
                host, binary_static_params);
}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(jit_generator *host,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), lambda_jit_injectors_t()) {}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(jit_generator *host,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const lambda_jit_injectors_t &lambda_jit_injectors)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_injector::static_params_t(), lambda_jit_injectors) {}

template <cpu_isa_t isa>
jit_uni_postops_injector_t<isa>::jit_uni_postops_injector_t(jit_generator *host,
        const post_ops_t &post_ops,
        const binary_injector::static_params_t &binary_static_params,
        const eltwise_injector::static_params_t &eltwise_static_params)
    : jit_uni_postops_injector_t(host, post_ops, binary_static_params,
            eltwise_static_params, lambda_jit_injectors_t()) {}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    injector_utils::vmm_index_set_t vmm_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        vmm_idxs.emplace(i);
    compute_vector_range(vmm_idxs, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        size_t start_idx, size_t end_idx) {
    compute_vector_range(
            start_idx, end_idx, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {

    std::size_t rhs_arg_idx = 0;
    for (const auto &post_op : post_ops_.entry_) {
        if (post_op.is_eltwise()) {
            alg_to_eltwise_injector_.at(post_op.eltwise.alg)
                    .compute_vector_range(vmm_idxs);
        } else if (post_op.is_binary()) {
            binary_injector_->compute_vector_range(
                    vmm_idxs, rhs_arg_idx, post_op, rhs_arg_params);
            ++rhs_arg_idx;
        } else {
            const auto lam = lambda_jit_injectors_.find(post_op.kind);
            if (lam != lambda_jit_injectors_.end()) lam->second();
        }
    }
}
template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    compute_vector_range(vmm_idxs, binary_injector::rhs_arg_dynamic_params_t());
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::prepare_table(bool gen_table) {
    for (auto &alg_elt_inject : alg_to_eltwise_injector_)
        alg_elt_inject.second.prepare_table(gen_table);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(size_t idx,
        const binary_injector::rhs_arg_dynamic_params_t &rhs_arg_params) {
    compute_vector_range({idx}, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::compute_vector(size_t idx) {
    compute_vector_range({idx});
}

template <cpu_isa_t isa>
void jit_uni_postops_injector_t<isa>::set_lambda_injector(
        dnnl_primitive_kind_t kind, const std::function<void()> &jit_injector) {
    lambda_jit_injectors_[kind] = jit_injector;
}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops) {}

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d,
        bool sum_at_pos_0_only, const bool sum_requires_scale_one)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d)
    , sum_at_pos_0_only(sum_at_pos_0_only)
    , sum_requires_scale_one(sum_requires_scale_one) {};

post_ops_ok_args_t::post_ops_ok_args_t(const cpu_isa_t isa,
        const std::vector<post_op_type> &accepted_post_op_types,
        const post_ops_t &post_ops, const memory_desc_wrapper *dst_d)
    : isa(isa)
    , accepted_post_op_types(accepted_post_op_types)
    , post_ops(post_ops)
    , dst_d(dst_d) {}

bool post_ops_ok(const post_ops_ok_args_t &post_ops_ok_args) {
    const cpu_isa_t isa = post_ops_ok_args.isa;
    const std::vector<post_op_type> &accepted_post_op_types
            = post_ops_ok_args.accepted_post_op_types;
    const post_ops_t &post_ops = post_ops_ok_args.post_ops;
    const memory_desc_wrapper *dst_d = post_ops_ok_args.dst_d;
    const bool sum_at_pos_0_only = post_ops_ok_args.sum_at_pos_0_only;
    const bool sum_requires_scale_one = post_ops_ok_args.sum_requires_scale_one;

    const auto is_accepted_postop = [&](const int idx) {
        for (const auto &post_op : accepted_post_op_types) {
            const auto &entry = post_ops.entry_[idx];
            switch (post_op) {
                case sum:
                    if (entry.is_sum(sum_requires_scale_one))
                        return IMPLICATION(sum_at_pos_0_only, idx == 0);
                    break;
                case eltwise:
                    if (entry.is_eltwise()) return true;
                    break;
                case binary:
                    if (entry.is_binary()) {
                        assert(dst_d != nullptr && "dst_d is null");

                        bool res = IMPLICATION(entry.binary.src1_desc.data_type
                                        == data_type::bf16,
                                utils::one_of(
                                        isa, avx512_core_bf16, avx512_core));
                        res &= binary_injector::binary_args_broadcast_supported(
                                post_ops, *dst_d);

                        return res;
                    }
                    break;
                default: assert(false && "Unhandled post_op type");
            }
        }
        return false;
    };

    for (int i = 0; i < post_ops.len(); i++) {
        if (!is_accepted_postop(i)) return false;
    }

    return true;
}

template class jit_uni_postops_injector_t<avx512_core_bf16>;
template class jit_uni_postops_injector_t<avx512_core>;
template class jit_uni_postops_injector_t<avx512_common>;
template class jit_uni_postops_injector_t<avx2>;
template class jit_uni_postops_injector_t<avx>;
template class jit_uni_postops_injector_t<sse41>;

} // namespace injector
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
