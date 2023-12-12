/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAITS_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/binding_axis.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/transform/parallel_workload_attr.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class fusion_anchor_mgr_t;
struct brgemm_fusion_register;
struct mixed_parti_t;
struct fusion_anchor_t;

namespace op_traits {
struct copyable_t : public virtual op_base_trait_t {
    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr)
            = 0;
};

/**
 * A class is auto-copyable if we can construct a valid copy of the node with
 * the in/out tensors and the attrs of the node. The op name should be in the op
 * registery
 * */
struct auto_copyable_t : public copyable_t {
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
};

/**
 * @brief The util trait template for Ops that is "almost" auto-copyable except
 * that they need to copy the data in the trait.
 *
 * @tparam TArgs op_traits that has copy_from methods
 */
template <typename... TArgs>
struct auto_copyable_with_trait_t : public auto_copyable_t {
    template <typename T>
    static void copy_impl(auto_copyable_with_trait_t *from, sc_op *to) {
        auto pto = dynamic_cast<T *>(to);
        assert(pto);
        auto pfrom = dynamic_cast<T *>(from);
        assert(pfrom);
        pto->copy_from(pfrom);
    }

    template <typename T0, typename... T>
    static void copy_impl(auto_copyable_with_trait_t *from,
            typename std::enable_if<(sizeof...(T) > 0), sc_op *>::type to) {
        copy_impl<T0>(from, to);
        copy_impl<T...>(from, to);
    }
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override {
        auto ret = auto_copyable_t::copy(ins, outs, mgr);
        copy_impl<TArgs...>(this, ret.get());
        return ret;
    }
};

// the OP can be optimized if some of the inputs are constants
struct constant_optimizable_t : public virtual op_base_trait_t {
    // do optimization and return the new optmized op. If no optimization can be
    // applied, return null
    virtual sc_op_ptr constant_optimize(sc_graph_t &graph) = 0;
};

// the part of OP's workload can be computed, e.g. intrisics(brgemm), tensor
// slice.
struct workload_computable_t : public virtual op_base_trait_t {
    using shape_dtype_pair = std::pair<sc_dims, sc_data_type_t>;
    static const size_t read_weight = parallel_workload::read_weight;
    static const size_t write_weight = parallel_workload::write_weight;
    static constexpr const char *workload_number
            = parallel_workload::attr_workload_number;
    // compute workload with given input and output tensor pointers, according
    // to read/write times and operator numbers.
    virtual size_t compute_workload(const std::vector<shape_dtype_pair> &ins,
            const std::vector<shape_dtype_pair> &outs)
            = 0;
};

// the OP can accept a fusion manager to do post fusion
struct post_fusion_acceptable_t : public virtual op_base_trait_t {
    virtual ir_module_ptr get_func(context_ptr ctx,
            const std::shared_ptr<fusion_anchor_mgr_t> &fuse_mgr,
            const std::string &func_name)
            = 0;
};

// the OP can be fused into brgemm calculation.
struct brgemm_fusion_acceptable_t : public virtual op_base_trait_t {
    static constexpr const char *brgemm_fusion = "brgemm_fusion";
    bool fuse_in_brgemm_ = false;
    brgemm::alg_kind_t alg_kind_ = brgemm::alg_kind_t::alg_kind_undef;
    virtual bool register_brgemm_fusion(const context_ptr &ctx,
            const std::vector<tensor_slice *> &outputs,
            const std::vector<const tensor_slice *> &inputs,
            brgemm_fusion_register &brg_reg)
            = 0;
    void copy_from(brgemm_fusion_acceptable_t *from) {
        fuse_in_brgemm_ = from->fuse_in_brgemm_;
        alg_kind_ = from->alg_kind_;
    }
};

// quantize
struct may_quantize_t : public virtual op_base_trait_t {
    virtual sc_op_ptr do_compensations(
            sc_graph_t &mgr, const context_ptr &ctx) {
        need_compensation_ = false;
        return sc_op_ptr();
    }
    bool should_quantized_ = false;
    bool is_quantized_ = false;
    bool need_compensation_ = true;
};

struct mixed_partition_acceptable : public virtual op_base_trait_t {
    // create a new partition for current op
    virtual void create_mixed_partition(mixed_parti_t *parti) = 0;

    // append current op to the existed partition
    virtual void append_mixed_partition(mixed_parti_t *parti) = 0;

    // search fusion anchor for current op in given partition
    virtual void search_anchor(mixed_parti_t *parti) = 0;

    // commit current op into given partition
    virtual void commit_into_anchor(fusion_anchor_t *committed_anchor) = 0;

    // infer binding axis from inputs to outputs
    virtual void infer_binding_axis(binding_axis_map &bdax_map) = 0;

    // infer binding axis from outputs to inputs
    virtual void pre_infer_binding_axis(binding_axis_map &bdax_map) = 0;
};

struct data_compensation_t : public virtual op_base_trait_t {};
struct weight_compensation_t : public virtual op_base_trait_t {};
struct constant_compensation_t : public virtual op_base_trait_t {};

} // namespace op_traits

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
