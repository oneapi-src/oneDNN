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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_MEMORY_MOVEMENT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_MEMORY_MOVEMENT_HPP

#include <numeric>
#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class concat_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    concat_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    concat_op_t(const std::vector<graph_tensor_ptr> &candidates, int axis);

    int64_t get_axis() { return axis_; }

    // For each input tensor, concat optimization pass will try to make the
    // parent op directly write it into the output of concat; If so, we mark the
    // corresponding input as invalid (false). For the input tensor that cannot
    // be optimized, we need to generate IR to copy it into the output of
    // concat; we mark this tensor as valid (true), because this is the default
    // action of concatenating.
    std::vector<bool> is_input_valid_;
    // if returned value is false, then this concat has been optimized and some
    // inputs's strides are changed
    bool all_inputs_valid() {
        return std::all_of(is_input_valid_.begin(), is_input_valid_.end(),
                [](bool valid) { return valid; });
    }

protected:
    // axis_ is with respect to blocking format.
    int64_t axis_;
    // To make sense, the axis_ should be combined with a fixed format.
    sc_data_format_t ori_format_;
};

/**
 * Transpose the input tensor
 * Inputs:
 *  - A single tensor to transpose
 * Outputs:
 *  - The transposed tensor
 * Attrs:
 *  - order: vector<int> - order of the input axis w.r.t output axis
 * */
class transpose_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    transpose_op_t(graph_tensor_ptr v, std::vector<int> &order);

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

    shape_rl_vec get_dynamic_shape_relations() const override;

private:
    std::vector<int> order_;
};

/**
 * Creates a view of an input tensor with a different shape
 * Inputs:
 *  - A single tensor to reshape
 * Outputs:
 *  - The reshaped tensor
 * Attrs:
 *  - shape: vector<int> - the output blocking shape
 *  - format: sc_data_format_t - default: any. the format of the output logical
 *    tensor
 *  - expand_dim: bool - default: false. whether the tensor_view is a pure
 *    dimension expansion (e.g. [a, b] --> [1, 1, a, b])
 * */
class tensor_view_op_t : public movement_op_t,
                         public op_traits::auto_copyable_t,
                         public op_traits::batchwise_shrinkable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes);
    tensor_view_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    sc_dims get_shapes() const;
    std::vector<expr> get_shapes_expr();
    bool try_penetrate(sc_data_format_t &new_output_format) const;
    shape_rl_vec get_dynamic_shape_relations() const override;
    sc_dims get_bwise_fuse_shrink_dims() override;
    sc_op_ptr bw_shrinked_copy(
            gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) override;

    void infer_binding_axis(bound_axis_map &bdax_map) override;
    void pre_binding_axis(bound_axis_map &bdax_map) override;

private:
    sc_dims shapes_;
};

/**
 * Creates a copy of an input tensor with a different shape.
 * Currenly only used in case whole graph has only single reshape op for perf.
 * Inputs:
 *  - A single tensor to reshape
 * Outputs:
 *  - The reshaped tensor
 * Attrs:
 *  - shape: vector<int> - the output blocking shape
 * */
class reshape_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();
    reshape_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    ir_module_ptr get_func(context_ptr ctx) override;

private:
    sc_dims shapes_;
};

class split_op_t : public movement_op_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    split_op_t(graph_tensor_ptr v, int dim, const sc_dims &shapes);

private:
    unsigned dim_;
    sc_dims shapes_;
};

class reorder_op_t : public movement_op_t,
                     public op_traits::auto_copyable_t,
                     public op_traits::batchwise_shrinkable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    reorder_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    reorder_op_t(graph_tensor_ptr v, sc_data_format_t input_format,
            sc_data_format_t output_format);
    ir_module_ptr get_func(context_ptr ctx) override;
    const sc_data_format_kind_t &get_input_format_kind() const {
        return info_.inputs_[0]->details_.get_format().format_code_;
    }
    const sc_data_format_kind_t &get_output_format_kind() const {
        return info_.outputs_[0]->details_.get_format().format_code_;
    }
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    const sc_data_format_t &get_output_format() const {
        return info_.outputs_[0]->details_.get_format();
    }
    const sc_data_format_t &get_input_format() const {
        return info_.inputs_[0]->details_.get_format();
    }
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;
    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override;
    void update_fuse_attr();
    bool check_padding() const;
    bool use_output_loop() const;
    bool support_output_loop() const;
    bool support_optimized_kernel(const context_ptr &ctx) const;
    bool meet_vnni_reorder_require(const context_ptr &ctx) const;
    sc_dims get_bwise_fuse_shrink_dims() override;
    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;
    void collect_shrinked_axis_map(
            int bw_size, gt2axis_map &bw_axis_map) override;
    void infer_binding_axis(bound_axis_map &bdax_map) override;
    void pre_binding_axis(bound_axis_map &bdax_map) override;

private:
    sc_dims plain_dims_;
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
