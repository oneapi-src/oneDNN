/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_MEMORY_MOVEMENT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_MEMORY_MOVEMENT_HPP

#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {

/**
 * Transpose the input tensor
 * Inputs:
 *  - A single tensor to transpose
 * Outputs:
 *  - The transposed tensor
 * Attrs:
 *  - order: vector<int> - order of the input axes w.r.t output axes
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
    bool try_penetrate(sc_data_format_t &new_output_format) const;
    sc_dims get_bwise_fuse_shrink_dims() override;
    sc_op_ptr bw_shrinked_copy(
            gt2gt_map &bw_lt_map, sc_graph_t &shrinked_graph) override;

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
    std::vector<int> get_impl_dispatch_candidates() const override;
    bool check_padding() const;
    bool use_output_loop() const;
    bool support_output_loop() const;
    sc_dims get_bwise_fuse_shrink_dims() override;
    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;
    void collect_shrinked_axes_map(
            int bw_size, gt2axes_map &bw_axes_map) override;

private:
    sc_dims plain_dims_;
};
} // namespace sc
#endif
