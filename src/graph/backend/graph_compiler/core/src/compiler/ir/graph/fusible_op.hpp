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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/trait/may_broadcast.hpp>
#include <compiler/ir/graph/trait/may_inplace.hpp>
#include <compiler/ir/graph/traits.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

struct infer_status_map_t;
struct tensor_slice;

/**
 * A fuser will do actual code injection on the fusion point. It will be managed
 * by the fusion manager.
 * */
class fusible_op_t : public sc_op,
                     public op_traits::workload_computable_t,
                     public op_traits::mixed_partition_acceptable {
public:
    // when fusible_op_t is as a started op in the graph/subgraph, query_format
    // return certain format.
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    ir_module_ptr get_func(context_ptr ctx) override;

    // prepares the Op's anchor-irrelevant internal states for generating IR for
    // the input shapes. Also propagates the tensor slice shapes from input to
    // output in the fusion data
    virtual void prepare_fusion_data(fdata_map &fdmap) = 0;
    /**
     * 'infer_slice_ranges' is used to infer slice ranges for all fusible op in
     * fusion manger, espically helpful for input arg op, because it could not
     * know the slice information, it need to inferred by its partner input in
     * binary or trinary op, such as add/mul, etc. Additional, sometimes, the
     * slice range may be changed, e.g. reduce_op_t or movement type ops. we
     * need to address for this speciall condition. Actually,
     * 'infer_slice_ranges' can be viewed as `post_slice_ranges` and very simple
     * `pre_slice_ranges` with only one previous op inferred.
     * */
    virtual void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map)
            = 0;

    void search_anchor(mixed_parti_t *parti) override;

    /**
     * 'pre_slice_ranges' is used to infer slice ranges especially
     * for pre-op fusion. As mentioned above `infer_slice_ranges` can infer
     * slice for input arg op, however, if sometimes the input of one fusible op
     * is a sub fusion graph, it is expected to infer input slice by given
     * output slice.
     * */
    virtual void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map)
            = 0;
    /**
     * Does the actual code injection to the current func on an input
     * and output tensor slices.
     *
     * */
    virtual void compute_block(context_ptr ctx,
            const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs)
            = 0;
    /**
     * Compute workload of a fusible op during its compute block, ins and outs
     * are from tensor slice.
     * */
    size_t compute_workload(const std::vector<shape_dtype_pair> &ins,
            const std::vector<shape_dtype_pair> &outs) override;
    /**
     * A wrapper of `compute_workload`, get workload from tensor slice.
     * */
    virtual size_t compute_fusible_workload(const context_ptr &ctx,
            const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs);

    void create_mixed_partition(mixed_parti_t *parti) override;

    void append_mixed_partition(mixed_parti_t *parti) override;

    void commit_into_anchor(fuse_anchor_map_t *committed_anchor) override;

    void infer_binding_axis(bound_axis_map &bdax_map) override {}

    void pre_binding_axis(bound_axis_map &bdax_map) override {}

    ~fusible_op_t() override = default;

    int anchor_id_ = -1;
};

using fusion_op_ptr = std::shared_ptr<fusible_op_t>;

#define DECLARE_COMPUTE() \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs) override;

#define DECLARE_QUERY_AND_COMPUTE() \
    virtual void prepare_fusion_data(fdata_map &fdmap) override; \
    virtual void infer_slice_ranges( \
            fslice_map &fsmap, infer_status_map_t &stat_map) override; \
    virtual void pre_slice_ranges( \
            fslice_map &fsmap, infer_status_map_t &stat_map) override; \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs) override;

#define DECLARE_QUERY_AND_DEFAULT_COMPUTE() \
    virtual void prepare_fusion_data(fdata_map &fdmap) override; \
    virtual void infer_slice_ranges( \
            fslice_map &fsmap, infer_status_map_t &stat_map) override {} \
    virtual void pre_slice_ranges( \
            fslice_map &fsmap, infer_status_map_t &stat_map) override {} \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs) override {}

/**
 * The input argument Op
 * Inputs:
 *  - None
 * Outputs:
 *  - One or more tensors
 * Attrs:
 *  - values: std::shared_ptr<static_data_t> - (Optional) if the values of this
 *    input is known at compile time, the values
 *  - keep_plain: bool - default = false. It will only be used when
 *    graph.attrs_["is_input_plain"]=false. If keep_plain=true, we will keep the
 *    plain format for this input. Otherwise we may reset this input to blocking
 *    format
 * */
class input_op : public fusible_op_t {
public:
    DECLARE_QUERY_AND_DEFAULT_COMPUTE();

    input_op(const sc_dims &dims = {}, sc_data_type_t dtype = datatypes::f32);
    input_op(const logical_tensor_t &lt);
    input_op(const std::vector<graph_tensor_ptr> &outs);

    void initialize_dynamic_placeholder();
    const bool is_arg_input() {
        return attrs_.get_or_else("temp.arg_input", false);
    }
};

/**
 * The output argument Op
 * Inputs:
 *  - One or more tensors
 * Outputs:
 *  - None
 * Attrs:
 *  - target_formats: std::vector<sc_data_format_t> - default: vector of plain
 *    format. It will be used only if the graph's `is_output_plain` attr is
 *    true. target_formats is the format of the output tensor. They should be
 *    either plain or simply permuted. If the target_formats is set, the
 *    output tensor will be reordered into the specified format.
 *  - target_strides: std::vector<sc_dims> - default: vector of dense strides
 *    target_strides is the stride of the output tensor.
 * */
class output_op : public fusible_op_t {
public:
    DECLARE_QUERY_AND_DEFAULT_COMPUTE();

    output_op(const graph_tensor_ptr &v);
    output_op(const std::vector<graph_tensor_ptr> &in);
};

/**
 * The op that produces constant tensors.
 * Inputs:
 *  - None
 * Outputs:
 *  - One single output tensor
 * Attrs:
 *  - values: std::shared_ptr<static_data_t> - the data
 *  - dtype: sc_data_type_t - the datatype. (todo: remove this attr)
 *  - plain_dims: dims (todo: remove this attr)
 *  - format: sc_data_format_t (todo: remove this attr)
 * */
class constant_op_t : public fusible_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_DEFAULT_COMPUTE();

    constant_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    constant_op_t(const std::shared_ptr<static_data_t> v, sc_data_type_t dtype,
            const sc_dims &plain_dims,
            const sc_data_format_t &format = sc_data_format_t());
    std::shared_ptr<static_data_t> get_constant_values() const {
        return const_values_;
    }
    sc_data_type_t get_constant_dtype() const {
        return info_.outputs_[0]->details_.dtype_;
    }
    const sc_dims &get_constant_plain_dims() const {
        return info_.outputs_[0]->details_.get_plain_dims();
    }
    const sc_data_format_t &get_constant_format() {
        return info_.outputs_[0]->details_.get_format();
    }
    sc_dims get_constant_blocking_dims() {
        return sc_data_format_t::get_blocking_shapes(
                get_constant_plain_dims(), get_constant_format());
    }
    bool compare_contents(const sc_op *other,
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter) const override;
    size_t hash_contents(
            const std::function<bool(const sc_op *, const std::string &)>
                    &filter) const override;

    // if necessary, reset const_values according possible `var` from attrs
    void reset_const_values();

private:
    std::shared_ptr<static_data_t> const_values_;
};

// this structure is used to store vectorized information, including axis and
// lanes;
struct vectorized_info_t {
    // the last valid axis to vectorized, skip invalid dims whose length equals
    // to 1, the default is last dim for most conditions
    int axis = -1;
    // vectorized lanes
    uint32_t lanes = 0;
};

class binary_elementwise_op_t : public fusible_op_t,
                                public op_traits::may_inplace_t,
                                public op_traits::may_broadcast_t,
                                public op_traits::batchwise_shrinkable_t,
                                public op_traits::brgemm_fusion_acceptable_t,
                                public op_traits::auto_copyable_with_trait_t<
                                        op_traits::brgemm_fusion_acceptable_t> {
public:
    int get_broadcast_input() const;
};

inline bool is_broadcast_op(const sc_op *op) {
    return (op->isa<op_traits::may_broadcast_t>()
            && op->dyn_cast<const op_traits::may_broadcast_t>()
                            ->get_non_broadcast_input_index(true)
                            .size()
                    != op->get_inputs().size());
}

class unary_elementwise_op_t : public fusible_op_t,
                               public op_traits::may_inplace_t,
                               public op_traits::brgemm_fusion_acceptable_t,
                               public op_traits::batchwise_shrinkable_t,
                               public op_traits::auto_copyable_with_trait_t<
                                       op_traits::brgemm_fusion_acceptable_t> {
};

// used for classification
class movement_op_t : public fusible_op_t, public op_traits::may_quantize_t {};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
