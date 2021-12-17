/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSIBLE_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <unordered_map>

namespace sc {
struct fusion_anchor_data;

using slice_range = std::vector<std::pair<expr, expr>>;
using slice_range_list = std::vector<slice_range>;
using slice_range_map = std::unordered_map<int, slice_range_list>;

inline std::vector<expr> get_slice_idx(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(r.first);
    }
    return ret;
}

inline std::vector<expr> get_slice_shape(const slice_range &range) {
    std::vector<expr> ret;
    for (auto &r : range) {
        ret.emplace_back(r.second);
    }
    return ret;
}

/**
 * A slice of the tensor.
 * @param tptr_ the base tensor_ptr
 * @param shape_ the slice shape
 * */
struct tensor_slice {
    tensorptr tptr_;
    std::vector<expr> shape_;
    tensor_slice() = default;

    tensor_slice(const expr &tsr);

    tensor_slice(const expr &tsr, slice_range &&ranges);

    // Gets the start address of the tensor slice
    expr get_tensor_ptr() const { return tptr_; }

    // Gets the shape of the sliced tensor
    const std::vector<expr> &get_shape() const { return shape_; }

    int64_t nslice_dims() const { return static_cast<int64_t>(shape_.size()); }
    int64_t nbase_dims() const {
        return static_cast<int64_t>(get_base_dims().size());
    }

    // Gets the offset of the sliced tensor
    const std::vector<expr> &get_offset() const { return tptr_->base_->idx_; }

    // Gets the ranges of the sliced tensor
    slice_range get_ranges() const;

    // Gets the real shape of base tensor (const version)
    const std::vector<expr> &get_base_dims() const;

    // Gets the dtype of base tensor
    sc_data_type_t get_base_dtype() const;

    // Gets the real tensor of tensor slice, not the tensor_ptr
    tensor get_real_tensor() const;

    // check whether slice is full on specific axes
    bool full_on_axes(const std::vector<int> &axes) const;

    // is_full
    bool is_full() const;
};

/**
 * A fuser will do actual code injection on the fusion point. It will be managed
 * by the fusion manager.
 * */
class fusible_op_t : public sc_op, public op_traits::workload_computable_t {
public:
    // when fusible_op_t is as a started op in the graph/subgraph, query_format
    // return certain format.
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    ir_module_ptr get_func(context_ptr ctx) override;

    // prepares the Op's anchor-irrelevant internal states for generating IR for
    // the input shapes. Also propagates the tensor slice shapes from input to
    // output in the fusion data
    virtual void prepare_fusion_data(context_ptr ctx,
            const std::vector<tensor_slice> &src,
            const std::vector<tensor_slice> &dst, fusion_anchor_data &result)
            = 0;
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
    virtual void infer_slice_ranges(fusion_anchor_data &result) = 0;

    /**
     * 'pre_slice_ranges' is used to infer slice ranges especially
     * for pre-op fusion. As mentioned above `infer_slice_ranges` can infer
     * slice for input arg op, however, if sometimes the input of one fusible op
     * is a sub fusion graph, it is expected to infer input slice by given
     * output slice.
     * */
    virtual void pre_slice_ranges(fusion_anchor_data &result) = 0;
    /**
     * Does the actual code injection to the current func on an input
     * and output tensor slices.
     *
     * */
    virtual void compute_block(context_ptr ctx,
            const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs,
            fusion_anchor_data &result)
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
            const std::vector<const tensor_slice *> &inputs,
            fusion_anchor_data &result);
    ~fusible_op_t() override = default;

    // todo: shall we move this field to fdata?
    int anchor_id = 0;
};

using fusion_op_ptr = std::shared_ptr<fusible_op_t>;

#define DECLARE_COMPUTE() \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs, \
            fusion_anchor_data &result) override;

#define DECLARE_QUERY_AND_COMPUTE() \
    virtual void prepare_fusion_data(context_ptr ctx, \
            const std::vector<tensor_slice> &src, \
            const std::vector<tensor_slice> &dst, fusion_anchor_data &result) \
            override; \
    virtual void infer_slice_ranges(fusion_anchor_data &result) override; \
    virtual void pre_slice_ranges(fusion_anchor_data &result) override; \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs, \
            fusion_anchor_data &result) override;

#define DECLARE_QUERY_AND_DEFAULT_COMPUTE() \
    virtual void prepare_fusion_data(context_ptr ctx, \
            const std::vector<tensor_slice> &src, \
            const std::vector<tensor_slice> &dst, fusion_anchor_data &result) \
            override; \
    virtual void infer_slice_ranges(fusion_anchor_data &result) override {} \
    virtual void pre_slice_ranges(fusion_anchor_data &result) override {} \
    virtual void compute_block(context_ptr ctx, \
            const std::vector<tensor_slice *> &dst, \
            const std::vector<const tensor_slice *> &inputs, \
            fusion_anchor_data &result) override {}

slice_range_map search_known_slice_ranges(
        fusible_op_t *cur, fusion_anchor_data &fdata);
void set_unknown_slice_ranges(fusible_op_t *cur,
        slice_range_map known_ranges_map, fusion_anchor_data &fdata);
void infer_binary_slice_ranges(fusible_op_t *cur, fusion_anchor_data &fdata);
sc_dims get_expr_to_dims(const std::vector<expr> &dims);

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
    std::shared_ptr<static_data_t> get_constant_values() {
        return const_values_;
    }
    sc::sc_data_type_t get_constant_dtype() {
        return info_.outputs_[0]->details_.dtype_;
    }
    const sc_dims &get_constant_plain_dims() {
        return info_.outputs_[0]->details_.get_plain_dims();
    }
    const sc_data_format_t &get_constant_format() {
        return info_.outputs_[0]->details_.get_format();
    }
    sc_dims get_constant_blocking_dims() {
        return sc_data_format_t::get_blocking_shapes(
                get_constant_plain_dims(), get_constant_format());
    }
    bool compare_contents(const sc_op *other) const override;
    size_t hash_contents() const override;

private:
    std::shared_ptr<static_data_t> const_values_;
};

// this structure is used to store vectorized information, including axis and
// lanes;
struct vectorized_info_t {
    // the last valid axis to vectorized, skip invalid dims whose length equals
    // to 1, the default is last dim for most conditions
    int axis;
    // vectorized lanes
    uint32_t lanes;
};

enum class elt_operator {
    ADD,
    SUB,
    MUL,
    DIV,
    MIN,
    MAX,
    SQD_DIFF,
};

class binary_elementwise_op_t : public fusible_op_t,
                                public op_traits::may_broadcast_t,
                                public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    binary_elementwise_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            elt_operator elt_op, int inplace = 1);
    binary_elementwise_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    int get_broadcast_input() const override;
    std::vector<int> infer_broadcast_axis() const override;

    void set_elt_operator(elt_operator elt_op) { elt_op_ = elt_op; }

    uint32_t get_lanes() const { return vx_info_.lanes; }

    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    // get real broadcast axis, generaly, you should set bc_axis on plain format
    // semantics if necessary.
    std::vector<int> get_bc_axis() const;
    vectorized_info_t &get_vx_info() { return vx_info_; }

private:
    elt_operator elt_op_;
    int inplace_;
    vectorized_info_t vx_info_;
};

class add_op_t : public binary_elementwise_op_t {
public:
    add_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::ADD, inplace) {}
    add_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::ADD);
        op_name_ = "add";
    }
};

class sub_op_t : public binary_elementwise_op_t,
                 public op_traits::may_quantize_t {
public:
    sub_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::SUB, inplace) {}
    sub_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::SUB);
        op_name_ = "sub";
    }
};

class mul_op_t : public binary_elementwise_op_t {
public:
    mul_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::MUL, inplace) {}
    mul_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::MUL);
        op_name_ = "mul";
    }
};

class div_op_t : public binary_elementwise_op_t {
public:
    div_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::DIV, inplace) {}
    div_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::DIV);
        op_name_ = "div";
    }
};

class min_op_t : public binary_elementwise_op_t {
public:
    min_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::MIN, inplace) {}
    min_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::MIN);
        op_name_ = "min";
    }
};

class max_op_t : public binary_elementwise_op_t {
public:
    max_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(
                std::move(lhs), std::move(rhs), elt_operator::MAX, inplace) {}
    max_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::MAX);
        op_name_ = "max";
    }
};

class unary_elementwise_op_t : public fusible_op_t,
                               public op_traits::auto_copyable_t {
public:
    void infer_slice_ranges(fusion_anchor_data &result) override;
    void pre_slice_ranges(fusion_anchor_data &result) override;
    void prepare_fusion_data(context_ptr ctx,
            const std::vector<tensor_slice> &src,
            const std::vector<tensor_slice> &dst,
            fusion_anchor_data &result) override;

    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs,
            fusion_anchor_data &result) override;

    unary_elementwise_op_t(graph_tensor_ptr v, const std::string &op_name);
    unary_elementwise_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    vectorized_info_t &get_vx_info() { return vx_info_; }

    virtual expr compute_element(expr in) = 0;

private:
    vectorized_info_t vx_info_;
};

// used for classification
class movement_op_t : public fusible_op_t, public op_traits::may_quantize_t {};

class transpose_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    transpose_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    transpose_op_t(graph_tensor_ptr v, std::vector<int> &axes);

    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

private:
    std::vector<int> axes_;
};

/**
 * Creates a view of an input tensor with a different shape
 * Inputs:
 *  - A single tensor to reshape
 * Outputs:
 *  - The reshaped tensor
 * Attrs:
 *  - shape: vector<int> - the output blocking shape
 * */
class tensor_view_op_t : public movement_op_t,
                         public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;

    tensor_view_op_t(graph_tensor_ptr v, const sc_dims &shapes);
    tensor_view_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    sc_dims get_shapes() const;
    bool try_penetrate(sc_data_format_t &new_output_format) const;

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
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    ir_module_ptr get_func(context_ptr ctx) override;

private:
    sc_dims shapes_;
};

class split_op_t : public movement_op_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;

    split_op_t(graph_tensor_ptr v, int dim, const sc_dims &shapes);

private:
    unsigned dim_;
    sc_dims shapes_;
};

class reorder_op_t : public movement_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    reorder_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    reorder_op_t(graph_tensor_ptr v, sc_data_format_t input_format,
            sc_data_format_t output_format);
    ir_module_ptr get_func(context_ptr ctx) override;
    const sc_data_format_kind_t &get_input_format_kind() const {
        return input_format_.format_code_;
    }
    const sc_data_format_kind_t &get_output_format_kind() const {
        return output_format_.format_code_;
    }
    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;
    const sc_data_format_t &get_output_format() const { return output_format_; }
    const sc_data_format_t &get_input_format() const { return input_format_; }
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;
    bool check_padding() const;
    bool use_output_loop() const;

private:
    sc_dims plain_dims_;
    sc_data_format_t input_format_;
    sc_data_format_t output_format_;
};

enum class reduce_operator : int {
    add = 0,
    mul,
};

// reduce op
class reduce_op_t : public fusible_op_t, public op_traits::auto_copyable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<sc_data_format_t>> &in_formats,
            std::vector<std::vector<sc_data_format_t>> &out_formats) override;

    reduce_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    reduce_op_t(graph_tensor_ptr v, const std::string &rd_name,
            const std::vector<int> &rd_axis,
            reduce_operator rd_op = reduce_operator::add,
            bool keep_dims = false, bool need_mean = true);
    uint32_t get_lanes() const { return vx_info_.lanes; }
    // get real reduce axis, generaly, you should set rd_axis on plain format
    // semantics.
    std::vector<int> get_rd_axis() const;
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

private:
    // the axis which need reduction
    std::vector<int> plain_rd_axis_;
    // type of reduction
    reduce_operator rd_op_;
    // name of reduce_op_t
    std::string rd_name_;
    // if keep_dims=True, if will retain length=1 even though be reduced.
    bool keep_dims_;
    // whether need to compute mean
    bool need_mean_;
    // use vectorized
    vectorized_info_t vx_info_;
};

// reduce_add_op_t is derived from reduce_op_t
class reduce_add_op_t : public reduce_op_t {
public:
    reduce_add_op_t(graph_tensor_ptr v, const std::string &rd_name,
            const std::vector<int> &rd_axis, bool keep_dims = false,
            bool need_mean = true)
        : reduce_op_t(std::move(v), rd_name, rd_axis, reduce_operator::add,
                keep_dims, need_mean) {}
};

// reduce_mul_op_t is derived from reduce_op_t
class reduce_mul_op_t : public reduce_op_t {
public:
    reduce_mul_op_t(graph_tensor_ptr v, const std::string &rd_name,
            const std::vector<int> &rd_axis, bool keep_dims = false,
            bool need_mean = true)
        : reduce_op_t(std::move(v), rd_name, rd_axis, reduce_operator::mul,
                keep_dims, need_mean) {}
};

// squared_difference: (x-mean)^2
// squared_diff should support both elementwise and broad-cast mode.
class squared_diff_op_t : public binary_elementwise_op_t {
public:
    squared_diff_op_t(graph_tensor_ptr lhs, graph_tensor_ptr rhs,
            bool vectorized = false, int inplace = 1)
        : binary_elementwise_op_t(std::move(lhs), std::move(rhs),
                elt_operator::SQD_DIFF, inplace) {}
    squared_diff_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
        : binary_elementwise_op_t(ins, outs, attrs) {
        set_elt_operator(elt_operator::SQD_DIFF);
        op_name_ = "sqd_diff";
    }
};

} // namespace sc
#endif
