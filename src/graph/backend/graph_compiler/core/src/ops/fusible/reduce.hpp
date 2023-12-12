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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REDUCE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REDUCE_HPP

#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusion_data.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace op_traits {
struct maybe_split_optimized_t : public virtual op_base_trait_t {
    // returns true if the reduce_op can be splitted into reduce_compute +
    // reduce_collect
    virtual bool can_split_op() const = 0;
    // split into reduce_compute + reduce_collect
    virtual graph_tensor_ptr split_op(
            const context_ptr &ctx, sc_graph_t &graph, int num_threads)
            = 0;
};
} // namespace op_traits

enum class reduce_operator : int {
    add = 0,
    mul,
    max,
    min,
};

// reduce op
class reduce_op_t : public fusible_op_t,
                    public op_traits::auto_copyable_t,
                    public op_traits::maybe_split_optimized_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    reduce_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    reduce_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            const reduce_operator &rd_op_, const any_map_t &attrs);

    reduce_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            reduce_operator rd_op = reduce_operator::add,
            bool keep_dims = false);
    uint32_t get_lanes() const { return vx_info_.lanes; }
    // get real reduce axis, generaly, you should set rd_axis on plain format
    // semantics.
    std::vector<int> get_rd_axis() const;
    // get type of reduction
    const reduce_operator get_rd_op() const { return rd_op_; }
    // get a compressed int rd_axis.
    int get_compressed_rd_axis_int() const;
    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

    // returns true if the reduce_op can be splitted into reduce_compute +
    // reduce_collect
    bool can_split_op() const override;
    // split into reduce_compute + reduce_collect
    graph_tensor_ptr split_op(const context_ptr &ctx, sc_graph_t &graph,
            int num_threads) override;
    shape_rl_vec get_dynamic_shape_relations() const override;

    void infer_binding_axis(binding_axis_map &bdax_map) override;
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

private:
    // the axis which need reduction
    std::vector<int> plain_rd_axis_;
    // type of reduction
    reduce_operator rd_op_;
    // if keep_dims=True, if will retain length=1 even though be reduced.
    bool keep_dims_;
    // use vectorized
    vectorized_info_t vx_info_;
};

// reduce_sum_op_t is derived from reduce_op_t
class reduce_sum_op_t : public reduce_op_t {
public:
    reduce_sum_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : reduce_op_t(std::move(v), rd_axis, reduce_operator::add, keep_dims) {}
    reduce_sum_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

// reduce_prod_op_t is derived from reduce_op_t
class reduce_prod_op_t : public reduce_op_t {
public:
    reduce_prod_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : reduce_op_t(std::move(v), rd_axis, reduce_operator::mul, keep_dims) {}
    reduce_prod_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

// reduce_max_op_t is derived from reduce_op_t
class reduce_max_op_t : public reduce_op_t {
public:
    reduce_max_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : reduce_op_t(std::move(v), rd_axis, reduce_operator::max, keep_dims) {}
    reduce_max_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

// reduce_min_op_t is derived from reduce_op_t
class reduce_min_op_t : public reduce_op_t {
public:
    reduce_min_op_t(graph_tensor_ptr v, const std::vector<int> &rd_axis,
            bool keep_dims = false)
        : reduce_op_t(std::move(v), rd_axis, reduce_operator::min, keep_dims) {}
    reduce_min_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

class reduce_impl_op_t : public fusible_op_t {
public:
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    infer_status_code pre_infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap) override;

    reduce_impl_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
            reduce_operator rd_op, bool keep_dims);
    // get real sorted reduce axis
    const std::vector<int> &get_rd_axis() const;
    // get type of reduction
    const reduce_operator get_rd_op() const { return rd_op_; }

    void infer_binding_axis(binding_axis_map &bdax_map) override;
    void pre_infer_binding_axis(binding_axis_map &bdax_map) override;

    // set the attributes of the reduce buffer, like init values
    virtual void set_reduce_buffer(const tensor &buf) = 0;

protected:
    // the axis which need reduction
    std::vector<int> real_rd_axis_;
    // type of reduction
    reduce_operator rd_op_;
    // if keep_dims=True, if will retain length=1 even though be reduced.
    bool keep_dims_;
    // use vectorized
    vectorized_info_t vx_info_;
};

/**
 * Reduce Op will be replace by reduce_compute_op + reduce_collect_op.
 * If there is no parallelism in the reduction axis, reduce_compute_op will do
 * elementwise reduction on the result tensor. reduce_collect_op will be a
 * no-op, and is just a placeholder to tell fusion manager to place the
 * computation after the reduce_op in an outer-loop anchor.
 *
 * If there is parallelism in the reduction axis, reduce_compute_op will do
 * partial reduction on the thread-shared tensor. Another reduce_op will collect
 * the result and do final reduction.
 * To optimize frequent memory load-store to thread-shared tensor, we further
 * transform reduce_compute_op in partial-reduction mode into
 * local-reduce_compute_op + copy-reduce_collect_op pair when the reduction
 * buffer is small enough to be held in registers. The local-reduce_compute_op's
 * output will be a small local tensor instead of a thread-shared global tensor.
 * The local tensor will be further optimized to registers. A
 * copy-reduce_collect_op will copy the local tensor (registers, after
 * tensor2var optimization) to the final global thread-shared tensor.
 * */
class reduce_compute_op_t : public reduce_impl_op_t,
                            public op_traits::copyable_t,
                            public op_traits::maybe_split_optimized_t {
public:
    bool local_mode_;
    infer_status_code infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap) override;
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;
    reduce_compute_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
            reduce_operator rd_op, bool keep_dims, bool local_mode);
    bool is_partial_reduce() const;
    // NOLINT because false alarm on copy()
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
    // split global-partial-reduce-compute into
    // local-reduce-compute+copy-reduce-collect
    bool can_split_op() const override;
    graph_tensor_ptr split_op(const context_ptr &ctx, sc_graph_t &graph,
            int num_threads) override;
    void set_reduce_buffer(const tensor &buf) override;
};

class reduce_collect_op_t : public reduce_impl_op_t,
                            public op_traits::copyable_t {
public:
    enum kind { NOOP, LAST_AXIS_COLLECT, COPY } op_;
    infer_status_code infer_slice_ranges(
            const context_ptr &ctx, fslice_map &fsmap) override;
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;
    reduce_collect_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::vector<int> &rd_axis,
            reduce_operator rd_op, bool keep_dims, kind op);
    bool is_place_holder_op() const { return op_ == NOOP; }
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
    void set_reduce_buffer(const tensor &buf) override;
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
