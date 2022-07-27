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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REDUCE_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_REDUCE_HPP

#include <string>
#include <utility>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>

namespace sc {
enum class reduce_operator : int {
    add = 0,
    mul,
};

// reduce op
class reduce_op_t : public fusible_op_t,
                    public op_traits::auto_copyable_t,
                    public op_traits::batchwise_shrinkable_t {
public:
    DECLARE_QUERY_AND_COMPUTE();

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

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

    sc_dims get_bwise_fuse_shrink_dims() override;
    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;
    void collect_shrinked_axes_map(
            int bw_size, gt2axes_map &bw_axes_map) override;

    // returns true if the reduce_op can be splitted into reduce_compute +
    // reduce_collect
    bool can_split_op() const;
    // split into reduce_compute + reduce_collect
    graph_tensor_ptr split_op(
            const context_ptr &ctx, sc_graph_t &graph, int num_threads);

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

class reduce_impl_op_t : public fusible_op_t {
public:
    void prepare_fusion_data(fdata_map &fdmap) override;
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;

    reduce_impl_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::string &rd_name,
            const std::vector<int> &rd_axis, reduce_operator rd_op,
            bool keep_dims, bool need_mean, uint64_t reduce_mean_num);
    // get real sorted reduce axis
    const std::vector<int> &get_rd_axis() const;

protected:
    // the axis which need reduction
    std::vector<int> real_rd_axis_;
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
    uint64_t reduce_mean_num_;
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
 * */
class reduce_compute_op_t : public reduce_impl_op_t,
                            public op_traits::copyable_t {
public:
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;
    reduce_compute_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::string &rd_name,
            const std::vector<int> &rd_axis, reduce_operator rd_op,
            bool keep_dims, bool need_mean, uint64_t reduce_mean_num);
    bool is_partial_reduce() const;
    // NOLINT because false alarm on copy()
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
};

class reduce_collect_op_t : public reduce_impl_op_t,
                            public op_traits::copyable_t {
public:
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;
    reduce_collect_op_t(const graph_tensor_ptr &in,
            const graph_tensor_ptr &old_out, const std::string &rd_name,
            const std::vector<int> &rd_axis, reduce_operator rd_op,
            bool keep_dims, bool need_mean, uint64_t reduce_mean_num);
    bool is_place_holder_op() const;
    sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
};
} // namespace sc
#endif
