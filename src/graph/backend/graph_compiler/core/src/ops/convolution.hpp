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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_CONVOLUTION_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_CONVOLUTION_HPP

#include <vector>
#include <compiler/ir/graph/graph_op.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <ops/templates/nested_conv_fwd.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {

inline sc_dims get_dilations(const any_map_t &attr) {
    // In dumped graph, dilation is represented as std::vector<int>
    // but in onednn graph doc, the type of dilation is s64
    sc_dims dilations;
    try {
        dilations = attr.get_or_else("dilations", sc_dims({1}));
    } catch (...) {
        auto dilation_i = attr.get_or_else("dilations", std::vector<int>({1}));
        dilations = sc_dims(dilation_i.begin(), dilation_i.end());
    }
    return dilations;
}

class SC_INTERNAL_API conv_fwd_core_op_t
    : public tunable_op_t,
      public op_traits::batchwise_shrinkable_t {
public:
    conv_fwd_core_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    void infer_out_tensor_details() override;
    static sc_dims infer_out_dims(sc_graph_t &owner_graph,
            const sc_dims &input_dims, const sc_dims &weight_dims,
            const sc_dims &pads_begin, const sc_dims &pads_end,
            const sc_dims &stride, const sc_dims &dilation,
            const any_map_t &attrs = {});
    static void infer_auto_pad(sc_graph_t &owner_graph,
            const sc_dims &input_dims, const sc_dims &weight_dims,
            const sc_dims &stride, const sc_dims &dilation, any_map_t &attrs,
            bool is_same_upper);
    sc_data_type_t infer_out_dtype(const sc_data_type_t &input_dtype,
            const sc_data_type_t &weight_dtype);
    void check_dtypes(const sc_data_type_t &data_dtype,
            const sc_data_type_t &weight_dtype,
            const sc_data_type_t &out_dtype = datatypes::undef);
    sc_op_ptr do_compensations(sc_graph_t &g, const context_ptr &ctx) override;
    sc_op_ptr get_data_compensation(sc_graph_t &g);
    sc_op_ptr get_weight_compensation(sc_graph_t &g);
    sc_op_ptr get_constant_compensation(sc_graph_t &g);
    bool use_nested_conv_fwd_generator();
    bool use_conv1d();
    sc_dims get_bwise_fuse_shrink_dims() override;
    void collect_shrinked_lt_map(int bw_size, gt2gt_map &bw_lt_map) override;
    void collect_shrinked_axis_map(
            int bw_size, gt2axis_map &bw_axis_map) override;
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;

    void set_config_by_key(
            const op_dispatch_key_t &key, const context_ptr &ctx) override;
    virtual sc_op_ptr copy(const std::vector<graph_tensor_ptr> &ins, // NOLINT
            const std::vector<graph_tensor_ptr> &outs,
            sc_graph_t &mgr) override;
    std::vector<int> get_impl_dispatch_candidates(
            const context_ptr &ctx) override;
    shape_rl_vec get_dynamic_shape_relations() const override;
    static shape_rl_vec get_shape_relations_impl(const sc_dims &data_plain_dims,
            const sc_dims &weight_plain_dims, const sc_dims &out_plain_dims,
            const any_map_t &attrs);
    reflection::shared_general_object_t get_dynamic_runtime_info() override;

private:
    int ndims_ = 0;
    nested_conv_fwd_config_t dynamic_conv_param {};
};

class SC_INTERNAL_API conv_bwd_data_core_op_t : public tunable_op_t {
public:
    conv_bwd_data_core_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {
        // TODO(XXX)
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
    }
    bool use_nested_generator();

private:
    int ndims_ = 0;
    bool is_1x1_;
};

class SC_INTERNAL_API conv_bwd_weight_core_op_t : public tunable_op_t {
public:
    conv_bwd_weight_core_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    body_generator_ptr create_generator() override;
    float get_gflop() override;
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {
        // TODO(XXX)
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
    }
    bool use_nested_generator();

private:
    int ndims_ = 0;
    bool is_1x1_;
};

} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
