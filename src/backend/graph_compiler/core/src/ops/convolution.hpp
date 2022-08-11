/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_CONVOLUTION_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_CONVOLUTION_HPP

#include <vector>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/graph/tunable_op.hpp>

namespace sc {
namespace ops {
class SC_INTERNAL_API conv_fwd_core_op_t : public tunable_op_t {
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
            const sc_dims &stride);
    sc_data_type_t infer_out_dtype(const sc_data_type_t &input_dtype,
            const sc_data_type_t &weight_dtype);
    void check_dtypes(const sc_data_type_t &data_dtype,
            const sc_data_type_t &weight_dtype,
            const sc_data_type_t &out_dtype = datatypes::undef);
    sc_op_ptr do_compensations(sc_graph_t &g, const context_ptr &ctx) override;
    sc_op_ptr get_data_compensation(sc_graph_t &g);
    sc_op_ptr get_weight_compensation(sc_graph_t &g);
    sc_op_ptr get_constant_compensation(sc_graph_t &g);

    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {
        /** TODO(XXX)
         * Please override this function, if planned to support commiting
         * convolution op like fusible op. Meanwhile, it is also required to
         * refactor current template implementation to tensor-slice based.
         * */
        stat_map.append_ops_by_status(this, infer_status_code::FAIL);
    }

private:
    int ndims_ = 0;
};

class conv_bwd_op_t : public tunable_op_t {
public:
    conv_bwd_op_t(const std::vector<graph_tensor_ptr> &ins,
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
};
} // namespace ops
} // namespace sc
#endif
