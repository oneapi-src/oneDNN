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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_POOLING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_POOLING_HPP

#include <vector>
#include "compiler/dimensions.hpp"
#include <compiler/ir/graph/fusible_op.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class pooling_type_t : int { avg = 0, max };
namespace pooling_attr_key {
constexpr const char *pooling_type = "pooling_type";
constexpr const char *strides = "strides";
constexpr const char *pads_begin = "pads_begin";
constexpr const char *pads_end = "pads_end";
constexpr const char *paddings = "paddings";
constexpr const char *kernel = "kernel";
constexpr const char *exclude_pad = "exclude_pad";
constexpr const char *rounding_type = "rounding_type";
constexpr const char *auto_pad = "auto_pad";
constexpr const char *input_shape = "input_shape";
constexpr const char *data_format = "data_format";
} // namespace pooling_attr_key

namespace auto_pad_options {
constexpr const char *none = "None";
constexpr const char *same_upper = "SAME_UPPER";
constexpr const char *same_lower = "SAME_LOWER";
constexpr const char *valid = "VALID";
} // namespace auto_pad_options

namespace rounding_type_options {
constexpr const char *floor = "floor";
constexpr const char *ceil = "ceil";
} // namespace rounding_type_options

namespace data_format_options {
constexpr const char *NCX = "NCX";
constexpr const char *NXC = "NXC";
} // namespace data_format_options
class pooling_op_t : public fusible_op_t,
                     public op_traits::auto_copyable_t,
                     public op_traits::may_quantize_t {
public:
    pooling_type_t pooling_type_;
    sc_dims stride_;
    sc_dims pads_begin_;
    sc_dims pads_end_;
    sc_dims kernel_;

    pooling_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    pooling_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs,
            const pooling_type_t &pl_type, const any_map_t &attrs);

    DECLARE_QUERY_AND_COMPUTE()

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

    std::vector<int> get_real_pooling_axis() const;

    std::vector<int> get_channel_axis() const;

private:
    sc_dims _calculate_output_dims(bool rounding_floor, bool channel_last);

    bool channel_last_;
    // use vectorized
    vectorized_info_t vx_info_;
};

class pooling_avg_op_t : public pooling_op_t {
public:
    using parent = pooling_op_t;
    /*
     * Inputs:
     * 1: input - input tensor. Required.
     * Attributes:
     * strides Required
     * pads_begin + pads_end  (or paddings)  Required
     * kernel Required
     * exclude_pad Required
     * rounding_type Optional
     * auto_pad Optional
     * data_format Optional
     */
    pooling_avg_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

class pooling_max_op_t : public pooling_op_t {
public:
    using parent = pooling_op_t;
    /*
     * Inputs:
     * 1: input - input tensor. Required.
     * Attributes:
     * strides Required
     * pads_begin + pads_end  (or paddings)  Required
     * kernel Required
     * rounding_type Optional
     * auto_pad Optional
     * data_format Optional
     */
    pooling_max_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};

class pooling_backprop_op_t : public fusible_op_t,
                              public op_traits::auto_copyable_t {
public:
    pooling_type_t pooling_type_;
    sc_dims stride_;
    sc_dims pads_begin_;
    sc_dims pads_end_;
    sc_dims kernel_;

    pooling_backprop_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    DECLARE_QUERY_AND_COMPUTE()

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;

    size_t compute_workload(const std::vector<shape_dtype_pair> &,
            const std::vector<shape_dtype_pair> &) override;

private:
    bool channel_last_;
    // use vectorized
    vectorized_info_t vx_info_;
};

class pooling_avg_backprop_op_t : public pooling_backprop_op_t {
public:
    using parent = pooling_backprop_op_t;
    /*
     * Inputs:
     * 1: output_delta  Required.
     * Attributes:
     * strides Required
     * pads_begin + pads_end  (or paddings)  Required
     * kernel Required.
     * exclude_pad Required.
     * input_shape Required.
     * auto_pad Optional
     * data_format Optional
     */
    pooling_avg_backprop_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    pooling_avg_backprop_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, sc_dims &input_shape,
            const any_map_t &attrs);
};

class pooling_max_backprop_op_t : public pooling_backprop_op_t {
public:
    using parent = pooling_backprop_op_t;
    /*
     * Inputs:
     * 1: output_delta   Required.
     * 2: input_forward   Required.
     * Attributes:
     * strides Required
     * pads_begin + pads_end  (or paddings)  Required
     * kernel Required
     * auto_pad Optional
     * data_format Optional
     */
    pooling_max_backprop_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
