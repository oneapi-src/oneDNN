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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_QUANTIZATION_QUANTIZE_INFO_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_QUANTIZATION_QUANTIZE_INFO_HPP

#include <vector>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <util/any_map.hpp>
namespace sc {
namespace quantize {

enum class tensor_type { input_tensor, weight_tensor, bias_tensor };

/** information for int8 quantization, include quantize/re-quantize/de-quantize.
 * @param dtype_ decide clip range, u8 [0,255], s8 [-128, 127]
 * @param out_scales_ scales for current quantize op output.
 * @param out_zero_points_ output zero points for asymmetric quantize, no need
 * input zero points.
 * @param in_scales_ scales for last quantize op output, used by re-quantize and
 * de-quantize op.
 * @param per_channel_ a param for conv op, if true, each channel has a scale
 * and zero point.
 * @param channel_axis_ an option param available when per_channel_ is true,
 * points which axis is channel.
 * @param asymmetric_ whether use symmetric or asymmetric quantization.(not used
 * yet)
 * */

struct quantize_infos_t {
    sc_data_type_t dtype_ = sc_data_type_t::u8(1);
    std::vector<float> scales_ = {1.f};
    std::vector<int> zero_points_ = {0};
    bool per_channel_ = false;
    int channel_axis_ = 0;
    bool asymmetric_ = false;
    bool dynamic_ = false;
    quantize_infos_t() = default;
    quantize_infos_t(sc_data_type_t dtype, const std::vector<float> &scales,
            const std::vector<int> &zero_points, bool per_channel = false,
            int channel_axis = 0, bool asymmetric = false, bool dynamic = false)
        : dtype_(dtype)
        , scales_(scales)
        , zero_points_(zero_points)
        , per_channel_(per_channel)
        , channel_axis_(channel_axis)
        , asymmetric_(asymmetric)
        , dynamic_(dynamic) {}
};
quantize_infos_t get_quantize_info_from_attrs(const any_map_t &attrs);

} // namespace quantize
} // namespace sc
#endif
