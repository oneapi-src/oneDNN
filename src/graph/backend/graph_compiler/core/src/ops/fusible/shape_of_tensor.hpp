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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_SHAPE_OF_TENSOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_FUSIBLE_SHAPE_OF_TENSOR_HPP

#include <functional>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/traits.hpp>
#include <compiler/ir/ir_module.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// the padding shape type decides the runtime block selector function usage.
// E.g. matmul and conv may have different block select.
// Currently we only support mamtul block select.
enum class padding_shape_etype_t : int {
    without_padding = 0,
    matmul_padding,
    conv_padding
};
namespace attr_keys {
// value is padding_shape_etype_t
static constexpr const char *padding_shape_type = "padding_shape_type";
// value is boolean
static constexpr const char *shape_of_tensor_is_batch
        = "shape_of_tensor_is_batch";
} // namespace attr_keys
// Get plain(may padded) shapes of a tensor
class shape_of_tensor_op_t : public fusible_op_t,
                             public op_traits::auto_copyable_t {
public:
    shape_of_tensor_op_t(const std::vector<graph_tensor_ptr> &ins,
            const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs);

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override;
    void prepare_fusion_data(fdata_map &fdmap) override {};
    void infer_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override;
    void pre_slice_ranges(
            fslice_map &fsmap, infer_status_map_t &stat_map) override {}
    void compute_block(context_ptr ctx, const std::vector<tensor_slice *> &dst,
            const std::vector<const tensor_slice *> &inputs) override;

private:
    // index of real/padded_plain shapes
    int shape_idx_;
    // decides whether and how to use padding on the shape according to the
    // related tunable op.
    padding_shape_etype_t shape_type_;
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
