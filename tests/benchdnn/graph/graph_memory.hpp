/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_MEMORY_HPP
#define BENCHDNN_GRAPH_MEMORY_HPP

#include <tuple>
#include "common.hpp"
#include "deserialize.hpp"
#include "dnnl_common.hpp"
#include <type_traits>
#include <unordered_map>

#include "setting_handler.hpp"
#include "utils/compare.hpp"
#include "utils/settings.hpp"

#ifdef DNNL_WITH_SYCL
#include "dnnl_sycl.hpp"
#endif

namespace graph {

struct dnn_graph_mem_t {

public:
    // Construct graph memory on ref path based on primitive mem
    // Apart from oneDNN memory tag, oneDNN Graph has op attributes `data_format`
    // (NXC/NCX) and `weights_format`(OIX/IOX/XOI/XIO) to indicate the order of
    // dimensions.
    //
    // For example, tensor with shape [1,4,4,3] and NXC data_format means
    // batch_size=1, spacial_dims=4x4, channel_dim=3. This semantic info is
    // necessary for some scenarios, i.e. per_channel quantization.
    //
    // As the order of dimensions in oneDNN memory descriptor is always
    // NC[D[H[W]]], to align with oneDNN, the shape and strides should be changed
    // properly in setting conversion stage to get the memory tag correctly.
    //
    // Meanwhile, in graph path, the shape and strides of the memory for graph
    // path should remain the same as in the deserialized graph. In addition,
    // it should has the same data as that for reference path.
    //
    // Therefore, it needs to be checked that if the shape and strides of the
    // logical tensor have been modified. If so, the driver should use the shape
    // and strides from deserialized graph instead.
    //
    //
    // The constructor accepts three boolean parameters:
    // 1. is_op_input: whether the logical tensor is an input of an op
    // 2. is_fake_output: for fake outputs, the driver cannot create memory
    // objects based on primitive memory for them, but construct memory
    // from graph shape. The default value is false.
    //
    dnn_graph_mem_t(const dnn_mem_t &mem, const deserialized_lt &lt,
            const bool is_op_input, const bool is_fake_output = false);

    dnnl::graph::tensor make_graph_tensor(const deserialized_lt &lt) const;

    const dnn_mem_t &get_mem() const { return mem_; }

    void map_mem() { mem_.map(); }
    void unmap_mem() { mem_.unmap(); }

private:
    dnn_mem_t mem_;
    std::shared_ptr<void> buffer_;
    dnnl::memory::dims graph_dims_;
    dnnl::memory::dims graph_strides_;
};

using partition_mem_map_t = std::unordered_map<size_t, dnn_graph_mem_t>;

} // namespace graph

#endif
