/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_INPUT_DISPLACER_HPP
#define BENCHDNN_GRAPH_INPUT_DISPLACER_HPP

#include "ref_primitive.hpp"

#include "utils/fill.hpp"

namespace graph {

enum class filling_type_t {
    undef = 0,
    quantization,
    pow2,
};

// tuple<
//     main op,
//     main op offset,
//     the tensor as a displace starting point,
//     filling_type
// >
using displace_t = ::std::tuple<::graph::deserialized_op, size_t,
        ::graph::deserialized_lt, filling_type_t>;

class partition_data_displacer_t {
public:
    partition_data_displacer_t() = default;
    partition_data_displacer_t(
            const deserialized_graph &dg, const dnnl::graph::partition &par);
    int displace_input_data(size_t lt_id, dnn_mem_t &mem, res_t *res);

private:
    const deserialized_graph *dg_ = nullptr;
    // A set of op_id values from a partition came to a displacer. Used to
    // identify at displacement stage if Deq is the starting point or not.
    std::unordered_set<size_t> op_ids_set_;
    ::std::unordered_map<size_t, displace_t> quantize_displace_;

    int gen_quantize_filling(const ::graph::deserialized_op &main_op, int arg,
            dnn_mem_t &mem, const ::std::string &dt, res_t *res);
    // Generates floating-point power-of-2 values in the target memory.
    int gen_pow2_filling(dnn_mem_t &mem, const_dnnl_memory_desc_t lt,
            const fill_cfg_t &fill_cfg, res_t *res) const;
};

} // namespace graph

#endif
