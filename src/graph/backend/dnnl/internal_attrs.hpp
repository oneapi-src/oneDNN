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

#ifndef GRAPH_BACKEND_DNNL_INTERNAL_ATTRS_HPP
#define GRAPH_BACKEND_DNNL_INTERNAL_ATTRS_HPP

#include <string>

#include "graph/interface/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace op_attr {

using namespace dnnl::impl::graph::op_attr;

// internal attributes: bool
const op_attr_t canonicalized = 0x10000;
const op_attr_t change_layout = 0x10001;
const op_attr_t is_constant = 0x10002;
const op_attr_t is_convtranspose = 0x10003;
const op_attr_t is_training = 0x10004;
const op_attr_t fwd_alg_kind = 0x10005;
const op_attr_t fuse_relu = 0x10006;
const op_attr_t with_bias = 0x10007;
const op_attr_t with_dw_bias = 0x10008; // seems not used.
const op_attr_t with_runtime_scales = 0x10009;
const op_attr_t with_runtime_zps = 0x10000a;
const op_attr_t with_runtime_src_zps = 0x1000b;
const op_attr_t with_runtime_dst_zps = 0x1000c;
const op_attr_t is_bias_add = 0x1000d;
const op_attr_t with_sum = 0x1000e;

// int64_t
const op_attr_t alg_kind = 0x10100;
const op_attr_t dw_groups = 0x10101;
const op_attr_t expand_to = 0x10102;
const op_attr_t fusion_info_key = 0x10103;

// string
const op_attr_t dw_filter_format = 0x10200;
const op_attr_t dw_type = 0x10201;
const op_attr_t from_format = 0x10202;
const op_attr_t insert_1dim = 0x10203;
const op_attr_t kind = 0x10204;
const op_attr_t permute_kind = 0x10205;
const op_attr_t to_format = 0x10206;

// float
const op_attr_t p = 0x10300;

// vector of int64_t
const op_attr_t dst_zps = 0x10400;
const op_attr_t src_zps = 0x10401;

static inline std::string internal_attr2str(op_attr_t attr) {
#define CASE(a) \
    case (a): return #a

    switch (attr) {
        CASE(canonicalized);
        CASE(change_layout);
        CASE(is_constant);
        CASE(is_convtranspose);
        CASE(is_training);
        CASE(fwd_alg_kind);
        CASE(fuse_relu);
        CASE(with_bias);
        CASE(with_dw_bias);
        CASE(with_runtime_scales);
        CASE(with_runtime_zps);
        CASE(with_runtime_src_zps);
        CASE(with_runtime_dst_zps);
        CASE(is_bias_add);
        CASE(with_sum);
        CASE(alg_kind);
        CASE(dw_groups);
        CASE(expand_to);
        CASE(fusion_info_key);
        CASE(dw_filter_format);
        CASE(dw_type);
        CASE(from_format);
        CASE(insert_1dim);
        CASE(kind);
        CASE(permute_kind);
        CASE(to_format);
        CASE(p);
        CASE(dst_zps);
        CASE(src_zps);
        default: return "undefined_attr";
    }
#undef CASE
}

} // namespace op_attr
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
