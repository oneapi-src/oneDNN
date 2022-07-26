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

#include <cassert>

#include "utils/debug.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

const char *dnnl_graph_runtime2str(unsigned runtime) {
    switch (runtime) {
        case DNNL_GRAPH_RUNTIME_NONE: return "none";
        case DNNL_GRAPH_RUNTIME_SEQ: return "sequential";
        case DNNL_GRAPH_RUNTIME_OMP: return "OpenMP";
        case DNNL_GRAPH_RUNTIME_TBB: return "TBB";
        case DNNL_GRAPH_RUNTIME_THREADPOOL: return "threadpool";
#ifdef DNNL_GRAPH_WITH_SYCL
        case DNNL_GRAPH_RUNTIME_SYCL: return "DPC++";
#endif
        default: return "unknown";
    }
}

const char *data_type2str(data_type_t v) {
    if (v == data_type::undef) return "undef";
    if (v == data_type::f16) return "f16";
    if (v == data_type::bf16) return "bf16";
    if (v == data_type::f32) return "f32";
    if (v == data_type::s32) return "s32";
    if (v == data_type::s8) return "s8";
    if (v == data_type::u8) return "u8";
    if (v == data_type::boolean) return "boolean";
    assert(!"unknown data_type");
    return "unknown data_type";
}

const char *engine_kind2str(engine_kind_t v) {
    if (v == engine_kind::any_engine) return "any";
    if (v == engine_kind::cpu) return "cpu";
    if (v == engine_kind::gpu) return "gpu";
    assert(!"unknown engine_kind");
    return "unknown engine_kind";
}

const char *fpmath_mode2str(fpmath_mode_t v) {
    if (v == fpmath_mode::strict) return "strict";
    if (v == fpmath_mode::bf16) return "bf16";
    if (v == fpmath_mode::f16) return "f16";
    if (v == fpmath_mode::any) return "any";
    if (v == fpmath_mode::tf32) return "tf32";
    assert(!"unknown fpmath_mode");
    return "unknown fpmath_mode";
}

const char *layout_type2str(layout_type_t v) {
    if (v == layout_type::undef) return "undef";
    if (v == layout_type::any) return "any";
    if (v == layout_type::strided) return "strided";
    if (v == layout_type::opaque) return "opaque";
    assert(!"unknown layout_type");
    return "unknown layout_type";
}

const char *property_type2str(property_type_t v) {
    if (v == property_type::undef) return "undef";
    if (v == property_type::variable) return "variable";
    if (v == property_type::constant) return "constant";
    assert(!"unknown property_type");
    return "unknown property_type";
}

std::string partition_kind2str(partition_kind_t v) {
    using namespace partition_kind;
#define CASE(x) \
    case (x): return #x

    switch (v) {
        CASE(undef);
        CASE(convolution_post_ops);
        CASE(convtranspose_post_ops);
        CASE(interpolate_post_ops);
        CASE(matmul_post_ops);
        CASE(reduction_post_ops);
        CASE(unary_post_ops);
        CASE(binary_post_ops);
        CASE(pooling_post_ops);
        CASE(batch_norm_post_ops);
        CASE(misc_post_ops);
        CASE(quantized_convolution_post_ops);
        CASE(quantized_convtranspose_post_ops);
        CASE(quantized_matmul_post_ops);
        CASE(quantized_unary_post_ops);
        CASE(quantized_pooling_post_ops);
        CASE(misc_quantized_post_ops);
        CASE(convolution_backprop_post_ops);
        CASE(mha);
        CASE(mlp);
        CASE(quantized_mha);
        CASE(quantized_mlp);
        CASE(residual_conv_blocks);
        CASE(quantized_residual_conv_blocks);
        default: return "unknown_kind";
    }
#undef CASE
}

partition_kind_t str2partition_kind(const std::string &str) {
    using namespace partition_kind;
#define IF_HANDLE(x) \
    if (str == #x) return x

    IF_HANDLE(undef);
    IF_HANDLE(convolution_post_ops);
    IF_HANDLE(convtranspose_post_ops);
    IF_HANDLE(interpolate_post_ops);
    IF_HANDLE(matmul_post_ops);
    IF_HANDLE(reduction_post_ops);
    IF_HANDLE(unary_post_ops);
    IF_HANDLE(binary_post_ops);
    IF_HANDLE(pooling_post_ops);
    IF_HANDLE(batch_norm_post_ops);
    IF_HANDLE(misc_post_ops);
    IF_HANDLE(quantized_convolution_post_ops);
    IF_HANDLE(quantized_convtranspose_post_ops);
    IF_HANDLE(quantized_matmul_post_ops);
    IF_HANDLE(quantized_unary_post_ops);
    IF_HANDLE(quantized_pooling_post_ops);
    IF_HANDLE(misc_quantized_post_ops);
    IF_HANDLE(convolution_backprop_post_ops);
    IF_HANDLE(mha);
    IF_HANDLE(mlp);
    IF_HANDLE(quantized_mha);
    IF_HANDLE(quantized_mlp);
    IF_HANDLE(residual_conv_blocks);
    IF_HANDLE(quantized_residual_conv_blocks);

    return undef;

#undef IF_HANDLE
}

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl
