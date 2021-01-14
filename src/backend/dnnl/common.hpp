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

#ifndef BACKEND_DNNL_COMMON_HPP
#define BACKEND_DNNL_COMMON_HPP

#include <utility>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph_types.h"

#include "interface/allocator.hpp"
#include "interface/common.hpp"
#include "interface/logical_tensor.hpp"
#include "utils/compatible.hpp"

#include "dnnl.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

using error = dnnl::error;
using memory = dnnl::memory;
using desc = memory::desc;
using format_tag = memory::format_tag;
using tag = memory::format_tag;
using data_type = typename memory::data_type;
using dims = typename memory::dims;
using dim = memory::dim;
using query = dnnl::query;
using kind = dnnl::primitive::kind;
using prop_kind = dnnl::prop_kind;
using algorithm = dnnl::algorithm;
using normalization_flag = dnnl::normalization_flags;
using query = dnnl::query;
using scale_t = std::vector<float>;
using exec_args = std::unordered_map<int, memory>;

struct allocator {
    static void *malloc(
            size_t size, const engine &p_engine, const impl::allocator_t *alc);

    static void free(
            void *p, const engine &p_engine, const impl::allocator_t *alc);
};

format_tag get_default_format(size_t ndim);

format_tag get_default_format(const dims adims);

dims get_compatible_dilates(const dims &dilates, size_t input_size = 4);

dims group_dims(const dims &adims, dim groups);

std::pair<std::vector<float>, std::vector<float>> compute_scales(
        float src_scale, float dst_scale, std::vector<float> weight_scales);

inline int op_scale_mask(dim scale_size) {
    return scale_size > 1 ? 2 : 0;
}

inline int tensor_scale_mask(dim scale_size, bool grouped) {
    return scale_size > 1 ? grouped ? 3 : 1 : 0;
}

inline int tensor_zp_mask(dim zp_size) {
    return zp_size > 1 ? 1 : 0;
}

engine make_dnnl_engine(const impl::engine_t &g_engine);

stream make_dnnl_stream(const engine &p_engine, const impl::stream_t &g_stream);

memory::desc make_dnnl_memory_desc(const impl::logical_tensor_t &lt);

memory make_dnnl_memory(const impl::tensor_t &atensor, const engine &p_engine);

memory::desc expand(const memory::desc &adesc, int tgt_ndims);

memory::desc permute_NXC2NCX(const memory::desc &adesc);

memory::desc permute_XIO2OIX(const memory::desc &adesc);

#ifndef NDEBUG
#define BACKEND_DNNL_ENFORCE(condition, message) \
    do { \
        error::wrap_c_api((condition) ? dnnl_success : dnnl_invalid_arguments, \
                (message)); \
    } while (false)
#else
#define BACKEND_DNNL_ENFORCE(condition, message)
#endif

#define BACKEND_DNNL_TYPE_DISPATCH(type_enum, type_key, ...) \
    switch (type_enum) { \
        case data_type::f32: { \
            using type_key = float; \
            __VA_ARGS__ \
        } break; \
        default: error::wrap_c_api(dnnl_unimplemented, "Unimplemented type"); \
    }

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
