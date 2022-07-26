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

#ifndef GRAPH_BACKEND_DNNL_COMMON_HPP
#define GRAPH_BACKEND_DNNL_COMMON_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

#include "graph/interface/allocator.hpp"
#include "graph/interface/logical_tensor.hpp"
#include "graph/interface/value.hpp"

#include "graph/utils/compatible.hpp"

namespace dnnl {
namespace impl {
namespace graph {
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

struct dnnl_allocator_t {
    static void *malloc(size_t size, const dnnl::engine &p_engine,
            const allocator_t *alc, allocator_t::mem_type_t type);

    static void free(
            void *p, const dnnl::engine &p_engine, const allocator_t *alc);

#ifdef DNNL_WITH_SYCL
    static void free(void *p, const dnnl::engine &p_engine,
            const allocator_t *alc, const ::sycl::event &deps);
#endif
};

format_tag get_ncx_format(size_t ndim);

format_tag get_ncx_format(const dims &adims);

dims get_compatible_dilates(const dims &dilates, size_t input_size = 4);

dims group_dims(const dims &adims, dim groups);

engine make_dnnl_engine(const engine_t &g_engine);

stream make_dnnl_stream(const engine &p_engine, const stream_t &g_stream);

memory::desc make_dnnl_memory_desc(const logical_tensor_t &lt);

memory make_dnnl_memory(const tensor_t &atensor, const engine &p_engine);

dnnl::memory make_dnnl_memory(const dnnl::memory::desc &md,
        const dnnl::engine &p_engine, void *handle);

memory::desc expand(const memory::desc &adesc, int tgt_ndims);

memory::desc permute_last_two_dims(const memory::desc &adesc);

memory::desc permute_NXC2NCX(const memory::desc &adesc);

memory::desc permute_NCX2NXC(const memory::desc &adesc);

memory::desc permute_XIO2OIX(const memory::desc &adesc);

memory::desc transpose(const memory::desc &adesc, dim dim0, dim dim1);

memory::desc to_grouped(const memory::desc &adesc, dim groups);

memory::desc from_grouped(const memory::desc &adesc);

memory::desc to_format_any(const memory::desc &adesc);

memory::desc permute_NCX2NXC(const memory::desc &adesc);

memory::desc permute_OIX2XIO(const memory::desc &adesc);

dims get_ncx_strides(const dims &shape);

dims get_nxc_strides(const dims &shape);

memory::desc to_nxc_format(const memory::desc &adesc);

bool is_format(const memory::desc &adesc, memory::format_tag tag);

bool is_format(const memory::desc &adesc, const std::string &tag);

bool is_4c_blocked(const memory::desc &adesc);

memory::desc to_ncx_format(const memory::desc &adesc);

status_t fill_layout_info(logical_tensor_t *lt, const memory::desc &md);

status_t fill_layout_info(
        const std::shared_ptr<value_t> &val, const memory::desc &md);

#ifndef NDEBUG
#define BACKEND_DNNL_ENFORCE(condition, message) \
    do { \
        error::wrap_c_api((condition) ? dnnl_success : dnnl_invalid_arguments, \
                (message)); \
    } while (false)
#else
#define BACKEND_DNNL_ENFORCE(condition, message)
#endif

#define BACKEND_DNNL_CHECK(statement) \
    do { \
        status_t ret = (statement); \
        if (ret != status::success) return ret; \
    } while (false)

#define BACKEND_DNNL_TYPE_DISPATCH(type_enum, type_key, ...) \
    switch (type_enum) { \
        case data_type::f32: { \
            using type_key = float; \
            __VA_ARGS__ \
        } break; \
        default: error::wrap_c_api(dnnl_unimplemented, "Unimplemented type"); \
    }

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
