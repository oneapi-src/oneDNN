/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_CPU_ENGINE_HPP
#define CPU_CPU_ENGINE_HPP

#include <assert.h>

#include "dnnl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/mkldnn_sel_build.hpp"

#include "cpu/platform.hpp"

#if defined(SELECTIVE_BUILD_ANALYZER)

namespace dnnl {
namespace impl {
namespace cpu {

template<typename pd_t>
engine_t::primitive_desc_create_f primitive_desc_builder(const char *name) {
    OV_ITT_SCOPED_TASK(
        dnnl::FACTORY_MKLDNN,
        openvino::itt::handle<pd_t>(std::string("REG$CPUEngine$") + typeid(pd_t).name() + "$" + name));
    return &primitive_desc_t::create<pd_t>;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

# define CPU_INSTANCE(...) MKLDNN_MACRO_OVERLOAD(CPU_INSTANCE, __VA_ARGS__),
# define CPU_INSTANCE_1(name) dnnl::impl::cpu::primitive_desc_builder<name::pd_t>(OV_CC_TOSTRING(name))
# define CPU_INSTANCE_2(name, arg1) dnnl::impl::cpu::primitive_desc_builder<name<arg1>::pd_t>(OV_CC_TOSTRING(name ## _ ## arg1))
# define CPU_INSTANCE_3(name, arg1, arg2) dnnl::impl::cpu::primitive_desc_builder<name<arg1, arg2>::pd_t>(OV_CC_TOSTRING(name ## _ ## arg1 ## _ ## arg2))
# define CPU_INSTANCE_4(name, arg1, arg2, arg3) dnnl::impl::cpu::primitive_desc_builder<name<arg1, arg2, arg3>::pd_t>(OV_CC_TOSTRING(name ## _ ## arg1 ## _ ## arg2 ## _ ## arg3))
# define CPU_INSTANCE_5(name, arg1, arg2, arg3, arg4) dnnl::impl::cpu::primitive_desc_builder<name<arg1, arg2, arg3, arg4>::pd_t>(OV_CC_TOSTRING(name ## _ ## arg1 ## _ ## arg2 ## _ ## arg3 ## _ ## arg4))

#elif defined(SELECTIVE_BUILD)

# define CPU_INSTANCE_BUILDER_0(...)
# define CPU_INSTANCE_BUILDER_1(...) __VA_ARGS__,
# define CPU_INSTANCE_BUILDER(name, ...) OV_CC_EXPAND(OV_CC_CAT(CPU_INSTANCE_BUILDER_, OV_CC_SCOPE_IS_ENABLED(OV_CC_CAT(MKLDNN_, name)))(__VA_ARGS__))

# define CPU_INSTANCE(...) MKLDNN_MACRO_OVERLOAD(CPU_INSTANCE, __VA_ARGS__)
# define CPU_INSTANCE_1(name) CPU_INSTANCE_BUILDER(name, &primitive_desc_t::create<name::pd_t>)
# define CPU_INSTANCE_2(name, arg1) CPU_INSTANCE_BUILDER(name ## _ ## arg1, &primitive_desc_t::create<name<arg1>::pd_t>)
# define CPU_INSTANCE_3(name, arg1, arg2) CPU_INSTANCE_BUILDER(name ## _ ## arg1 ## _ ## arg2, &primitive_desc_t::create<name<arg1, arg2>::pd_t>)
# define CPU_INSTANCE_4(name, arg1, arg2, arg3) CPU_INSTANCE_BUILDER(name ## _ ## arg1 ## _ ## arg2 ## _ ## arg3, &primitive_desc_t::create<name<arg1, arg2, arg3>::pd_t>)
# define CPU_INSTANCE_5(name, arg1, arg2, arg3, arg4) CPU_INSTANCE_BUILDER(name ## _ ## arg1 ## _ ## arg2 ## _ ## arg3 ## _ ## arg4, &primitive_desc_t::create<name<arg1, arg2, arg3, arg4>::pd_t>)

#else

# define CPU_INSTANCE(...) MKLDNN_MACRO_OVERLOAD(CPU_INSTANCE, __VA_ARGS__),
# define CPU_INSTANCE_1(name) &primitive_desc_t::create<name::pd_t>
# define CPU_INSTANCE_2(name, arg1) &primitive_desc_t::create<name<arg1>::pd_t>
# define CPU_INSTANCE_3(name, arg1, arg2) &primitive_desc_t::create<name<arg1, arg2>::pd_t>
# define CPU_INSTANCE_4(name, arg1, arg2, arg3) &primitive_desc_t::create<name<arg1, arg2, arg3>::pd_t>
# define CPU_INSTANCE_5(name, arg1, arg2, arg3, arg4) &primitive_desc_t::create<name<arg1, arg2, arg3, arg4>::pd_t>

#endif

#define CPU_INSTANCE_X64(...) DNNL_X64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64(...) DNNL_AARCH64_ONLY(CPU_INSTANCE(__VA_ARGS__))
#define CPU_INSTANCE_AARCH64_ACL(...) \
    DNNL_AARCH64_ACL_ONLY(CPU_INSTANCE(__VA_ARGS__))

namespace dnnl {
namespace impl {
namespace cpu {

#define DECLARE_IMPL_LIST(kind) \
    const engine_t::primitive_desc_create_f *get_##kind##_impl_list( \
            const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(logsoftmax);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling_v2);
DECLARE_IMPL_LIST(reduction);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);

#undef DECLARE_IMPL_LIST

class cpu_engine_t : public engine_t {
public:
    cpu_engine_t()
        : engine_t(engine_kind::cpu, get_default_runtime(engine_kind::cpu)) {}

    /* implementation part */
    status_t create_memory_storage(memory_storage_t **storage, unsigned flags,
            size_t size, void *handle) override;

    status_t create_stream(stream_t **stream, unsigned flags,
            const stream_attr_t *attr) override;

    const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override;
    const reorder_primitive_desc_create_f *get_reorder_implementation_list(
            const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override;
    const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override;
    const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) const override {
        static const primitive_desc_create_f empty_list[] = {nullptr};

// clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        return get_##kind##_impl_list((const kind##_desc_t *)desc);
        switch (desc->kind) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(logsoftmax);
            CASE(matmul);
            case primitive_kind::pooling:
            CASE(pooling_v2);
            CASE(reduction);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            default: assert(!"unknown primitive kind"); return empty_list;
        }
#undef CASE
    }
    // clang-format on
};

class cpu_engine_factory_t : public engine_factory_t {
public:
    size_t count() const override { return 1; }
    status_t engine_create(engine_t **engine, size_t index) const override {
        assert(index == 0);
        *engine = new cpu_engine_t();
        return status::success;
    };
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
