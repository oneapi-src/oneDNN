/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#pragma once
#include "mkldnn_macros.hpp"

#if defined(SELECTIVE_BUILD_ANALYZER)

#include <openvino/cc/selective_build.h>

namespace dnnl {

OV_CC_DOMAINS(MKLDNN)

namespace impl {

template<typename pd_t, typename ... Args>
dnnl::impl::status_t pd_create(Args ... args) {
    OV_ITT_SCOPED_TASK(
        dnnl::FACTORY_MKLDNN,
        openvino::itt::handle(std::string("CREATE$PDFactory$") + typeid(pd_t).name()));
    return pd_t::create(args...);
}

}   // namespace impl

#define MKLDNN_DEF_PD_BUILDER(PDBuilderName, FnT, ...)          \
    template<typename pd_t>                                     \
    FnT PDBuilderName(char const *name) {                       \
        OV_ITT_SCOPED_TASK(                                     \
            dnnl::FACTORY_MKLDNN,                               \
            openvino::itt::handle(std::string("REG$PDFactory$") \
                + typeid(pd_t).name() + "$" + name));           \
        return dnnl::impl::pd_create<pd_t, __VA_ARGS__>;        \
    }

# define REG_MKLDNN_FN(...) MKLDNN_MACRO_OVERLOAD(REG_MKLDNN_FN, __VA_ARGS__)
# define REG_MKLDNN_FN_2(builder, name) builder<name::pd_t>(OV_CC_TOSTRING(name)),
# define REG_MKLDNN_FN_3(builder, name, T1) builder<name<T1>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1)),
# define REG_MKLDNN_FN_4(builder, name, T1, T2) builder<name<T1, T2>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1 ## _ ## T2)),
# define REG_MKLDNN_FN_5(builder, name, T1, T2, T3) builder<name<T1, T2, T3>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1 ## _ ## T2 ## _ ## T3)),
# define REG_MKLDNN_FN_6(builder, name, T1, T2, T3, T4) builder<name<T1, T2, T3, T4>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4)),
# define REG_MKLDNN_FN_7(builder, name, T1, T2, T3, T4, T5) builder<name<T1, T2, T3, T4, T5>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4 ## _ ## T5)),
# define REG_MKLDNN_FN_8(builder, name, T1, T2, T3, T4, T5, T6) builder<name<T1, T2, T3, T4, T5, T6>::pd_t>(OV_CC_TOSTRING(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4 ## _ ## T5 ## _ ## T6)),

} // namespace dnnl

#define MKLDNN_CSCOPE(region, ...) OV_SCOPE(MKLDNN, region, __VA_ARGS__)

#elif defined(SELECTIVE_BUILD)

#include <openvino/cc/selective_build.h>

# define MKLDNN_OBJ_BUILDER_0(...)
# define MKLDNN_OBJ_BUILDER_1(...) __VA_ARGS__,
# define MKLDNN_OBJ_BUILDER(name, ...) OV_CC_EXPAND(OV_CC_CAT(MKLDNN_OBJ_BUILDER_, OV_CC_EXPAND(OV_CC_SCOPE_IS_ENABLED(OV_CC_CAT(MKLDNN_, name))))(__VA_ARGS__))

# define REG_MKLDNN_FN(...) MKLDNN_MACRO_OVERLOAD(REG_MKLDNN_FN, __VA_ARGS__)
# define REG_MKLDNN_FN_1(name) MKLDNN_OBJ_BUILDER(name, &name::pd_t::create)
# define REG_MKLDNN_FN_2(name, T1) MKLDNN_OBJ_BUILDER(name ## _ ## T1, &name<T1>::pd_t::create)
# define REG_MKLDNN_FN_3(name, T1, T2) MKLDNN_OBJ_BUILDER(name ## _ ## T1 ## _ ## T2, &name<T1, T2>::pd_t::create)
# define REG_MKLDNN_FN_4(name, T1, T2, T3) MKLDNN_OBJ_BUILDER(name ## _ ## T1 ## _ ## T2 ## _ ## T3, &name<T1, T2, T3>::pd_t::create)
# define REG_MKLDNN_FN_5(name, T1, T2, T3, T4) MKLDNN_OBJ_BUILDER(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4, &name<T1, T2, T3, T4>::pd_t::create)
# define REG_MKLDNN_FN_6(name, T1, T2, T3, T4, T5) MKLDNN_OBJ_BUILDER(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4 ## _ ## T5, &name<T1, T2, T3, T4, T5>::pd_t::create)
# define REG_MKLDNN_FN_7(name, T1, T2, T3, T4, T5, T6) MKLDNN_OBJ_BUILDER(name ## _ ## T1 ## _ ## T2 ## _ ## T3 ## _ ## T4 ## _ ## T5 ## _ ## T6, &name<T1, T2, T3, T4, T5, T6>::pd_t::create)

#define MKLDNN_CSCOPE(region, ...) OV_SCOPE(MKLDNN, region, __VA_ARGS__)

#else

# define REG_MKLDNN_FN(...) MKLDNN_MACRO_OVERLOAD(REG_MKLDNN_FN, __VA_ARGS__)
# define REG_MKLDNN_FN_1(name) name::pd_t::create,
# define REG_MKLDNN_FN_2(name, T1) name<T1>::pd_t::create,
# define REG_MKLDNN_FN_3(name, T1, T2) name<T1, T2>::pd_t::create,
# define REG_MKLDNN_FN_4(name, T1, T2, T3) name<T1, T2, T3>::pd_t::create,
# define REG_MKLDNN_FN_5(name, T1, T2, T3, T4) name<T1, T2, T3, T4>::pd_t::create,
# define REG_MKLDNN_FN_6(name, T1, T2, T3, T4, T5) name<T1, T2, T3, T4, T5>::pd_t::create,
# define REG_MKLDNN_FN_7(name, T1, T2, T3, T4, T5, T6) name<T1, T2, T3, T4, T5, T6>::pd_t::create,

#define MKLDNN_CSCOPE(region, ...) __VA_ARGS__

#endif
