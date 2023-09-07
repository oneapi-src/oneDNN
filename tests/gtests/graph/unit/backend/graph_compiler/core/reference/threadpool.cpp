
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
#include <util/def.hpp>
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
#include "graph/unit/unit_test_common.hpp"
#include <runtime/context.hpp>
struct gc_env_initializer {
    gc_env_initializer() {
        dnnl::impl::graph::gc::runtime::get_default_stream = []() {
            static auto the_stream = []() {
                dnnl::impl::graph::gc::runtime::stream_t ret
                        = dnnl::impl::graph::gc::runtime::default_stream;
                ::set_test_engine_kind(
                        dnnl::impl::graph::engine_kind_t::dnnl_cpu);
                ret.vtable_.stream = ::get_stream();
                return ret;
            }();
            return &the_stream;
        };
    }
};
static gc_env_initializer gc_test_init;
#endif
