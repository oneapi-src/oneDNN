/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_CONTEXT_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_CONTEXT_HPP
#include <memory>
#include "test_utils.hpp"
#include <compiler/config/context.hpp>
#include <util/utils.hpp>

inline dnnl::impl::graph::gc::context_ptr get_test_ctx() {
    namespace gc = dnnl::impl::graph::gc;
    static gc::context_ptr ctx = []() {
        auto ret = std::make_shared<gc::context_t>(*gc::get_default_context());
        ctx->flags_.mixed_fusion_ = false;
        ctx->flags_.use_cost_model_ = false;
        return ret;
    }();
    return ctx;
}

inline dnnl::impl::graph::gc::context_ptr get_test_ctx_without_amx() {
    namespace gc = dnnl::impl::graph::gc;
    // forcibly disable fAVX512AMXTILE
    if (IS_AMX_AVAILABLE()) {
        static auto ctx = []() {
            auto new_ctx = std::make_shared<gc::context_t>(*get_test_ctx());
            new_ctx->machine_.cpu_flags_.fAVX512AMXTILE = false;
            return new_ctx;
        }();
        return ctx;
    } else {
        return get_test_ctx();
    }
};

#endif
