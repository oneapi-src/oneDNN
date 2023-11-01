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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_CONTEXT_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_CONTEXT_HPP
#include <memory>
#include <string>
#include <runtime/target_machine.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct engine_t;
}

enum class jit_kind {
    cfake = 0,
#if defined(SC_LLVM_BACKEND)
    llvm,
#endif
    xbyak,
};

enum class sc_opt_level : int { lv0 = 0, lv1, lv2, lv3 };
enum class fusion_opt_level : int { lv0 = 0, lv1, lv2, lv3 };

struct scflags_t {
    enum class brgemm_t : int { dnnl = 0, max_num };

    jit_kind jit_kind_ = jit_kind::cfake;
    sc_opt_level opt_level_ = sc_opt_level::lv3;
    int backend_opt_level_ = 3;
    bool tensor_inplace_ = true;
    bool bf16_fast_trunc_ = false;
    bool const_share_ = true;
    bool trace_ = false;
    bool dead_write_elimination_ = true;
    int buffer_schedule_ = 3; // 0 off, 1 whole reuse, 2 size first, 3 hot first
    brgemm_t brgemm_backend_ = brgemm_t::dnnl;
    int kernel_optim_ = 1; // 0 off, 1 external-runtime-oriented opt,
    bool index2var_ = true;
    bool tensor2var_ = true;
    bool print_ir_ = false;
    bool ssa_passes_ = false;
    bool prefetch_ = true;
    fusion_opt_level fusion_level_ = fusion_opt_level::lv3;
    bool use_cost_model_ = true;
    bool debug_info_ = false;
    // whether jit supports directly generating amx intrinsics instead of using
    // dnnl
    bool jit_support_amx_intrinsics_ = false;
    bool concat_optimization_ = true;
    bool graph_default_private_ = true;
};

struct context_t {
    runtime::engine_t *engine_;
    scflags_t flags_;
    runtime::target_machine_t machine_;
    context_t(const scflags_t &flags, runtime::target_machine_t &&machine,
            runtime::engine_t *engine = nullptr);
    context_t(const context_t &) = default;
    uint16_t get_max_vector_lanes(sc_data_etype etype) const;
    bool use_amx() const;
};
using context_ptr = std::shared_ptr<context_t>;

SC_API context_ptr make_context_from_env();
SC_API context_ptr get_default_context();
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
