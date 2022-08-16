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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_CONTEXT_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_CONFIG_CONTEXT_HPP
#include <memory>
#include <string>
#include <runtime/target_machine.hpp>

namespace sc {
namespace runtime {
struct engine_t;
}

enum class jit_kind {
    cfake = 0,
    llvm,
};

struct scflags_t {
    enum class brgemm_t : int { dnnl = 0, max_num };

    jit_kind jit_kind_ = jit_kind::cfake;
    int backend_opt_level = 3;
    bool bf16_fast_trunc_ = false;
    bool boundary_check_ = false;
    bool trace_ = false;
    bool dead_write_elimination_ = true;
    int buffer_schedule_ = 3; // 0 off, 1 whole reuse, 2 size first, 3 hot first
    brgemm_t brgemm_backend_ = brgemm_t::dnnl;
    bool kernel_optim_ = true;
    bool index2var_ = true;
    bool print_ir_ = false;
    bool ssa_passes_ = false;
    bool brgemm_use_amx_ = false;
    std::string dump_graph_;
    std::string graph_dump_results_;
    bool value_check_ = false;
    bool mixed_fusion_ = true;
    bool use_cost_model_ = true;
};

struct context_t {
    sc::runtime::engine_t *engine_;
    scflags_t flags_;
    runtime::target_machine_t machine_;
    context_t(const scflags_t &flags, runtime::target_machine_t &&machine,
            runtime::engine_t *engine = nullptr);
    context_t(const context_t &) = default;
    uint32_t get_max_vector_lanes(sc_data_etype etype) const;
};
using context_ptr = std::shared_ptr<context_t>;

SC_API context_ptr get_default_context();
} // namespace sc
#endif
