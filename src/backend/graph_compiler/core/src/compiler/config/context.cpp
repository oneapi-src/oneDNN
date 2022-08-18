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
#include <immintrin.h>
#include <utility>
#include "context.hpp"
#include <compiler/jit/jit.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <runtime/config.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/target_machine.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

#ifndef _WIN32
extern char **environ;
#endif

SC_MODULE(target)
namespace sc {
using namespace env_key;

static void check_within(
        int &val, int lo, int hi, int defaultv, const char *prompt) {
    if (val < lo || val > hi) {
        SC_MODULE_WARN << prompt << val << ", set to default = " << defaultv;
        val = defaultv;
    }
}

static void parse_bool(const char *name, bool &v) {
    auto strv = sc::utils::getenv_string(name);
    if (!strv.empty()) { v = bool(std::stoi(strv)); };
}

context_ptr get_default_context() {
    static auto v = []() {
        runtime::target_machine_t tm = runtime::get_native_target_machine();
        scflags_t flags;

        // todo: set the flags in target machine from the environment vars
        jit_kind jit = jit_kind::llvm;
        {
            const char *jit_env_var_name = "DNNL_GRAPH_SC_CPU_JIT";
            const char *cfakejit_switch_name = "c";
            auto buf = sc::utils::getenv_string(jit_env_var_name);
            if (!buf.empty()) {
#ifdef SC_CFAKE_JIT_ENABLED
                if (buf == cfakejit_switch_name) {
                    jit = jit_kind::cfake;
                }
#else
                if (false) {
                    // make compiler happy
                }
#endif
                else if (buf == "builtin") {
                    jit = jit_kind::xbyak;
                } else if (buf == "llvm") {
                    jit = jit_kind::llvm;
                } else {
                    SC_MODULE_WARN << "Bad value for SC_CPU_JIT=" << buf
                                   << ", setting to default value="
                                      "llvm";
                }
            }
        }
        flags.jit_kind_ = jit;
        jit_engine_t::set_target_machine(jit, tm);
        set_runtime_target_machine(tm);
        std::string tracep = sc::utils::getenv_string(env_names[SC_TRACE]);
        if (!tracep.empty() && tracep != "0") {
            flags.trace_ = true;
            SC_MODULE_WARN << "Trace is ON";
        }

        std::string dumped = sc::utils::getenv_string(env_names[SC_DUMP_GRAPH]);
        if (!dumped.empty()) {
            if (dumped == "1") {
                flags.dump_graph_ = "sc_graph";
            } else {
                flags.dump_graph_ = dumped;
            }
            SC_MODULE_WARN << "Dump graph is ON";
        }

        flags.graph_dump_results_
                = sc::utils::getenv_string(env_names[SC_GRAPH_DUMP_TENSORS]);

        if (sc::utils::getenv_int(env_names[SC_VALUE_CHECK])) {
            flags.value_check_ = true;
            SC_MODULE_WARN << "Enabled value check";
        }

        int opt_level = sc::utils::getenv_int(env_names[SC_OPT_LEVEL], 3);
        check_within(
                opt_level, 0, 3, 3, "Bad optimization level in SC_OPT_LEVEL: ");
        flags.backend_opt_level = opt_level;
        if (opt_level == 0) {
            // disable opt passes
            flags.buffer_schedule_ = 0;
            flags.dead_write_elimination_ = false;
            flags.kernel_optim_ = false;
            flags.index2var_ = false;
        }

        auto buf_sched_str
                = sc::utils::getenv_string(env_names[SC_BUFFER_SCHEDULE]);
        if (!buf_sched_str.empty()) {
            flags.buffer_schedule_ = std::stoi(buf_sched_str);
            check_within(flags.buffer_schedule_, 0, 3, 3,
                    "Bad buffer schedule level in SC_BUFFER_SCHEDULE: ");
        }

        auto backend = sc::utils::getenv_string(env_names[SC_KERNEL]);
        if (backend.empty() || backend == "dnnl") {
            flags.brgemm_backend_ = scflags_t::brgemm_t::dnnl;
        } else {
            flags.brgemm_backend_ = scflags_t::brgemm_t::dnnl;
            SC_MODULE_WARN << "Unknown SC_KERNEL: " << backend
                           << ", set to default = dnnl";
        }

        parse_bool(env_names[SC_MICRO_KERNEL_OPTIM], flags.kernel_optim_);
        parse_bool(env_names[SC_DEAD_WRITE_ELIMINATION],
                flags.dead_write_elimination_);
        parse_bool(env_names[SC_INDEX2VAR], flags.index2var_);
        parse_bool(env_names[SC_PRINT_IR], flags.print_ir_);
        parse_bool(env_names[SC_MIXED_FUSION], flags.mixed_fusion_);
        parse_bool(env_names[SC_COST_MODEL], flags.use_cost_model_);
        parse_bool(env_names[SC_SSA_PASSES], flags.ssa_passes_);
        parse_bool(env_names[SC_XBYAK_JIT_SAVE_OBJ], flags.xbyak_jit_save_obj_);
        parse_bool(env_names[SC_XBYAK_JIT_ASM_LISTING],
                flags.xbyak_jit_asm_listing_);
        parse_bool(env_names[SC_XBYAK_JIT_LOG_STACK_FRAME_MODEL],
                flags.xbyak_jit_log_stack_frame_model_);
        parse_bool(env_names[SC_XBYAK_JIT_PAUSE_AFTER_CODEGEN],
                flags.xbyak_jit_pause_after_codegen_);

        if (sc::utils::getenv_int(env_names[SC_BOUNDARY_CHECK])) {
            flags.boundary_check_ = true;
            SC_MODULE_WARN << "Enabled boundary check";
        };
        flags.brgemm_use_amx_ = tm.cpu_flags_.fAVX512AMXTILE
                && (tm.cpu_flags_.fAVX512AMXBF16
                        || tm.cpu_flags_.fAVX512AMXINT8);
        // todo: this env var is for linux kernels that under 5.15, current left
        // here for compatibility
        if (sc::utils::getenv_string("DNNL_MAX_CPU_ISA") == "AVX512_CORE_AMX") {
            flags.brgemm_use_amx_ = true;
        }

        return std::make_shared<context_t>(flags, std::move(tm));
    }();
    return v;
}

uint32_t context_t::get_max_vector_lanes(sc_data_etype etype) const {
    return machine_.get_device_flags().get_max_vector_lanes(etype);
}

context_t::context_t(const scflags_t &flags,
        runtime::target_machine_t &&machine, runtime::engine_t *engine)
    : engine_(engine ? engine : runtime::get_default_stream()->engine_)
    , flags_(flags)
    , machine_(std::move(machine)) {}

} // namespace sc
