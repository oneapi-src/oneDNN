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
#include <immintrin.h>
#include <utility>
#include "context.hpp"
#include <compiler/jit/cfake/cfake_jit.hpp>
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
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using namespace env_key;

static void check_within(
        int &val, int lo, int hi, int defaultv, const char *prompt) {
    if (val < lo || val > hi) {
        SC_MODULE_WARN << prompt << val << ", set to default = " << defaultv;
        val = defaultv;
    }
}

// The function reset some cpu flags by DNNL_MAX_CPU_ISA/ONEDNN_MAX_CPU_ISA, cpu
// flags will take the intersection of compiler/target machine/onednn env
// variable/os.
static void reset_cpu_flags_by_dnnl_envs(runtime::target_machine_t &tm) {
    std::string dnnl_isa = utils::getenv_string("DNNL_MAX_CPU_ISA");
    std::string onednn_isa = utils::getenv_string("ONEDNN_MAX_CPU_ISA");
    // DNNL_MAX_CPU_ISA and ONEDNN_MAX_CPU_ISA can not be set at same time.
    assert(dnnl_isa.empty() || onednn_isa.empty());
    std::string &max_isa = onednn_isa.empty() ? dnnl_isa : onednn_isa;
    if (!max_isa.empty()) {
        // copy here for recover if env is invalid
        auto old_cpu_flags = tm.cpu_flags_;
        do {
            // amx fp16
            if (max_isa == "AVX512_CORE_AMX_FP16") { break; }
            tm.cpu_flags_.fAVX512AMXFP16 = false;
            // amx
            if (max_isa == "AVX512_CORE_AMX") { break; }
            tm.cpu_flags_.fAVX512AMXTILE = false;
            tm.cpu_flags_.fAVX512AMXINT8 = false;
            tm.cpu_flags_.fAVX512AMXBF16 = false;
            // avx512 fp16
            if (max_isa == "AVX512_CORE_FP16") { break; }
            tm.cpu_flags_.fAVX512FP16 = false;
            // avx512 bf16
            if (max_isa == "AVX512_CORE_BF16") { break; }
            tm.cpu_flags_.fAVX512BF16 = false;
            // avx512 vnni
            if (max_isa == "AVX512_CORE_VNNI") { break; }
            tm.cpu_flags_.fAVX512VNNI = false;
            // avx512 core
            if (max_isa == "AVX512_CORE") { break; }
            tm.cpu_flags_.fAVX512F = false;
            tm.cpu_flags_.fAVX512BW = false;
            tm.cpu_flags_.fAVX512VL = false;
            tm.cpu_flags_.fAVX512DQ = false;
            // avx vnni 2
            if (max_isa == "AVX2_VNNI_2") { break; }
            // avx vnni
            if (max_isa == "AVX2_VNNI") { break; }
            // avx2
            if (max_isa == "AVX2") { break; }
            tm.cpu_flags_.fAVX2 = false;
            // avx
            if (max_isa == "AVX") { break; }
            tm.cpu_flags_.fAVX = false;
            // sse41
            if (max_isa == "SSE41") { break; }

            // unsupport
            SC_MODULE_WARN << "Unsupported ISA type: " << max_isa;

            // use old cpu flags
            tm.cpu_flags_ = old_cpu_flags;
        } while (false);
        runtime::target_machine_t::set_simd_length_and_max_cpu_threads(
                tm.cpu_flags_);
    }
    // double check amx by syscall
    if (tm.cpu_flags_.fAVX512AMXTILE
            && !dnnl::impl::cpu::x64::amx::is_available()) {
        tm.cpu_flags_.fAVX512AMXTILE = false;
        tm.cpu_flags_.fAVX512AMXINT8 = false;
        tm.cpu_flags_.fAVX512AMXBF16 = false;
    }
}

template <typename T>
static void parse_value(const char *name, T &v) {
    auto strv = utils::getenv_string(name);
    if (!strv.empty()) { v = T(std::stoi(strv)); };
}

context_ptr get_default_context() {
    static auto v = []() {
        runtime::target_machine_t tm = runtime::get_native_target_machine();
        scflags_t flags;

#if defined(SC_LLVM_BACKEND)
        jit_kind jit = jit_kind::llvm;
#elif SC_CFAKE_JIT_ENABLED
        jit_kind jit = jit_kind::cfake;
#else
        jit_kind jit = jit_kind::xbyak;
#endif
        {
            const char *jit_env_var_name = env_names[SC_CPU_JIT];
            const char *cfakejit_switch_name = "c";
            auto buf = utils::getenv_string(jit_env_var_name);
            if (!buf.empty()) {
#if SC_CFAKE_JIT_ENABLED
                if (buf == cfakejit_switch_name) {
                    jit = jit_kind::cfake;
                }
#else
                if (false) {
                    // make compiler happy
                }
#endif
#if SC_BUILTIN_JIT_ENABLED
                else if (buf == "builtin") {
                    jit = jit_kind::xbyak;
                }
#endif
#if defined(SC_LLVM_BACKEND)
                else if (buf == "llvm") {
                    jit = jit_kind::llvm;
                }
#endif
                else {
                    SC_MODULE_WARN << "Bad value for SC_CPU_JIT=" << buf
                                   << ", setting to default value="
#if defined(SC_LLVM_BACKEND)
                                      "llvm";
#elif SC_CFAKE_JIT_ENABLED
                                      "cfake";
#elif SC_BUILTIN_JIT_ENABLED
                                      "builtin";
#endif
                }
            }
        }

        flags.jit_kind_ = jit;
        jit_engine_t::set_target_machine(jit, flags, tm);
        reset_cpu_flags_by_dnnl_envs(tm);
        tm.brgemm_use_amx_ = tm.cpu_flags_.fAVX512AMXTILE
                && (tm.cpu_flags_.fAVX512AMXBF16
                        || tm.cpu_flags_.fAVX512AMXINT8);
        set_runtime_target_machine(tm);
        std::string tracep = utils::getenv_string(env_names[SC_TRACE]);
        if (!tracep.empty() && tracep != "0") {
            flags.trace_ = true;
            SC_MODULE_WARN << "Trace is ON";
        }

        int opt_level = utils::getenv_int(env_names[SC_OPT_LEVEL], 3);
        check_within(
                opt_level, 0, 3, 3, "Bad optimization level in SC_OPT_LEVEL: ");
        flags.opt_level_ = sc_opt_level(opt_level);

        if (opt_level == 0) {
            // disable opt passes
            flags.buffer_schedule_ = 0;
            flags.dead_write_elimination_ = false;
            flags.kernel_optim_ = 0;
            flags.index2var_ = false;
            flags.tensor_inplace_ = false;
        }

        return std::make_shared<context_t>(flags, std::move(tm));
    }();
    return v;
}

uint16_t context_t::get_max_vector_lanes(sc_data_etype etype) const {
    return machine_.get_device_flags().get_max_vector_lanes(etype);
}

bool context_t::use_amx() const {
    return machine_.use_amx();
}

context_t::context_t(const scflags_t &flags,
        runtime::target_machine_t &&machine, runtime::engine_t *engine)
    : engine_(engine ? engine : runtime::get_default_stream()->engine_)
    , flags_(flags)
    , machine_(std::move(machine)) {}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
