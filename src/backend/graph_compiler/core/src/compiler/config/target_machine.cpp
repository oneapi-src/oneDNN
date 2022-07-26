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
#include "context.hpp"
#include <compiler/jit/jit.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <runtime/config.hpp>
#include <runtime/context.hpp>
#include <runtime/env_vars.hpp>
#include <unordered_map>
#include <util/string_utils.hpp>
#include <util/utils.hpp>
#ifdef _MSC_VER
//  Windows
#include <intrin.h>
#define cpuid(info, x, y) __cpuidex(info, x, y)
#else
// #include <cpuid.h>
static void cpuid(int info[4], int InfoType, int v2) {
    __cpuid_count(InfoType, v2, info[0], info[1], info[2], info[3]);
}
#endif

#ifndef _WIN32
extern char **environ;
#endif

SC_MODULE(target)
namespace sc {
using namespace env_key;

target_machine_t::target_machine_t(
        type device_type, std::unique_ptr<machine_flags_t> device_flags)
    : device_type_(device_type), device_flags_(std::move(device_flags)) {}

target_machine_t::target_machine_t(const target_machine_t &other)
    : device_type_(other.device_type_), cpu_flags_(other.cpu_flags_) {}

const machine_flags_t &target_machine_t::get_device_flags() const {
    if (device_type_ == type::cpu) { return cpu_flags_; }
    return *device_flags_;
}

static int get_xcr0() {
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0); /* min VS2010 SP1 compiler is required */
#else
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
    return xcr0;
}

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
        target_machine_t tm = get_native_target_machine();
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
                else if (buf == "llvm") {
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
        parse_bool(env_names[SC_SSA_PASSES], flags.ssa_passes_);

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

// sets max_simd_bits
void target_machine_t::set_simd_length_and_max_cpu_threads(cpu_flags_t &flg) {
    if (flg.fAVX512F)
        flg.max_simd_bits = 512;
    else if (flg.fAVX2 || flg.fAVX)
        flg.max_simd_bits = 256;
    else if (flg.fSSE)
        flg.max_simd_bits = 128;
    else
        flg.max_simd_bits = 0;
}

uint32_t machine_flags_t::get_max_vector_lanes(sc_data_etype etype) const {
    return max_simd_bits / 8 / utils::get_sizeof_etype(etype);
}

uint32_t context_t::get_max_vector_lanes(sc_data_etype etype) const {
    return machine_.get_device_flags().get_max_vector_lanes(etype);
}

context_t::context_t(const scflags_t &flags, target_machine_t &&machine,
        runtime::engine_t *engine)
    : engine_(engine ? engine : runtime::get_default_stream()->engine_)
    , flags_(flags)
    , machine_(std::move(machine)) {}

target_machine_t get_native_target_machine() {
    target_machine_t tm(target_machine_t::type::cpu, nullptr);
    int xcr0 = get_xcr0();
    int info[4];
    cpuid(info, 0, 0);
    int nIds = info[0];

    cpuid(info, 0x80000000, 0);
    unsigned nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001) {
        cpuid(info, 0x00000001, 0);
        tm.cpu_flags_.fMMX = (info[3] & ((int)1 << 23)) != 0;
        tm.cpu_flags_.fSSE = (info[3] & ((int)1 << 25)) != 0;
        tm.cpu_flags_.fSSE2 = (info[3] & ((int)1 << 26)) != 0;
        tm.cpu_flags_.fSSE3 = (info[2] & ((int)1 << 0)) != 0;

        tm.cpu_flags_.fSSSE3 = (info[2] & ((int)1 << 9)) != 0;
        tm.cpu_flags_.fSSE41 = (info[2] & ((int)1 << 19)) != 0;
        tm.cpu_flags_.fSSE42 = (info[2] & ((int)1 << 20)) != 0;
        tm.cpu_flags_.fAES = (info[2] & ((int)1 << 25)) != 0;

        tm.cpu_flags_.fAVX
                = ((xcr0 & 0x6) == 0x6) && (info[2] & ((int)1 << 28)) != 0;
        tm.cpu_flags_.fFMA3 = (info[2] & ((int)1 << 12)) != 0;

        tm.cpu_flags_.fRDRAND = (info[2] & ((int)1 << 30)) != 0;
    }
    if (nIds >= 0x00000007) {
        cpuid(info, 0x00000007, 0);
        tm.cpu_flags_.fAVX2
                = tm.cpu_flags_.fAVX && (info[1] & ((int)1 << 5)) != 0;

        tm.cpu_flags_.fBMI1 = (info[1] & ((int)1 << 3)) != 0;
        tm.cpu_flags_.fBMI2 = (info[1] & ((int)1 << 8)) != 0;
        tm.cpu_flags_.fADX = (info[1] & ((int)1 << 19)) != 0;
        tm.cpu_flags_.fSHA = (info[1] & ((int)1 << 29)) != 0;
        tm.cpu_flags_.fPREFETCHWT1 = (info[2] & ((int)1 << 0)) != 0;

        tm.cpu_flags_.fAVX512F
                = ((xcr0 & 0xe6) == 0xe6) && (info[1] & ((int)1 << 16)) != 0;
        bool hasavx512 = tm.cpu_flags_.fAVX512F;
        tm.cpu_flags_.fAVX512CD = hasavx512 && (info[1] & ((int)1 << 28)) != 0;
        tm.cpu_flags_.fAVX512PF = hasavx512 && (info[1] & ((int)1 << 26)) != 0;
        tm.cpu_flags_.fAVX512ER = hasavx512 && (info[1] & ((int)1 << 27)) != 0;
        tm.cpu_flags_.fAVX512VL = hasavx512 && (info[1] & ((int)1 << 31)) != 0;
        tm.cpu_flags_.fAVX512BW = hasavx512 && (info[1] & ((int)1 << 30)) != 0;
        tm.cpu_flags_.fAVX512DQ = hasavx512 && (info[1] & ((int)1 << 17)) != 0;
        tm.cpu_flags_.fAVX512IFMA
                = hasavx512 && (info[1] & ((int)1 << 21)) != 0;
        tm.cpu_flags_.fAVX512VNNI
                = hasavx512 && (info[2] & ((int)1 << 11)) != 0;
        tm.cpu_flags_.fAVX512AMXBF16
                = hasavx512 && (info[3] & ((int)1 << 22)) != 0;
        tm.cpu_flags_.fAVX512AMXTILE
                = hasavx512 && (info[3] & ((int)1 << 24)) != 0;
        tm.cpu_flags_.fAVX512AMXINT8
                = hasavx512 && (info[3] & ((int)1 << 25)) != 0;
        tm.cpu_flags_.fAVX512VBMI = hasavx512 && (info[2] & ((int)1 << 1)) != 0;

        if (hasavx512) {
            int info2[4];
            cpuid(info2, 0x00000007, 1);
            tm.cpu_flags_.fAVX512BF16 = (info2[0] & ((int)1 << 5)) != 0;
        }
    }
    if (nExIds >= 0x80000001) {
        cpuid(info, 0x80000001, 0);
        tm.cpu_flags_.fx64 = (info[3] & ((int)1 << 29)) != 0;
        tm.cpu_flags_.fABM = (info[2] & ((int)1 << 5)) != 0;
        tm.cpu_flags_.fSSE4a = (info[2] & ((int)1 << 6)) != 0;
        tm.cpu_flags_.fFMA4 = (info[2] & ((int)1 << 16)) != 0;
        tm.cpu_flags_.fXOP = (info[2] & ((int)1 << 11)) != 0;
    }
    target_machine_t::set_simd_length_and_max_cpu_threads(tm.cpu_flags_);
    return tm;
}

} // namespace sc
