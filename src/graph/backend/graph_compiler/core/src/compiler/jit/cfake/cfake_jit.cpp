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

#include "cfake_jit.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/jit/jit.hpp>
#include <compiler/jit/symbol_resolver.hpp>
#include <runtime/config.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/memorypool.hpp> // to get the path of the runtime library
#include <unordered_map>
#include <util/file.hpp>
#include <util/scoped_timer.hpp>
#include <util/string_utils.hpp>
#include <util/utils.hpp>

#ifdef _WIN32
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

SC_MODULE(cfakejit)
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
using namespace runtime;

#ifdef _WIN32
std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const std::string &inpath, const std::string &outpath,
        statics_table_t &&globals, bool has_generic_wrapper) {
    // fix-me: (win32)
    throw std::runtime_error("make_jit_module().");
}

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const_ir_module_ptr module, bool generate_wrapper) {
    // fix-me: (win32)
    throw std::runtime_error("make_jit_module().");
}

void *cfake_jit_module_code_t::get_address_of_symbol(const std::string &name) {
    // fix-me: (win32)
    throw std::runtime_error("get_address_of_symbol().");
}

cfake_jit_module_code_t::~cfake_jit_module_code_t() {
    // fix-me: (win32)
    throw std::runtime_error("~cfake_jit_module()");
}

#else

static std::string get_include_path_() {
    std::string path;
    char home_path[512];
    if (utils::getenv(
                env_names[env_key::SC_C_INCLUDE], home_path, sizeof(home_path))
            != 0) {
        path = home_path;
    } else {
#ifdef SC_HOME
        path = MACRO_2_STR(SC_HOME) "/src";
#else
        std::cerr << "environment variable " << env_names[env_key::SC_C_INCLUDE]
                  << " is not set";
#endif
    }
    return path;
}

static const std::string &get_include_path() {
    static std::string path = get_include_path_();
    return path;
}

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const std::string &inpath, const std::string &outpath,
        statics_table_t &&globals, bool has_generic_wrapper,
        bool managed_thread_pool) const {
    auto timer = SC_SCOPED_TIMER_INFO("pass.time.cfake_jit", "");
    auto &home_inc = get_include_path();
    const auto &compiler_config = utils::compiler_configs_t::get();
    if (compiler_config.print_gen_code_) {
        std::ifstream f(inpath);
        if (f.is_open()) std::cerr << f.rdbuf();
    }

    const std::string &command = cfake_jit::get_compiler_command();
    // Mandatory compiler options...
    std::vector<std::string> option = {command, "-I", home_inc, "-o", outpath,
            inpath, "-shared", "-fPIC", "-std=c++11", "-DSC_JIT_SOURCE=1"};
#if SC_PROFILING == 1
    option.emplace_back("-g");
#endif

    // Discretionary compiler options...
    const std::string &options_group
            = utils::compiler_configs_t::get().jit_cc_options_;

    std::vector<std::string> discretionary_options;
    if (options_group.empty() || (options_group == "default")) {
        discretionary_options
                = std::vector<std::string> {"-march=native", "-mno-mmx", "-w"};
        assert(opt_level_ >= 0 && opt_level_ <= 3);
        discretionary_options.emplace_back("-O");
        discretionary_options.back() += std::to_string(opt_level_);
        const auto &envflags = utils::compiler_configs_t::get().cpu_jit_flags_;
        for (const auto &i : envflags) {
            discretionary_options.emplace_back(i);
        }
        if (debug_info_) { discretionary_options.emplace_back("-g"); }
    } else if (options_group == "xbyak-dev") {
        discretionary_options = std::vector<std::string> {"-O3",
                "-march=native",

                // Produce assembly that our (WIP) maint/dev/gnu-as-to-xbyak.py
                // script knows how to parse.
                "-x c++", "-save-temps=obj", "-fverbose-asm", "-masm=intel",

                // Suppress object code that's too complicated for us to
                // replicate via Xbyak (at least for the moment).
                // TODO(xxx): It's possible that some of the following options
                // are unnecessary for that goal.
                "-fno-unwind-tables", "-fno-asynchronous-unwind-tables",
                "-fno-rtti", "-fno-exceptions", "-fno-stack-protector"};
    } else {
        throw std::runtime_error(
                "Unsupported value for env var SC_JIT_CC_OPTIONS_GROUP.");
    }
    option.insert(option.end(), discretionary_options.begin(),
            discretionary_options.end());

    int exit_status;
    bool success
            = utils::create_process_and_await(command, option, exit_status);
    void *compiled_module = nullptr;
    if (success) {
        if (exit_status) {
            std::ostringstream os;
            os << "c compiler returns non-zero code: " << exit_status;
            throw std::runtime_error(os.str());
        }
        compiled_module = dlopen(outpath.c_str(), RTLD_LAZY);
        if (!compiled_module) {
            std::ostringstream os;
            os << "dlopen: " << dlerror();
            throw std::runtime_error(os.str());
        }
        for (auto &kv : get_runtime_function_map()) {
            void **ptr = reinterpret_cast<void **>(
                    dlsym(compiled_module, (kv.first + "_fptr").c_str()));
            if (ptr) { *ptr = kv.second; }
        }
        typedef void (*init_func_t)(void *ctx, void *mod);
        auto init_func = reinterpret_cast<init_func_t>(
                dlsym(compiled_module, "__sc_init__"));
        if (init_func) { init_func(nullptr, globals.data_.data_); }
    } else {
        // If we call 'unlink', it will overwrite errno.
        const int fork_errno = errno;
        unlink(inpath.c_str());

        std::ostringstream os;
        os << "Error when fork: " << utils::get_error_msg(fork_errno);
        throw std::runtime_error(os.str());
    }
    std::shared_ptr<cfake_jit_module_code_t> code(
            new cfake_jit_module_code_t(compiled_module, inpath, outpath,
                    has_generic_wrapper, managed_thread_pool));
    return std::make_shared<jit_module>(std::move(globals), code);
}

statics_table_t cfake_jit::codegen_to_cpp(std::ostream &os,
        const_ir_module_ptr &new_mod, const const_ir_module_ptr &module,
        bool generate_wrapper, bool &out_managed_thread_pool,
        c_generator_optional_out_t *optout) {
    auto gen = create_c_generator(os, context_, generate_wrapper, optout);
    new_mod = gen(module);
    out_managed_thread_pool = new_mod->attr_.get<bool>(
            ir_module_t::attr_key_t::MANAGED_THREAD_POOL);
    return std::move(*new_mod->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS));
}

statics_table_t cfake_jit::codegen_to_cpp(std::ostream &os,
        const_ir_module_ptr &new_mod, const const_ir_module_ptr &module,
        bool generate_wrapper) {
    bool dummy;
    return codegen_to_cpp(os, new_mod, module, generate_wrapper, dummy);
}

std::shared_ptr<jit_module> cfake_jit::make_jit_module(
        const_ir_module_ptr module, bool generate_wrapper) {
    auto unique_name = utils::get_unique_name_for_file();
    // If we're invoking gcc/g++ with the "-save-temps=obj" option, we
    // want to have just one "." character in the name. Otherwise it
    // seems to get confused and the saved intermediate files lack the
    // unique_name part.
    const auto &tmpdir = utils::compiler_configs_t::get_temp_dir_path();
    std::string outpath = tmpdir + "/cfake_jit_module-" + unique_name + ".so";
    std::string inpath = tmpdir + "/cfake_jit_module-" + unique_name + ".cpp";

    std::ofstream of;
    utils::open_file_for_write(of, inpath);

    const auto &compiler_config = utils::compiler_configs_t::get();
    std::ofstream dump_main_f;
    std::ofstream dump_header_f;
    std::ofstream dump_data_f;
    c_generator_optional_out_t optional_dump {
            &dump_main_f, &dump_header_f, &dump_data_f};
    c_generator_optional_out_t *ptr_optional_dump = nullptr;
    if (!compiler_config.dump_gen_code_.empty()) {
        std::string dump_path_base = compiler_config.dump_gen_code_ + '/';
        dump_path_base
                += module->attr_.get_or_else(ir_module_t::attr_key_t::NAME,
                        "/cfake_jit_module-" + unique_name);

        std::string dump_path = dump_path_base + ".cpp";
        std::string dump_header_path = dump_path_base + ".hpp";
        std::string dump_data_path = dump_path_base + "_data.cpp";
        utils::open_file_for_write(dump_main_f, dump_path);
        utils::open_file_for_write(dump_header_f, dump_header_path);
        utils::open_file_for_write(dump_data_f, dump_data_path);
        ptr_optional_dump = &optional_dump;
    }

    bool managed_thread_pool;
    const_ir_module_ptr new_mod;
    auto attr_table = codegen_to_cpp(of, new_mod, module, generate_wrapper,
            managed_thread_pool, ptr_optional_dump);
    of.close();

    auto ret = make_jit_module(inpath, outpath, std::move(attr_table),
            generate_wrapper, managed_thread_pool);
    ret->code_->postprocess(new_mod, ret->globals_);
    return ret;
}

void *cfake_jit_module_code_t::get_address_of_symbol(const std::string &name) {
    return dlsym(module_, name.c_str());
}

cfake_jit_module_code_t::~cfake_jit_module_code_t() {
    if (module_) {
        dlclose(module_);

        unlink(path_.c_str());
        unlink(src_path_.c_str());

        module_ = nullptr;
    }
}

#endif

void *cfake_jit_module_code_t::get_function(
        const std::string &name, void *&wrapper) {
    void *fun = get_address_of_symbol(name);
    wrapper = get_address_of_symbol(name + "_0wrapper");
    return fun;
}

template <typename T, typename TF>
constexpr uintptr_t myoffsetof(TF T::*fld) {
    return reinterpret_cast<uintptr_t>(&(*(T *)nullptr.*fld));
}
#define foffset(F) myoffsetof(&cpu_flags_t::F)

static const std::unordered_map<std::string, uintptr_t> &
get_compiler_flag_map() {
    static std::unordered_map<std::string, uintptr_t> ret = {
            {"__MMX__", foffset(fMMX)},
            {"__x86_64__", foffset(fx64)},
            {"__ABM__", foffset(fABM)},
            {"__RDRND__", foffset(fRDRAND)},
            {"__BMI__", foffset(fBMI1)},
            {"__BMI2__", foffset(fBMI2)},
            {"__ADX__", foffset(fADX)},
            {"__PREFETCHWT1__", foffset(fPREFETCHWT1)},

            {"__SSE__", foffset(fSSE)},
            {"__SSE2__", foffset(fSSE2)},
            {"__SSE3__", foffset(fSSE3)},
            {"__SSSE3__", foffset(fSSSE3)},
            {"__SSE4_1__", foffset(fSSE41)},
            {"__SSE4_2__", foffset(fSSE42)},
            {"__SSE4A__", foffset(fSSE4a)},
            {"__AES__", foffset(fAES)},
            {"__SHA__", foffset(fSHA)},

            {"__AVX__", foffset(fAVX)},
            {"__XOP__", foffset(fXOP)},
            {"__FMA__", foffset(fFMA3)},
            {"__FMA4__", foffset(fFMA4)},
            {"__AVX2__", foffset(fAVX2)},

            {"__AVX512F__", foffset(fAVX512F)},
            {"__AVX512CD__", foffset(fAVX512CD)},
            {"__AVX512PF__", foffset(fAVX512PF)},
            {"__AVX512ER__", foffset(fAVX512ER)},
            {"__AVX512VL__", foffset(fAVX512VL)},
            {"__AVX512BW__", foffset(fAVX512BW)},
            {"__AVX512DQ__", foffset(fAVX512DQ)},
            {"__AVX512IFMA__", foffset(fAVX512IFMA)},
            {"__AVX512VBMI__", foffset(fAVX512VBMI)},
            {"__AVX512BF16__", foffset(fAVX512BF16)},
            {"__AVX512FP16__", foffset(fAVX512FP16)},

            {"__AMX_FP16__", foffset(fAVX512AMXFP16)},
            {"__AMX_BF16__", foffset(fAVX512AMXBF16)},
            {"__AMX_INT8__", foffset(fAVX512AMXTILE)},
            {"__AMX_TILE__", foffset(fAVX512AMXINT8)},
    };
    return ret;
}

static bool &get_flag_field(cpu_flags_t &flg, uintptr_t diff) {
    return *(bool *)((char *)&flg + diff);
}

static cpu_flags_t do_get_compiler_flags() {
    cpu_flags_t ret;
    const auto &flagmap = get_compiler_flag_map();
    std::vector<std::string> option = {cfake_jit::get_compiler_command(),
            "-march=native", "-dM", "-E", "-x", "c++", "-"};
    int exit_status;
    std::string rstdout, rstdin; // empty input
    bool success
            = utils::create_process_and_await(cfake_jit::get_compiler_command(),
                    option, exit_status, &rstdin, &rstdout);
    if (success && !exit_status) {
        for (auto &v : utils::string_split(rstdout, " ")) {
            if (!v.empty() && v[0] == '_') {
                auto itr = flagmap.find(v);
                if (itr != flagmap.end()) {
                    get_flag_field(ret, itr->second) = true;
                }
            }
        }
        target_machine_t::set_simd_length_and_max_cpu_threads(ret);
        return ret;
    }
    ret.max_simd_bits = 0;
    SC_WARN << "Cannot run g++ to detect SIMD length!\n";
    return ret;
}

const cpu_flags_t &cfake_jit::get_compiler_flags() {
    static auto flags = do_get_compiler_flags();
    return flags;
}

std::string &cfake_jit::get_compiler_command() {
    static std::string cmd = []() { return std::string("g++"); }();
    return cmd;
}

void cfake_jit::set_target_machine(target_machine_t &tm) {
    auto flg_map = get_compiler_flag_map();
    auto f = get_compiler_flags();
    f.dataCacheLevels_ = tm.cpu_flags_.dataCacheLevels_;
    f.dataCacheSize_ = tm.cpu_flags_.dataCacheSize_;
    f.family = tm.cpu_flags_.family;
    f.model = tm.cpu_flags_.model;
    f.step = tm.cpu_flags_.step;
    for (auto &itr : flg_map) {
        if (get_flag_field(tm.cpu_flags_, itr.second)
                != get_flag_field(f, itr.second)) {
            SC_MODULE_WARN << "The flag " << itr.first
                           << " is enabled on your hardware but is "
                              "disabled by the default compiler.";
        }
    }
    if (tm.cpu_flags_.max_simd_bits > f.max_simd_bits) {
        SC_MODULE_WARN << "The hardware max SIMD length is larger than the "
                          "compiler SIMD length";
    }

    bool vnni_enabled = tm.cpu_flags_.fAVX512VNNI;
    bool amx_f16_enabled = tm.cpu_flags_.fAVX512AMXFP16;
    bool amx_bf16_enabled = tm.cpu_flags_.fAVX512AMXBF16;
    bool amx_tile_enabled = tm.cpu_flags_.fAVX512AMXTILE;
    bool amx_int8_enabled = tm.cpu_flags_.fAVX512AMXINT8;
    tm.cpu_flags_ = f;
    tm.cpu_flags_.fAVX512VNNI = vnni_enabled;
    tm.cpu_flags_.fAVX512AMXBF16 = amx_bf16_enabled;
    tm.cpu_flags_.fAVX512AMXFP16 = amx_f16_enabled;
    tm.cpu_flags_.fAVX512AMXTILE = amx_tile_enabled;
    tm.cpu_flags_.fAVX512AMXINT8 = amx_int8_enabled;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
