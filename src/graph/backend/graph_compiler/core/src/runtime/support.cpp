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

#include <cmath>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.hpp"
#include <runtime/data_type.hpp>
#include <runtime/env_var.hpp>
#include <runtime/env_vars.hpp>
#include <runtime/logging.hpp>
#include <runtime/managed_thread_pool_exports.hpp>
#include <runtime/os.hpp>
#include <runtime/parallel.hpp>
#include <runtime/runtime.hpp>
#include <util/def.hpp>
#include <util/os.hpp>
#include <util/string_utils.hpp>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

SC_MODULE(runtime.support)

using namespace dnnl::impl::graph::gc;
extern "C" void print_float(float f) {
    printf("%f\n", f);
}

extern "C" void print_index(uint64_t f) {
    printf("%llu\n", static_cast<unsigned long long>(f)); // NOLINT
}

extern "C" void print_int(int f) {
    printf("%d\n", f);
}

extern "C" void print_str(char *f) {
    fputs(f, stdout);
}

extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align) {
    return aligned_alloc(align, (sz / align + 1) * align);
}

extern "C" void sc_global_aligned_free(void *ptr, size_t align) {
    aligned_free(ptr);
}

alignas(64) extern const uint32_t sc_log_const_int_vals_1[32] = {0x3f800000,
        0x3f780000, 0x3f700000, 0x3f680000, 0x3f600000, 0x3f580000, 0x3f580000,
        0x3f500000, 0x3f480000, 0x3f480000, 0x3f400000, 0x3f400000, 0x3f380000,
        0x3f380000, 0x3f300000, 0x3f300000, 0x3fa80000, 0x3fa80000, 0x3fa00000,
        0x3fa00000, 0x3fa00000, 0x3f980000, 0x3f980000, 0x3f900000, 0x3f900000,
        0x3f900000, 0x3f900000, 0x3f880000, 0x3f880000, 0x3f880000, 0x3f800000,
        0x3f800000};
alignas(64) extern const uint32_t sc_log_const_int_vals_2[32] = {0xc2b00f34,
        0xc2affef2, 0xc2afee29, 0xc2afdccd, 0xc2afcad6, 0xc2afb837, 0xc2afb837,
        0xc2afa4e4, 0xc2af90cf, 0xc2af90cf, 0xc2af7be9, 0xc2af7be9, 0xc2af661e,
        0xc2af661e, 0xc2af4f5c, 0xc2af4f5c, 0xc2b09a6f, 0xc2b09a6f, 0xc2b08174,
        0xc2b08174, 0xc2b08174, 0xc2b06731, 0xc2b06731, 0xc2b04b82, 0xc2b04b82,
        0xc2b04b82, 0xc2b04b82, 0xc2b02e3e, 0xc2b02e3e, 0xc2b02e3e, 0xc2b00f34,
        0xc2b00f34};
alignas(64) extern const uint32_t sc_erf_const_int_vals[6][32] = {
        {0xa6f2cb94, 0x32827792, 0x3381cc0c, 0x34523d4a, 0x351ac44d, 0x35f36d88,
                0x36ee8229, 0x37b8a3bb, 0x3867a213, 0x3940033b, 0x3a2a5a1d,
                0x3ae35863, 0x3b7828f2, 0x3c08b14b, 0x3c515ed3, 0xbb503236,
                0xbd8d8e5e, 0xbe8abcd9, 0xbf0c19a2, 0xbeccb328, 0x3e176ced,
                0x3f470d99, 0x3f7abb28, 0x3f800000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000},
        {0x3f4c422a, 0x3f4c421f, 0x3f4c4207, 0x3f4c41cb, 0x3f4c413b, 0x3f4c3fad,
                0x3f4c3a2f, 0x3f4c2d40, 0x3f4c146a, 0x3f4bc341, 0x3f4ad08c,
                0x3f48f8cf, 0x3f45fac7, 0x3f404e07, 0x3f3b980f, 0x3f48dff3,
                0x3f78b21b, 0x3fbb0704, 0x40019c32, 0x3fe536d6, 0x3f81331e,
                0x3e6c8684, 0x3c98f936, 0x00000000, 0x3f800000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000},
        {0xb62173f4, 0x3735e4cf, 0x37f2ff89, 0x388c23be, 0x3917535c, 0x39ab2ab0,
                0x3a60fadb, 0x3af9b960, 0x3b6e5491, 0x3c0a4ec5, 0x3ca5aa8c,
                0x3d2138d9, 0x3d8737d4, 0x3ddfb660, 0x3e0f27ab, 0x3d94004b,
                0xbe0efdeb, 0xbf1d96c3, 0xbf89db58, 0xbf6d9897, 0xbef69fb8,
                0xbdc4f8a8, 0xbbde6422, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000},
        {0xbe081a19, 0xbe084570, 0xbe08639b, 0xbe089837, 0xbe08f409, 0xbe09ab95,
                0xbe0b66d0, 0xbe0e400a, 0xbe124df8, 0xbe1bde02, 0xbe2f19c9,
                0xbe4931bf, 0xbe685fbc, 0xbe89c95f, 0xbe96cbca, 0xbe8044aa,
                0xbe0550f2, 0x3dcfd6a1, 0x3e94c826, 0x3e79345f, 0x3decec91,
                0x3ca46568, 0x3aa1e00a, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000},
        {0xba3d61db, 0x39f097a3, 0x3a5845dc, 0x3ab1fa35, 0x3b0cefb8, 0x3b653ab6,
                0x3bcae527, 0x3c221712, 0x3c6c5840, 0x3cc0a703, 0x3d1dcc19,
                0x3d63656d, 0x3d955907, 0x3dbf9910, 0x3dd53f69, 0x3db7dcef,
                0x3d639ebe, 0xba6ede48, 0xbd22be69, 0xbd041cf1, 0xbc64f5ab,
                0xbb097a32, 0xb8ebf380, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000},
        {0x3cb7d80c, 0x3c9b6050, 0x3c978d11, 0x3c92e850, 0x3c8d058b, 0x3c848454,
                0x3c6cd623, 0x3c4c824b, 0x3c2a7935, 0x3be0b390, 0x3b0651ac,
                0xbb232f53, 0xbbd42fa0, 0xbc2c5366, 0xbc492c9e, 0xbc2a7aa6,
                0xbbd55d04, 0xba823a76, 0x3b102aa8, 0x3ae25a7e, 0x3a31f792,
                0x38b84375, 0x3689bb5a, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000}};

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
size_t get_os_page_size() {
#ifdef _WIN32
    // fix-me: (win32) impl
    return 4096;
#else
    static size_t v = getpagesize();
    return v;
#endif
}
} // namespace runtime

runtime_config_t &runtime_config_t::get() noexcept {
    static runtime_config_t cfg {};
    return cfg;
}

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_CUSTOM
extern int get_max_threadpool_concurrency();
#define SET_NUM_THREADS_SUCCESS(NUM) ((NUM) <= get_max_threadpool_concurrency())
#else
// We are free to set num threads in TBB and OMP
#define SET_NUM_THREADS_SUCCESS(NUM) true
#endif

bool runtime_config_t::set_num_threads(int num) const {
    thread_pool_table_->set_num_threads(num);
    return SET_NUM_THREADS_SUCCESS(num);
}

using namespace env_key;
runtime_config_t::runtime_config_t() {
    thread_pool_table_ = &sc_pool_table;
    constexpr int default_MTP =
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
            1;
#else
            0;
#endif
    managed_thread_pool_
            = (utils::getenv_int(env_names[SC_MANAGED_THREAD_POOL], default_MTP)
                    != 0);

    if (managed_thread_pool_) {
        thread_pool_table_->parallel_call_managed = &sc_parallel_call_managed;
    }
    trace_initial_cap_ = utils::getenv_int(env_names[SC_TRACE_INIT_CAP], 4096);
    trace_out_path_ = utils::getenv_string(env_names[SC_TRACE]);
    char mode = 0;

    if (trace_out_path_.size() == 1) {
        // for SC_TRACE=1
        mode = trace_out_path_[0] - '0';
        trace_out_path_ = "sctrace.json";
    } else if (trace_out_path_.size() > 2 && trace_out_path_[1] == ',') {
        // for SC_TRACE=2,abc.json
        mode = trace_out_path_[0] - '0';
        trace_out_path_ = trace_out_path_.substr(2);
    } else if (trace_out_path_.empty()) {
        mode = 0;
    } else {
        // for SC_TRACE=abc.json
        mode = 1;
    }

    switch (mode) {
        case 0: trace_mode_ = trace_mode_t::OFF; break;
        case 1: trace_mode_ = trace_mode_t::FAST; break;
        case 2: trace_mode_ = trace_mode_t::KERNEL; break;
        case 3: trace_mode_ = trace_mode_t::MULTI_THREAD; break;
        default:
            trace_mode_ = trace_mode_t::OFF;
            trace_out_path_ = "";
            break;
    }

    constexpr int default_verbose = 0;
    int tmp_get_verbose_level
            = utils::getenv_int(env_names[SC_VERBOSE], default_verbose);
    if (tmp_get_verbose_level < 0 || tmp_get_verbose_level > 2) {
        tmp_get_verbose_level = 0;
    }
    verbose_level_ = tmp_get_verbose_level;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
