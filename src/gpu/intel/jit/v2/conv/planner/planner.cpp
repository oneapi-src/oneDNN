/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/planner/planner.hpp"

#include "gpu/intel/jit/v2/conv/model.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/planner/bench.hpp"
#include "gpu/intel/jit/v2/conv/planner/mkl_iface.hpp"
#include "gpu/intel/jit/v2/conv/planner/model_fit.hpp"
#include "gpu/intel/jit/v2/conv/planner/search.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

enum class planner_mode_t {
    undef,
    trace,
    bench,
    search,
    auto_search,
};

struct params_t {
    planner_mode_t mode = planner_mode_t::undef;
    kernel_desc_t desc;
};

static params_t params;

bool find_remove(const char *arg, std::string &s) {
    auto pos = s.find(arg);
    if (pos == std::string::npos) return false;
    s.replace(pos, std::strlen(arg), "");
    return true;
}

void print_help() {
    std::cout
            << R"(Usage: gpu_conv_planner [--help] [--bench] [--search] [--auto-search] [kernel descriptor arguments]

Optional arguments:
  --help                Shows help message and exits.
  --bench               Runs benchmarking with provided kernel descriptor.
  --search              Runs search, iterate through missing kernel descriptor properties.
  --auto-search         Runs auto-search to rebuild kernel registry.

)";
    std::cout << "Kernel descriptor arguments:" << std::endl;
    kernel_desc_t::show_help();
}

void init_params(
        int argc, const char **argv, const bench_manager_t &bench_mger) {
    std::ostringstream oss;
    for (int i = 1; i < argc; i++)
        oss << " " << argv[i];
    auto cmd_args = oss.str();
    bool has_bench = find_remove("--bench", cmd_args);
    bool has_search = find_remove("--search", cmd_args);
    bool has_auto_search = find_remove("--auto-search", cmd_args);
    bool has_help = (argc == 1) || find_remove("--help", cmd_args);

    if (has_help) {
        print_help();
        exit(0);
    }

    int mode_count = 0;
    mode_count += (int)has_bench;
    mode_count += (int)has_search;
    mode_count += (int)has_auto_search;
    if (mode_count > 1) {
        std::cout << "Error: --bench, --search and --auto-search are exclusive."
                  << std::endl;
        exit(1);
    }
    if (has_bench) {
        params.mode = planner_mode_t::bench;
    } else if (has_search) {
        params.mode = planner_mode_t::search;
    } else if (has_auto_search) {
        params.mode = planner_mode_t::auto_search;
    } else {
        params.mode = planner_mode_t::trace;
    }
    switch (params.mode) {
        case planner_mode_t::search:
        case planner_mode_t::auto_search: (void)mkl_iface_t::instance(); break;
        default: break;
    }
    // Check if conv v2 is enabled.
    bool enable_conv_v2 = gpu_utils::dev_getenv("enable_conv_v2", false);
    if (!enable_conv_v2) {
        std::cout << "Error: conv_v2 is not enabled, set "
                     "enable_conv_v2=1 in environment."
                  << std::endl;
        exit(1);
    }

    if (params.mode != planner_mode_t::auto_search) {
        auto iface = params.desc.cli_iface();
        iface.parse(cmd_args, &params.desc);
        params.desc.set_defaults();
        params.desc.hw = hw_t(bench_mger.get_engine().get());
        problem_t prb;
        prb_tile_t s = problem_t::default_shape();
        prb.set_shape(s);
    }
}

void planner_main(int argc, const char **argv) {
    bench_manager_t bench_mger;
    init_params(argc, argv, bench_mger);
    switch (params.mode) {
        case planner_mode_t::trace: {
            auto plan = create_conv_plan_and_finalize_desc(params.desc);
            std::cout << std::endl;
            std::cout << ir_utils::add_tag("plan", plan.str()) << std::endl;
            break;
        }
        case planner_mode_t::bench: {
            bench(bench_mger, params.desc);
            break;
        }
        case planner_mode_t::auto_search: {
            auto_search(bench_mger);
            break;
        }
        case planner_mode_t::search: {
            search(bench_mger, params.desc);
            break;
        }
        default: ir_error_not_expected();
    }
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
