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

#include "gpu/jit/v2/conv/planner/planner.hpp"

#include "gpu/jit/v2/conv/model.hpp"
#include "gpu/jit/v2/conv/plan.hpp"
#include "gpu/jit/v2/conv/plan_registry.hpp"
#include "gpu/jit/v2/conv/planner/arg_parser.hpp"
#include "gpu/jit/v2/conv/planner/bench.hpp"
#include "gpu/jit/v2/conv/planner/model_fit.hpp"
#include "gpu/jit/v2/conv/planner/search.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

enum class planner_mode_t {
    undef,
    trace,
    bench,
    search,
};

struct params_t {
    planner_mode_t mode = planner_mode_t::undef;
    kernel_desc_t desc;

    void init_src_tag_default() {
        if (!desc.src_tag.is_empty()) return;
        desc.src_tag = make_conv_layout_tag(tensor_kind_t::src, "nxc:f32");
    }

    void init_wei_tag_default() {
        if (!desc.wei_tag.is_empty()) return;
        desc.wei_tag = make_conv_layout_tag(tensor_kind_t::wei, "gxio:f32");
    }

    void init_dst_tag_default() {
        if (!desc.dst_tag.is_empty()) return;
        desc.dst_tag = make_conv_layout_tag(tensor_kind_t::dst, "nxc:f32");
    }

    void init_iter_tile_default() {
        if (!desc.iter_tile.is_empty()) return;
        switch (desc.prop) {
            case prop_kind::forward:
                desc.iter_tile = prb_tile_t("mb16ic16oc16");
                break;
            case prop_kind::backward_data:
                desc.iter_tile = prb_tile_t("mb16ic16oc16");
                break;
            case prop_kind::backward_weights:
                desc.iter_tile = prb_tile_t("mb16ic16oc16");
                break;
            default: ir_error_not_expected();
        }
    }

    void init_thread_group_tile_default() {
        if (!desc.thread_group_tile.is_empty()) return;
        switch (desc.prop) {
            case prop_kind::forward:
                desc.thread_group_tile = prb_tile_t("oc4ow4");
                break;
            case prop_kind::backward_data:
                desc.thread_group_tile = prb_tile_t("ic4iw4");
                break;
            case prop_kind::backward_weights:
                desc.thread_group_tile = prb_tile_t("ic4oc4");
                break;
            default: ir_error_not_expected();
        }
    }

    void init_loop_nest_default() {
        if (!desc.loop_nest.is_empty()) return;
        desc.loop_nest.add(prb_dims::kw);
        desc.loop_nest.add(prb_dims::kh);
        desc.loop_nest.add(prb_dims::kd);
        desc.loop_nest.add(prb_dims::ic);
    }

    void init_desc_defaults() {
        init_src_tag_default();
        init_wei_tag_default();
        init_dst_tag_default();
        init_iter_tile_default();
        init_thread_group_tile_default();
        init_loop_nest_default();
    }
};

static params_t params;

void init_params(int argc, const char **argv) {
    arg_parser_t parser("planner");
    parser.add_argument("--bench")
            .help("Run benchmarking.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--search")
            .help("Run plan search.")
            .default_value(false)
            .implicit_value(true);
    parser.add_argument("--prop")
            .help("Propagation kind (fwd, bwd_d, bwd_w).")
            .action(ir_utils::str_to_prop_kind)
            .default_value(prop_kind::forward);
    parser.add_argument("--dw")
            .help("Whether the problem is a depthwise convolution.")
            .action(ir_utils::str_to_bool)
            .default_value(false);
    parser.add_argument("--src")
            .help("Source layout tag.")
            .default_value(std::string());
    parser.add_argument("--wei")
            .help("Weights layout tag.")
            .default_value(std::string());
    parser.add_argument("--dst")
            .help("Destination layout tag.")
            .default_value(std::string());
    parser.add_argument("--simd")
            .help("SIMD size.")
            .action(ir_utils::str_to_int)
            .default_value(16);
    parser.add_argument("--regs")
            .help("Number of registers.")
            .action(ir_utils::str_to_int)
            .default_value(128);
    parser.add_argument("--hw")
            .help("Hardware.")
            .action(str_to_hw)
            .default_value(hw_t());
    parser.add_argument("--fma")
            .help("FMA kind.")
            .action(str_to_fma_kind)
            .default_value(fma_kind_t::mad);
    parser.add_argument("--iter")
            .help("Iteration tile.")
            .action(str_to_prb_tile)
            .default_value(prb_tile_t());
    parser.add_argument("--tg")
            .help("Thread group tile.")
            .action(str_to_prb_tile)
            .default_value(prb_tile_t());
    parser.add_argument("--loop-nest")
            .help("Loop nest.")
            .action(str_to_loop_nest)
            .default_value(loop_nest_t());
    parser.add_argument("--a-access")
            .help("Access type for A.")
            .action(str_to_send_kind)
            .default_value(send_kind_t::undef);
    parser.add_argument("--b-access")
            .help("Access type for B.")
            .action(str_to_send_kind)
            .default_value(send_kind_t::undef);
    parser.add_argument("--c-access")
            .help("Access type for C.")
            .action(str_to_send_kind)
            .default_value(send_kind_t::undef);

    parser.parse_args(argc, argv);

    bool do_bench = parser.get<bool>("--bench");
    bool do_search = parser.get<bool>("--search");
    if (do_bench && do_search) {
        std::cout << "Error: --bench and --search are exclusive." << std::endl;
        exit(1);
    }
    if (do_bench) {
        params.mode = planner_mode_t::bench;
    } else if (do_search) {
        params.mode = planner_mode_t::search;
    } else {
        params.mode = planner_mode_t::trace;
    }

    auto &desc = params.desc;
    desc.prop = parser.get<prop_kind_t>("--prop");
    desc.is_dw = parser.get<bool>("--dw");
    desc.src_tag = make_conv_layout_tag(
            tensor_kind_t::src, parser.get<std::string>("--src"));
    desc.wei_tag = make_conv_layout_tag(
            tensor_kind_t::wei, parser.get<std::string>("--wei"));
    desc.dst_tag = make_conv_layout_tag(
            tensor_kind_t::dst, parser.get<std::string>("--dst"));
    desc.hw = parser.get<hw_t>("--hw");
    desc.fma = parser.get<fma_kind_t>("--fma");
    desc.simd = parser.get<int>("--simd");
    desc.regs = parser.get<int>("--regs");
    desc.iter_tile = parser.get<prb_tile_t>("--iter");
    desc.thread_group_tile = parser.get<prb_tile_t>("--tg");
    desc.loop_nest = parser.get<loop_nest_t>("--loop-nest");
    desc.a_access_kind = parser.get<send_kind_t>("--a-access");
    desc.b_access_kind = parser.get<send_kind_t>("--b-access");
    desc.c_access_kind = parser.get<send_kind_t>("--c-access");
    params.init_desc_defaults();
}

void planner_main(int argc, const char **argv) {
    init_params(argc, argv);
    switch (params.mode) {
        case planner_mode_t::trace: {
            auto plan = create_conv_plan(params.desc);
            std::cout << std::endl;
            std::cout << ir_utils::add_tag("plan", plan.str()) << std::endl;
            break;
        }
        case planner_mode_t::bench: {
            auto bd = bench(params.desc);
            auto model = model_fit(bd);
            break;
        }
        case planner_mode_t::search: {
            search(params.desc);
            break;
        }
        default: ir_error_not_expected();
    }
    dump_plan_registry();
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
