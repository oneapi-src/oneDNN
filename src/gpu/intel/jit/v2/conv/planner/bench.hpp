/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_PLANNER_BENCH_HPP
#define GPU_INTEL_JIT_V2_CONV_PLANNER_BENCH_HPP

#include "gpu/intel/jit/v2/conv/bench_data.hpp"
#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"
#include "gpu/intel/jit/v2/conv/plan_registry.hpp"

#include "oneapi/dnnl/dnnl.hpp"

#include <memory>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

class bench_manager_t {
public:
    bench_manager_t()
        : engine_(engine::kind::gpu, 0)
        , stream_(engine_, _stream_flags)
        , hw_(engine_.get()) {}
    const engine &get_engine() const { return engine_; }
    const stream &get_stream() const { return stream_; }
    const hw_t &hw() const { return hw_; }
    ~bench_manager_t();

private:
    static const stream::flags _stream_flags = static_cast<stream::flags>(
            stream_flags::in_order | stream_flags::profiling);

    engine engine_;
    stream stream_;
    hw_t hw_;
};

struct bench_input_params_t {
    static const int default_nprbs = 250;

    hw_t hw;
    prop_kind_t prop;
    layout_tag_t src_tag;
    layout_tag_t wei_tag;
    layout_tag_t dst_tag;
    prb_reqs_t reqs;
    bool is_dw = false;
    type_t bias_type;
    pvar_tile_t tile;
    int nprbs = 0;

    bench_input_params_t() = default;
    bench_input_params_t(const kernel_desc_t &kernel_desc, const hw_t &hw,
            int nprbs = default_nprbs)
        : hw(hw), nprbs(nprbs) {
        prop = kernel_desc.prop;
        src_tag = kernel_desc.src_tag;
        wei_tag = kernel_desc.wei_tag;
        dst_tag = kernel_desc.dst_tag;
        reqs = kernel_desc.reqs();
        is_dw = kernel_desc.is_dw;
        bias_type = kernel_desc.bias_type;
        tile = kernel_desc.iter_tile;
        for (auto &d : kernel_desc.thread_group_tile) {
            tile[d] = tile.get(d, 1) * kernel_desc.thread_group_tile[d];
        }
    }

    problem_t problem() const {
        problem_t prb;
        prb.set_hw(hw);
        prb.set_prop(prop);
        prb.set_src_tag(src_tag);
        prb.set_wei_tag(wei_tag);
        prb.set_dst_tag(dst_tag);
        prb.set_bias_type(bias_type);
        return prb;
    }
};

class bench_runner_impl_t;

class bench_runner_t {
public:
    bench_runner_t(const bench_manager_t &bench_mger,
            const bench_input_params_t &params);
    bench_data_t bench(const kernel_desc_t &kernel_desc);

private:
    std::shared_ptr<bench_runner_impl_t> impl_;
};

bench_data_t bench(const bench_manager_t &bench_mger,
        const kernel_desc_t &kernel_desc,
        int nprbs = bench_input_params_t::default_nprbs);
plan_registry_t::entry_t prepare_plan_registry_entry(
        const bench_manager_t &bench_mger, const kernel_desc_t &kernel_desc);

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
