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

#include "gpu/jit/v2/conv/planner/bench.hpp"

#include "common/dnnl_thread.hpp"
#include "gpu/jit/v2/conv/plan.hpp"
#include "gpu/jit/v2/conv/plan_preset.hpp"
#include "gpu/ocl/ocl_usm_utils.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;

extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
extern "C" dnnl_status_t dnnl_query_profiling_data(dnnl_stream_t stream,
        int32_t data_kind, int *num_entries, uint64_t *data);

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

static void fill_mem(stream &strm, const memory &mem) {
    auto eng = mem.get_engine();
    auto *ptr = mem.get_data_handle();
    auto md = mem.get_desc();
    size_t size = md.get_size();
    uint8_t pattern = 0;
    impl::gpu::ocl::usm::fill(strm.get(), ptr, &pattern, sizeof(pattern), size,
            0, nullptr, nullptr);
}

class memory_pool_t {
public:
    std::unordered_map<int, memory> get_args(
            const std::unordered_map<int, memory::desc> &mds) const {
        ir_assert(is_finalized_);
        std::unordered_map<int, memory> ret;
        for (auto &kv : mds) {
            int id = kv.first;
            auto &base_mem = base_mems_.at(id);
            auto &md = kv.second;
            auto eng = base_mem.get_engine();
            ir_assert(md.get_size() <= base_mem.get_desc().get_size());
            auto mem = ocl_interop::make_memory(md, eng,
                    ocl_interop::memory_kind::usm, base_mem.get_data_handle());
            ret.emplace(id, mem);
        }
        return ret;
    }

    void reserve(int id, const memory::desc &md) {
        size_t &size = arg_sizes_[id];
        size = std::max(size, md.get_size());
    }

    void finalize(stream &strm) {
        auto eng = strm.get_engine();
        for (auto &kv : arg_sizes_) {
            int id = kv.first;
            memory::dims dims = {(memory::dim)kv.second};
            memory::desc md(dims, memory::data_type::u8, memory::format_tag::a);
            auto mem = ocl_interop::make_memory(
                    md, eng, ocl_interop::memory_kind::usm);
            fill_mem(strm, mem);
            base_mems_.emplace(id, mem);
        }
        strm.wait();
        is_finalized_ = true;
    }

private:
    bool is_finalized_ = false;
    std::unordered_map<int, size_t> arg_sizes_;
    std::unordered_map<int, memory> base_mems_;
};

class bench_task_base_t {
public:
    static const int iters = 10;

    virtual bool init_primitive(engine &eng) = 0;

    void init_mem(memory_pool_t &mem_pool) {
        for (auto &kv : get_mds()) {
            mem_pool.reserve(kv.first, kv.second);
        }
    }

    dnnl_status_t bench(stream &strm, const memory_pool_t &mem_pool) {
        using namespace dnnl::impl;
        CHECK(dnnl_reset_profiling(strm.get()));
        auto args = mem_pool.get_args(get_mds());
        for (int i = 0; i < iters; i++) {
            prim_.execute(strm, args);
        }
        strm.wait();
        int nentries = 0;
        CHECK(dnnl_query_profiling_data(
                strm.get(), profiling_data_kind::time, &nentries, nullptr));

        assert(nentries == iters);

        std::vector<uint64_t> entries(nentries);
        CHECK(dnnl_query_profiling_data(strm.get(), profiling_data_kind::time,
                &nentries, entries.data()));
        time_ = entries[0];
        for (uint64_t t : entries)
            time_ = std::min(time_, t);

        return status::success;
    }

    uint64_t time() const { return time_; }

protected:
    void set_primitive(const primitive &prim) { prim_ = prim; }

private:
    std::unordered_map<int, memory::desc> get_mds() const {
        auto *pd_ptr
                = const_cast<dnnl_primitive_desc_t>(prim_.get_primitive_desc());
        primitive_desc_base pd(pd_ptr, true);
        std::vector<int> arg_ids = {
                DNNL_ARG_DIFF_DST,
                DNNL_ARG_DIFF_SRC,
                DNNL_ARG_DIFF_WEIGHTS,
                DNNL_ARG_DST,
                DNNL_ARG_SRC,
                DNNL_ARG_WEIGHTS,
        };
        std::unordered_map<int, memory::desc> ret;
        for (int id : arg_ids) {
            auto md = pd.query_md(dnnl::query::exec_arg_md, id);
            if (md.is_zero()) continue;
            ret.emplace(id, md);
        }
        return ret;
    }

    primitive prim_;
    uint64_t time_ = 0;
};

using problem_t = dnnl::impl::gpu::jit::v2::conv::problem_t;
using kernel_desc_t = dnnl::impl::gpu::jit::v2::conv::kernel_desc_t;
using bench_data_t = dnnl::impl::gpu::jit::v2::conv::bench_data_t;
using prb_tile_t = dnnl::impl::gpu::jit::prb_tile_t;
namespace prb_dims = dnnl::impl::gpu::jit::prb_dims;

class bench_task_t : public bench_task_base_t {
public:
    bench_task_t(const problem_t &prb) {
        mb = prb.shape()[prb_dims::mb];
        oc = prb.shape()[prb_dims::oc];
        ic = prb.shape()[prb_dims::ic];
        ih = prb.shape()[prb_dims::ih];
        iw = prb.shape()[prb_dims::iw];
        oh = prb.shape()[prb_dims::oh];
        ow = prb.shape()[prb_dims::ow];
        kh = prb.shape()[prb_dims::kh];
        kw = prb.shape()[prb_dims::kw];
        sh = prb.shape()[prb_dims::sh];
        sw = prb.shape()[prb_dims::sw];
        ph = prb.shape()[prb_dims::ph];
        pw = prb.shape()[prb_dims::pw];
    }

    bool init_primitive(engine &eng) override {
        try {
            memory::dims src_dims = {mb, ic, ih, iw};
            memory::dims wei_dims = {1, oc, ic, kh, kw};
            memory::dims dst_dims = {mb, oc, oh, ow};

            memory::dims strides = {sh, sw};
            memory::dims padding_l = {ph, pw};
            memory::dims padding_r = {ph, pw};

            memory::desc src_md(
                    src_dims, memory::data_type::f32, memory::format_tag::nhwc);
            memory::desc wei_md(wei_dims, memory::data_type::f32,
                    memory::format_tag::ghwio);
            memory::desc dst_md(
                    dst_dims, memory::data_type::f32, memory::format_tag::nhwc);

            primitive_attr attr;
            auto pd = convolution_forward::primitive_desc(eng,
                    dnnl::prop_kind::forward_inference,
                    algorithm::convolution_direct, src_md, wei_md,
                    memory::desc(), dst_md, strides, padding_l, padding_r,
                    attr);
            auto *impl_name = pd.impl_info_str();
            if (strcmp(impl_name, "jit:ir_v2") != 0) {
                std::cout << "Error: expected conv_v2." << std::endl;
                exit(1);
            }
            auto prim = convolution_forward(pd);
            set_primitive(prim);
            return true;
        } catch (dnnl::error &) { return false; }
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "mb" << mb;
        oss << "ic" << ic;
        oss << "ih" << ih;
        oss << "oc" << oc;
        oss << "oh" << oh;
        oss << "kh" << kh;
        if (sh != 1) oss << "sh" << sh;
        oss << "ph" << ph;
        return oss.str();
    }

private:
    memory::dim mb;
    memory::dim oc, ic;
    memory::dim ih, iw;
    memory::dim oh, ow;
    memory::dim kh, kw;
    memory::dim sh, sw;
    memory::dim ph, pw;
};

int random(int a, int b) {
    return a + rand() % (b - a + 1);
}

prb_tile_t random_shape() {
    prb_tile_t s = problem_t::default_shape();
    s[prb_dims::g] = 1;
    s[prb_dims::mb] = random(1, 16);
    s[prb_dims::ic] = random(1, 512);
    s[prb_dims::oc] = random(1, 512);
    s[prb_dims::iw] = s[prb_dims::ow] = random(1, 512);
    return s;
}

std::vector<problem_t> generate_problems(const kernel_desc_t &kd) {
    srand(kd.get_hash());
    ir_assert(!kd.is_dw);
    std::vector<problem_t> ret;
    const int nprbs = 100;
    const int max_iters = (1 << 20);
    for (int iter = 0; iter < max_iters; iter++) {
        problem_t prb;
        prb.set_hw(kd.hw);
        prb.set_prop(kd.prop);
        prb.set_shape(random_shape());
        prb.set_src_tag(kd.src_tag);
        prb.set_wei_tag(kd.wei_tag);
        prb.set_dst_tag(kd.dst_tag);
        if (!kd.fits(prb, /*check_tags=*/false)) continue;
        ir_assert(kd.fits(prb));
        ret.push_back(prb);
        if ((int)ret.size() >= nprbs) break;
    }
    if ((int)ret.size() < nprbs) {
        printf("Could not generate %d problems after %d iterations\n", nprbs,
                max_iters);
    }
    return ret;
}

std::vector<problem_t> load_problems(const std::string &path) {
    std::vector<problem_t> prbs;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        prbs.emplace_back(line);
    }
    return prbs;
}

bench_data_t bench(const kernel_desc_t &_kernel_desc) {
    engine eng(engine::kind::gpu, 0);
    auto kernel_desc = _kernel_desc;
    kernel_desc.hw = hw_t(eng.get());
    if (!kernel_desc.is_supported()) return {};
    auto plan = create_conv_plan(kernel_desc);
    if (!plan) return {};
    kernel_desc.finalize(plan);

    auto prbs = generate_problems(kernel_desc);
    int nprbs = (int)prbs.size();

    std::vector<bench_task_t> tasks;
    for (auto &prb : prbs) {
        tasks.emplace_back(prb);
    }

    {
        auto guard = plan_preset_t::instance().make_guard(kernel_desc);
        if (!tasks[0].init_primitive(eng)) return {};
    }

    parallel_nd(nprbs, [&](dim_t i) {
        auto guard = plan_preset_t::instance().make_guard(kernel_desc);
        bool ok = tasks[i].init_primitive(eng);
        if (!ok) throw std::runtime_error("Initialization failed");
    });

    auto flags = stream_flags::in_order | stream_flags::profiling;
    stream strm(eng, static_cast<stream::flags>(flags));
    memory_pool_t mem_pool;
    for (auto &t : tasks) {
        t.init_mem(mem_pool);
    }
    mem_pool.finalize(strm);

    bench_data_t bd(kernel_desc);
    for (int i = 0; i < nprbs; i++) {
        tasks[i].bench(strm, mem_pool);
        bd.add(prbs[i], tasks[i].time());
    }

    std::cout << bd << std::endl;

    return bd;
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
