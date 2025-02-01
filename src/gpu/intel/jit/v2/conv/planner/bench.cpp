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

#include "gpu/intel/jit/v2/conv/planner/bench.hpp"

#include "common/dnnl_thread.hpp"
#include "gpu/intel/jit/v2/conv/debug.hpp"
#include "gpu/intel/jit/v2/conv/plan.hpp"
#include "gpu/intel/jit/v2/conv/plan_registry.hpp"
#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"
#include "gpu/intel/ocl/usm_utils.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

using namespace dnnl;

#ifndef DNNL_EXPERIMENTAL_PROFILING
extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
extern "C" dnnl_status_t dnnl_query_profiling_data(dnnl_stream_t stream,
        int32_t data_kind, int *num_entries, uint64_t *data);
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {
namespace planner {

bench_manager_t::~bench_manager_t() {
    dump_plan_registry();
}

static void fill_mem(stream &strm, const memory &mem) {
    auto eng = mem.get_engine();
    auto *ptr = mem.get_data_handle();
    auto md = mem.get_desc();
    size_t size = md.get_size();
    uint8_t pattern = 0;
    impl::gpu::intel::ocl::usm::fill(strm.get(), ptr, &pattern, sizeof(pattern),
            size, 0, nullptr, nullptr);
}

class memory_pool_t {
public:
    std::unordered_map<int, memory> get_args(
            const std::unordered_map<int, memory::desc> &mds) const {
        gpu_assert(is_finalized_);
        std::unordered_map<int, memory> ret;
        for (auto &kv : mds) {
            int id = kv.first;
            auto &base_mem = base_mems_.at(id);
            auto &md = kv.second;
            auto eng = base_mem.get_engine();
            gpu_assert(md.get_size() <= base_mem.get_desc().get_size());
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

    operator bool() const { return !base_mems_.empty(); }

private:
    bool is_finalized_ = false;
    std::unordered_map<int, size_t> arg_sizes_;
    std::unordered_map<int, memory> base_mems_;
};

class bench_task_base_t {
public:
    static const int iters = 3;

    void init_mem(memory_pool_t &mem_pool) {
        for (auto &kv : get_mds()) {
            mem_pool.reserve(kv.first, kv.second);
        }
    }

    dnnl_status_t bench_async(stream &strm, const memory_pool_t &mem_pool) {
        using namespace dnnl::impl;
        auto args = mem_pool.get_args(get_mds());
        for (int i = 0; i < iters; i++) {
            prim_.execute(strm, args);
        }
        return status::success;
    }

    template <typename TaskVectorT>
    static dnnl_status_t sync(stream &strm, TaskVectorT &vec) {
        strm.wait();
        int ntasks = (int)vec.size();
        int nentries = 0;
        int nkernels = 0;
        CHECK(dnnl_query_profiling_data(
                strm.get(), profiling_data_kind::time, &nentries, nullptr));
        CHECK(dnnl_query_profiling_data(strm.get(),
                profiling_data_kind::time_per_kernel, &nkernels, nullptr));
        gpu_assert(nentries == ntasks * iters);

        std::vector<uint64_t> entries(nentries);
        std::vector<uint64_t> kernel_entries;
        CHECK(dnnl_query_profiling_data(strm.get(), profiling_data_kind::time,
                &nentries, entries.data()));
        int kernels_per_entry = ir_utils::safe_div(nkernels, nentries);
        if (kernels_per_entry > 1) {
            kernel_entries.resize(nkernels);
            CHECK(dnnl_query_profiling_data(strm.get(),
                    profiling_data_kind::time_per_kernel, &nkernels,
                    kernel_entries.data()));
        }
        auto get_bench_time = [&](int i, int j) {
            int idx = iters * i + j;
            if (kernels_per_entry == 1) return bench_time_t(entries[idx]);
            int beg = idx * kernels_per_entry;
            int end = idx * kernels_per_entry + kernels_per_entry;
            return bench_time_t(entries[idx], kernel_entries.begin() + beg,
                    kernel_entries.begin() + end);
        };
        for (int i = 0; i < ntasks; i++) {
            auto time = get_bench_time(i, 0);
            for (int j = 1; j < iters; j++) {
                auto j_time = get_bench_time(i, j);
                time = time.min(j_time);
            }
            vec[i].set_time(time);
        }
        return status::success;
    }

    const bench_time_t &time() const { return time_; }
    void set_time(const bench_time_t &time) { time_ = time; }

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
    bench_time_t time_;
};

using problem_t = dnnl::impl::gpu::intel::jit::v2::conv::problem_t;
using kernel_desc_t = dnnl::impl::gpu::intel::jit::v2::conv::kernel_desc_t;
using bench_data_t = dnnl::impl::gpu::intel::jit::v2::conv::bench_data_t;
using bench_time_t = dnnl::impl::gpu::intel::jit::v2::conv::bench_time_t;
using pvar_tile_t = dnnl::impl::gpu::intel::jit::pvar_tile_t;
namespace pvars = dnnl::impl::gpu::intel::jit::pvars;

class bench_task_t : public bench_task_base_t {
public:
    bench_task_t(const problem_t &prb) : prb_(prb) {
        g = prb.shape()[pvars::g];
        mb = prb.shape()[pvars::mb];
        oc = prb.shape()[pvars::oc];
        ic = prb.shape()[pvars::ic];
        ih = prb.shape()[pvars::ih];
        iw = prb.shape()[pvars::iw];
        oh = prb.shape()[pvars::oh];
        ow = prb.shape()[pvars::ow];
        kh = prb.shape()[pvars::kh];
        kw = prb.shape()[pvars::kw];
        sh = prb.shape()[pvars::sh];
        sw = prb.shape()[pvars::sw];
        ph = prb.shape()[pvars::ph];
        pw = prb.shape()[pvars::pw];
    }

    const problem_t &prb() const { return prb_; }

    bool init_primitive(engine &eng) {
        try {
            memory::dims src_dims = {mb, g * ic, 1, ih, iw};
            memory::dims wei_dims = {g, oc, ic, 1, kh, kw};
            memory::dims dst_dims = {mb, g * oc, 1, oh, ow};
            memory::dims bias_dims = {g * oc};

            memory::dims strides = {1, sh, sw};
            memory::dims padding_l = {0, ph, pw};
            memory::dims padding_r = {0, ph, pw};

            switch (prb_.prop()) {
                case prop_kind::forward_inference:
                case prop_kind::forward_training: {
                    auto src_md = to_memory_desc(prb_.src_tag(), src_dims);
                    auto wei_md = to_memory_desc(
                            prb_.wei_tag(), wei_dims, /*is_wei=*/true);
                    auto dst_md = to_memory_desc(prb_.dst_tag(), dst_dims);

                    primitive_attr attr;
                    auto pd = convolution_forward::primitive_desc(eng,
                            static_cast<enum prop_kind>(prb_.prop()),
                            algorithm::convolution_direct, src_md, wei_md,
                            memory::desc(), dst_md, strides, padding_l,
                            padding_r, attr);
                    auto *impl_name = pd.impl_info_str();
                    if (strcmp(impl_name, "jit:ir_v2") != 0) {
                        std::cout << "Error: expected conv_v2." << std::endl;
                        exit(1);
                    }
                    auto prim = convolution_forward(pd);
                    set_primitive(prim);
                    return true;
                }
                case prop_kind::backward_data: {
                    auto diff_src_md = to_memory_desc(prb_.src_tag(), src_dims);
                    auto wei_md = to_memory_desc(
                            prb_.wei_tag(), wei_dims, /*is_wei=*/true);
                    auto diff_dst_md = to_memory_desc(prb_.dst_tag(), dst_dims);

                    // Uses the C API as fwd_hint is not currently optional
                    // under the C++ API.
                    primitive_attr attr;
                    dnnl_primitive_desc_t c_pd = nullptr;
                    CHECK(dnnl_convolution_backward_data_primitive_desc_create(
                            &c_pd, eng.get(), alg_kind::convolution_direct,
                            diff_src_md.get(), wei_md.get(), diff_dst_md.get(),
                            &strides[0], nullptr, &padding_l[0], &padding_r[0],
                            nullptr, attr.get()));
                    auto pd = convolution_backward_data::primitive_desc(c_pd);

                    auto *impl_name = pd.impl_info_str();
                    if (strcmp(impl_name, "jit:ir_v2") != 0) {
                        std::cout << "Error: expected conv_v2." << std::endl;
                        exit(1);
                    }
                    auto prim = convolution_backward_data(pd);
                    set_primitive(prim);
                    return true;
                }
                case prop_kind::backward_weights: {
                    auto src_md = to_memory_desc(prb_.src_tag(), src_dims);
                    auto diff_wei_md = to_memory_desc(
                            prb_.wei_tag(), wei_dims, /*is_wei=*/true);
                    auto diff_dst_md = to_memory_desc(prb_.dst_tag(), dst_dims);
                    memory::desc diff_bias_md;
                    if (!prb_.bias_type().is_undef()) {
                        auto tag = make_conv_layout_tag(tensor_kind_t::bias,
                                "a:" + prb_.bias_type().str());
                        diff_bias_md = to_memory_desc(tag, bias_dims);
                    }

                    // Uses the C API as fwd_hint is not currently optional
                    // under the C++ API.
                    primitive_attr attr;
                    dnnl_primitive_desc_t c_pd = nullptr;
                    CHECK(dnnl_convolution_backward_weights_primitive_desc_create(
                            &c_pd, eng.get(), alg_kind::convolution_direct,
                            src_md.get(), diff_wei_md.get(), diff_bias_md.get(),
                            diff_dst_md.get(), &strides[0], nullptr,
                            &padding_l[0], &padding_r[0], nullptr, attr.get()));
                    auto pd = convolution_backward_weights::primitive_desc(
                            c_pd);

                    auto *impl_name = pd.impl_info_str();
                    if (strcmp(impl_name, "jit:ir_v2") != 0) {
                        std::cout << "Error: expected conv_v2." << std::endl;
                        exit(1);
                    }
                    auto prim = convolution_backward_weights(pd);
                    set_primitive(prim);
                    return true;
                }
                default:
                    std::cout << "Error: unexpected propagation kind"
                              << std::endl;
                    exit(1);
            }
        } catch (dnnl::error &e) {
            std::cout << "Initialization Exception: " << e.message << "\n";
            return false;
        }
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "g" << g;
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
    memory::desc to_memory_desc(const layout_tag_t &tag,
            const memory::dims &dims, bool is_wei = false) const {
        auto type = static_cast<dnnl::memory::data_type>(to_dnnl(tag.type()));
        layout_raw_tag_t raw_tags[] = {
                layout_raw_tag_t("a", 1),
                layout_raw_tag_t("axb", 5),
                layout_raw_tag_t("abx", 5),
                layout_raw_tag_t("axbc", 6),
                layout_raw_tag_t("axcb", 6),
        };
        for (auto &raw_tag : raw_tags) {
            if (tag.raw_tag() == raw_tag) {
                memory::dims strides(dims.size());
                memory::dim stride = 1;
                for (int i = raw_tag.nentries() - 1; i >= 0; i--) {
                    auto &e = raw_tag.entries()[i];
                    strides[e.index()] = stride;
                    stride *= dims[e.index()];
                }
                return memory::desc(dims, type, strides);
            }
        }
        gpu_error_not_expected() << "Unknown tag: " << tag.str();
        return memory::desc();
    }

    problem_t prb_;
    memory::dim mb, g;
    memory::dim oc, ic;
    memory::dim ih, iw;
    memory::dim oh, ow;
    memory::dim kh, kw;
    memory::dim sh, sw;
    memory::dim ph, pw;
};

dim_t random(dim_t a, dim_t b) {
    return a + rand() % (b - a + 1);
}

struct random_dim_t {
    dim_t lo = 0;
    dim_t hi = 0;
    dim_t tile = 0;

    random_dim_t(const pvar_t &dim, dim_t _tile) : tile(_tile) {}
    random_dim_t with_range(dim_t _lo, dim_t _hi) {
        auto ret = *this;
        ret.lo = utils::div_up(_lo, tile);
        ret.hi = _hi / tile;
        return ret;
    }
    explicit operator bool() const { return lo <= hi; }
    bool with_tile() const { return tile > 1; }
    dim_t operator()() const {
        gpu_assert(*this);
        return random(lo, hi) * tile;
    }
};

struct random_dim_set_t {
    std::vector<random_dim_t> dims;

    random_dim_set_t(const random_dim_t &d) {
        if (!d) return;
        dims.push_back(d);
    }
    random_dim_set_t operator|(const random_dim_set_t &other) const {
        random_dim_set_t ret = *this;
        ret.dims.insert(ret.dims.end(), other.dims.begin(), other.dims.end());
        return ret;
    }
    size_t size() const { return dims.size(); }
    bool with_tile() const { return dims[0].with_tile(); }
    dim_t operator()() const {
        dim_t idx = random(0, static_cast<dim_t>(size()) - 1);
        return dims[idx]();
    }
};

random_dim_set_t operator|(const random_dim_t &a, const random_dim_set_t &b) {
    return random_dim_set_t(a) | b;
}

pvar_tile_t random_shape(
        prop_kind_t prop, bool is_dw, const pvar_tile_t &tile) {
    auto make_random_dim = [&](const pvar_t &dim, dim_t lo = 0, dim_t hi = 0) {
        auto ret = random_dim_t(dim, tile.get(dim, 1));
        return ret.with_range(lo, hi);
    };
    auto make_random_dim_set
            = [&](const pvar_t &dim, dim_t s, dim_t m, dim_t l) {
                  auto d = make_random_dim(dim);
                  auto d_s = d.with_range(1, s);
                  auto d_m = d.with_range(s + 1, m);
                  auto d_l = d.with_range(m + 1, l);
                  return d_s | d_m | d_l;
              };
    pvar_tile_t s = problem_t::default_shape();
    auto g = make_random_dim(pvars::g, 2, 512);
    auto mb = make_random_dim_set(pvars::mb, 1, 16, 128);
    auto ic = make_random_dim_set(pvars::ic, 64, 512, 2048);
    auto oc = make_random_dim_set(pvars::oc, 64, 512, 2048);
    auto ow = make_random_dim_set(pvars::ow, 64, 512, 2048);
    auto iw = make_random_dim_set(pvars::iw, 64, 512, 2048);
    if (is_dw) {
        s[pvars::g] = g();
        s[pvars::mb] = mb();
        s[pvars::ic] = 1;
        s[pvars::oc] = 1;
        s[pvars::iw] = s[pvars::ow] = (ow.with_tile() ? ow() : iw());
    } else {
        s[pvars::g] = 1;
        s[pvars::mb] = mb();
        s[pvars::ic] = ic();
        s[pvars::oc] = oc();
        s[pvars::iw] = s[pvars::ow] = (ow.with_tile() ? ow() : iw());
    }
    return s;
}

double footprint(const layout_tag_t &src, const layout_tag_t &wei,
        const layout_tag_t &dst, const pvar_tile_t &shape) {
#define GET(name) shape[pvars::name]
    double src_elems
            = (double)GET(g) * GET(mb) * GET(ic) * GET(id) * GET(ih) * GET(iw);
    double wei_elems
            = (double)GET(g) * GET(oc) * GET(ic) * GET(kd) * GET(kh) * GET(kw);
    double dst_elems
            = (double)GET(g) * GET(mb) * GET(oc) * GET(od) * GET(oh) * GET(ow);
#undef GET
    double ret = 0;
    ret += src_elems * src.type().size();
    ret += wei_elems * wei.type().size();
    ret += dst_elems * dst.type().size();
    return ret;
}

pvar_tile_t expand_tile(
        prop_kind_t prop, const prb_reqs_t &reqs, const pvar_tile_t &_tile) {
    pvar_tile_t tile = _tile;
    for (auto &d : conv_index_dims(prop)) {
        dim_t mod = reqs.max_factor(d);
        mod = math::lcm(mod, tile.get(d, 1));
        if (mod == 1) continue;
        tile[d] = mod;
    }
    return tile;
}

std::vector<problem_t> generate_problems(const bench_input_params_t &params) {
    if (params.nprbs == 0) return {};
    const double max_ops = 1e10;
    const double max_bytes = 100e6;
    auto tile = expand_tile(params.prop, params.reqs, params.tile);
    srand(static_cast<unsigned>(
            ir_utils::get_hash(params.reqs.str()) & 0xFFFFFFFFu));
    std::vector<problem_t> ret;
    const int max_iters = (1 << 24);
    for (int iter = 0; iter < max_iters; iter++) {
        auto shape = random_shape(params.prop, params.is_dw, tile);
        if (problem_t::ops(params.prop, shape) > max_ops) continue;
        if (footprint(params.src_tag, params.wei_tag, params.dst_tag, shape)
                > max_bytes)
            continue;
        auto prb = params.problem();
        prb.set_shape(shape);
        if (!params.reqs.fits(prb.shape())) continue;
        ret.push_back(prb);
        if ((int)ret.size() >= params.nprbs) break;
    }
    if ((int)ret.size() < params.nprbs) {
        std::cout << "Could not generate " << params.nprbs << " problems after "
                  << max_iters << " iterations" << std::endl;
        std::cout << params.reqs << std::endl;
        exit(1);
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

bench_data_t bench(const bench_manager_t &bench_mger,
        const kernel_desc_t &kernel_desc, std::vector<bench_task_t> &tasks,
        memory_pool_t *mem_pool_ptr = nullptr) {
    int ntasks = (int)tasks.size();

    auto eng = bench_mger.get_engine();
    auto strm = bench_mger.get_stream();
    std::cout << "Running benchmark for descriptor: " << kernel_desc.cmd_str()
              << std::endl;
    gpu_assert(!kernel_desc.spec.is_dynamic());
    auto kernel_desc_min_dims = kernel_desc;
    kernel_desc_min_dims.spec.mode = specialization_mode_t::min_dims;
    {
        auto guard = debug_t::make_kernel_desc_setter(kernel_desc_min_dims);
        if (!tasks[0].init_primitive(eng)) return {};
    }

    parallel_nd(ntasks, [&](dim_t i) {
        auto guard = debug_t::make_kernel_desc_setter(kernel_desc_min_dims);
        bool ok = tasks[i].init_primitive(eng);
        if (!ok) throw std::runtime_error("Initialization failed");
    });

    memory_pool_t _mem_pool;
    memory_pool_t &mem_pool = (mem_pool_ptr ? *mem_pool_ptr : _mem_pool);
    if (!mem_pool) {
        for (auto &t : tasks) {
            t.init_mem(mem_pool);
        }
        mem_pool.finalize(strm);
    }

    bench_data_t bd(0, kernel_desc);
    dnnl_reset_profiling(strm.get());
    for (int i = 0; i < ntasks; i++) {
        tasks[i].bench_async(strm, mem_pool);
    }
    bench_task_base_t::sync(strm, tasks);
    for (int i = 0; i < ntasks; i++) {
        bd.add(tasks[i].prb(), tasks[i].time());
    }
    std::cout << bd << std::endl;
    return bd;
}

class bench_runner_impl_t {
public:
    bench_runner_impl_t(const bench_manager_t &bench_mger,
            const bench_input_params_t &params)
        : bench_mger_(bench_mger) {
        auto prbs = generate_problems(params);
        for (auto &prb : prbs) {
            tasks_.emplace_back(prb);
        }
    }

    bench_data_t bench(const kernel_desc_t &_kernel_desc) {
        if (tasks_.empty()) return bench_data_t();
        auto kernel_desc = _kernel_desc;
        if (!create_conv_plan(kernel_desc, bench_mger_.hw())) return {};
        return planner::bench(bench_mger_, kernel_desc, tasks_, &mem_pool_);
    }

private:
    const bench_manager_t &bench_mger_;
    std::vector<bench_task_t> tasks_;
    memory_pool_t mem_pool_;
};

bench_runner_t::bench_runner_t(
        const bench_manager_t &bench_mger, const bench_input_params_t &params)
    : impl_(std::make_shared<bench_runner_impl_t>(bench_mger, params)) {}

bench_data_t bench_runner_t::bench(const kernel_desc_t &kernel_desc) {
    return impl_->bench(kernel_desc);
}

bench_data_t bench(const bench_manager_t &bench_mger,
        const kernel_desc_t &_kernel_desc, int nprbs) {
    auto kernel_desc = _kernel_desc;
    if (!create_conv_plan(kernel_desc, bench_mger.hw())) return {};
    bench_runner_t runner(bench_mger,
            bench_input_params_t(kernel_desc, bench_mger.hw(), nprbs));
    return runner.bench(kernel_desc);
}

bool try_create(
        const bench_manager_t &bench_mger, const kernel_desc_t &kernel_desc) {
    bench_input_params_t params(kernel_desc, bench_mger.hw(), /*nprbs=*/1);
    bench_task_t task(generate_problems(params)[0]);
    auto engine = bench_mger.get_engine();
    auto guard = debug_t::instance().make_kernel_desc_setter(kernel_desc);
    return task.init_primitive(engine);
}

layout_tag_t &get_out_tag(kernel_desc_t &kernel_desc) {
    switch (kernel_desc.prop) {
        case prop_kind::forward: return kernel_desc.dst_tag;
        case prop_kind::backward_data: return kernel_desc.src_tag;
        case prop_kind::backward_weights: return kernel_desc.wei_tag;
        default: gpu_error_not_expected();
    }
    return kernel_desc.dst_tag;
}

std::vector<type_t> get_out_types(const kernel_desc_t &kernel_desc) {
    std::vector<type_t> ret;
    switch (kernel_desc.prop) {
        case prop_kind::forward:
            ret.push_back(type_t::s8());
            ret.push_back(type_t::f16());
            ret.push_back(type_t::f32());
            break;
        case prop_kind::backward_data: break;
        case prop_kind::backward_weights:
            ret.push_back(type_t::f32());
            if (kernel_desc.wei_tag.type().is_bf16())
                ret.push_back(type_t::bf16());
        default: break;
    }
    return ret;
}

kernel_desc_t try_extensions(
        const bench_manager_t &bench_mger, const kernel_desc_t &kernel_desc) {
    auto &desc_out_type = kernel_desc.c_type();
    std::vector<prb_reqs_t> reqs_vec({kernel_desc.reqs()});
    std::vector<int> out_type_sizes({desc_out_type.size()});
    extensions_t ext;
    for (auto &out_type : get_out_types(kernel_desc)) {
        if (out_type.size() == desc_out_type.size()) continue;
        auto d = kernel_desc;
        auto &tag = get_out_tag(d);
        tag = layout_tag_t(tag.desc(), out_type, tag.raw_tag());
        if (!create_conv_plan(d, bench_mger.hw())) continue;
        if (!try_create(bench_mger, d)) continue;
        ext.add(extensions_t::out_size(out_type.size()));
        reqs_vec.push_back(d.reqs());
        out_type_sizes.push_back(out_type.size());
    }

    if (kernel_desc.prop == prop_kind::backward_weights
            && !kernel_desc.with_bias_bwd_w()) {
        auto d = kernel_desc;
        d.bias_type = type_t::f32();
        if (create_conv_plan(d, bench_mger.hw()) && try_create(bench_mger, d)) {
            ext.add(extension_kind_t::bias);
            reqs_vec.push_back(d.reqs());
            out_type_sizes.push_back(desc_out_type.size());
        }
    }

    // Try Stream-K.
    if (kernel_desc.prop != prop_kind::backward_data
            || (kernel_desc.a_type() == type_t::f32()
                    && kernel_desc.b_type() == type_t::f32())) {
        auto d = to_stream_k(kernel_desc, /*check_ext=*/false);
        if (!d.is_empty()) {
            if (create_conv_plan(d, bench_mger.hw())
                    && try_create(bench_mger, d)) {
                ext.add(extension_kind_t::stream_k);
            }
        }
    }

    auto _kernel_desc = kernel_desc;
    _kernel_desc.ext = ext;
    return _kernel_desc;
}

} // namespace planner
} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
