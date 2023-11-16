/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_IR_CONFIG_HPP
#define GPU_JIT_IR_CONFIG_HPP

#include <iostream>
#include <sstream>
#include <unordered_map>

#include "gpu/config.hpp"
#include "gpu/jit/ir/blocking.hpp"
#include "gpu/jit/ir/hw.hpp"
#include "gpu/jit/ir/post_ops.hpp"
#include "gpu/jit/ir/problem.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class layout_param_t : public dnnl::impl::gpu::param_t {
public:
    const layout_t &user() const { return user_; }
    const layout_t &compute() const { return compute_; }
    const layout_t &user_unnormalized() const { return user_unnormalized_; }
    const layout_t &compute_unnormalized() const {
        return compute_unnormalized_;
    }

    const std::string &user_unnormalized_tag() const {
        return user_unnormalized_tag_;
    }
    const std::string &compute_unnormalized_tag() const {
        return compute_unnormalized_tag_;
    }

    void set_from_str(const std::string &s) override {
        auto parts = gpu_utils::split(s, ".");
        switch ((int)parts.size()) {
            case 1:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[0];
                break;
            case 2:
                compute_unnormalized_tag_ = parts[0];
                user_unnormalized_tag_ = parts[1];
                break;
            default: ir_error_not_expected();
        }
    }

    void set_user(const layout_t &l) { user_ = l; }
    void set_compute(const layout_t &l) { compute_ = l; }
    void set_user_unnormalized(const layout_t &l, const std::string &tag) {
        user_unnormalized_ = l;
        user_unnormalized_tag_ = tag;
    }
    void set_compute_unnormalized(const layout_t &l, const std::string &tag) {
        compute_unnormalized_ = l;
        compute_unnormalized_tag_ = tag;
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=";
        oss << compute_unnormalized_tag_;
        if (user_unnormalized_tag_ != compute_unnormalized_tag_)
            oss << "." << user_unnormalized_tag_;
        return oss.str();
    }

private:
    layout_t user_;
    layout_t compute_;
    layout_t user_unnormalized_;
    layout_t compute_unnormalized_;
    std::string user_unnormalized_tag_;
    std::string compute_unnormalized_tag_;
};

class src_layout_param_t : public layout_param_t {
public:
    std::string name() const override { return "src"; }
    std::string desc() const override { return "Source layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class dst_layout_param_t : public layout_param_t {
    std::string name() const override { return "dst"; }
    std::string desc() const override { return "Destination layout."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class exec_cfg_param_t : public value_param_t<exec_config_t> {
public:
    using value_param_t::is_overridden;
    using value_param_t::value_param_t;

    std::string name() const override { return "exec-cfg"; }
    std::string desc() const override {
        return "Execution config (hardware config, number of registers, SIMD, "
               "etc).";
    }
    bool is_overridable() const override { return true; }
    bool is_default(const std::string &key) const override {
        if (key == "regs") return false;
        if (key == "simd") return false;
        if (key == "vec") return value_.vec_size() == value_.simd();
        ir_error_not_expected() << key;
        return false;
    }

    std::vector<std::string> accepted_keys() const override {
        std::vector<std::string> ret;
        ret.push_back("regs");
        ret.push_back("simd");
        ret.push_back("vec");
        return ret;
    }

    void set_from_str(
            const std::string &key, const std::string &value) override {
        if (key == "regs") {
            value_.set_regs(std::stoi(value));
        } else if (key == "simd") {
            value_.set_simd(std::stoi(value));
        } else if (key == "vec") {
            value_.set_vec_size(std::stoi(value));
        } else {
            ir_error_not_expected() << key;
        }
    }

    std::string str(const std::string &key) const override {
        std::ostringstream oss;
        if (key == "regs") {
            oss << "regs=" << value_.regs();
        } else if (key == "simd") {
            oss << "simd=" << value_.simd();
        } else if (key == "vec") {
            if (!is_default("vec")) oss << "vec=" << value_.vec_size();
        }
        return oss.str();
    }
};

class grid_param_t : public value_param_t<grid_info_t> {
public:
    using value_param_t::value_param_t;

    bool is_overridable() const override { return false; }
};

class kernel_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "kernel-grid"; }
    std::string desc() const override {
        return "Number of thread groups across dimensions (kernel grid).";
    }
    bool is_overridable() const override { return false; }
};

class thread_group_grid_param_t : public grid_param_t {
public:
    std::string name() const override { return "tg-grid"; }
    std::string desc() const override { return "Thread group grid."; }
    bool is_overridable() const override { return false; }
};

class tile_param_t : public param_t {
public:
    using value_t = prb_tile_t;

    const value_t &get() const { return tile_; }

    bool is_empty() const { return tile_.is_empty(); }

    int get(const prb_dim_t &dim) const { return tile_.get(dim, 1); }

    int operator()(const prb_dim_t &dim) const { return get(dim); }

    void set_from_str(const std::string &s) override {
        tile_ = prb_tile_t();
        for (auto &kv : ir_utils::to_string_int_map(s)) {
            tile_[prb_dim_t::from_name(kv.first)] = kv.second;
        }
    }

    void set(const prb_dim_t &dim, int size) { tile_[dim] = size; }

    void set(const value_t &value) { tile_ = value; }

    template <typename T>
    void set(const dim_map_t<T, int> &tile) {
        for (auto &d : tile) {
            set(d.str(), tile[d]);
        }
    }

    std::string str() const override {
        std::ostringstream oss;
        oss << short_name() << "=" << tile_.str();
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    value_t tile_;
};

class dims_param_t : public tile_param_t {
public:
    std::string name() const override { return "dims"; }
    std::string desc() const override { return "Problem dimensions."; }
    bool is_overridable() const override { return false; }
};

class iter_dims_param_t : public tile_param_t {
public:
    std::string name() const override { return "iter"; }
    std::string short_name() const override { return "i"; }
    std::string desc() const override {
        return "Iteration-level dimension blocks.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class loop_dims_param_t : public dims_param_t {
public:
    std::string name() const override { return "loop"; }
    std::string short_name() const override { return "l"; }
    std::string desc() const override { return "Loop-level dimension blocks."; }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class padded_dims_param_t : public tile_param_t {
public:
    std::string name() const override { return "pad"; }
    std::string desc() const override {
        return "Padded dimensions (rounded-up for blocks and to comply with "
               "required zero padding in output layouts) .";
    }
    bool is_overridable() const override { return false; }
};

class thread_group_dims_param_t : public tile_param_t {
public:
    std::string name() const override { return "tg"; }
    std::string short_name() const override { return "T"; }
    std::string desc() const override {
        return "Thread group-level dimension blocks.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return false; }
};

class unroll_param_t : public tile_param_t {
public:
    std::string name() const override { return "unroll"; }
    std::string short_name() const override { return "u"; }
    std::string desc() const override {
        return "Per-dimension unroll factors.";
    }
    bool is_overridable() const override { return true; }
    bool is_default() const override { return is_empty(); }
};

class prim_config_t : public container_config_t {
public:
    ~prim_config_t() override = default;
    std::string str() const override = 0;

    virtual prb_tile_t shape(bool pad) const = 0;
    virtual const std::vector<prb_dim_t> &index_dims() const = 0;
    virtual int pad_block(const prb_dim_t &d) const = 0;

    void set_zp_cfg(const zero_points_config_t &zp_cfg) { zp_cfg_ = zp_cfg; }
    const zero_points_config_t &zp_cfg() const { return zp_cfg_; }

    void set_params_id(int id) { params_id_ = id; }
    void set_bufs_hint(int bufs_hint) { bufs_hint_ = bufs_hint; }
    int bufs_hint() const { return bufs_hint_; }

    void maybe_override_from_env() {
#ifdef DNNL_DEV_MODE
        auto cfg_env = gpu_utils::dev_getenv("cfg", std::string());
        if (!cfg_env.empty()) override_set(cfg_env, /*is_env=*/true);
#endif
    }

    void set_params(const blocking_params_t &params) {
        ir_assert(!params.is_empty());
        const auto &blocking = params.blocking();
        if (!loop_dims().is_overridden()) loop_dims().set(blocking.loop());
        if (!thread_group_dims().is_overridden())
            thread_group_dims().set(blocking.thread_group());
        if (!iter_dims().is_overridden()) iter_dims().set(blocking.iter());

        // update padded dimensions based on what was set just above
        for (auto &d : index_dims()) {
            int blk = loop_dim(d) * thread_group_dim(d) * iter_dim(d);
            int padded = utils::rnd_up(dim(d), math::lcm(blk, pad_block(d)));
            padded_dims().set(d, padded);
        }

        set_params_id(params.id());
        set_bufs_hint(params.bufs_hint());
    }

    blocking_params_t params(
            int bufs_hint = blocking_params_t::bufs_hint_undef) const {
        blocking_t blocking;
        for (auto &d : index_dims()) {
            int loop = loop_dim(d);
            int tg = thread_group_dim(d);
            int iter = iter_dim(d);
            if (loop != 1) blocking.set_loop(d, loop);
            if (tg != 1) blocking.set_thread_group(d, tg);
            if (iter != 1) blocking.set_iter(d, iter);
        }
        blocking.set_simd(exec_cfg().vec_size());
        blocking_params_t ret(blocking, bufs_hint);
        ret.set_id(params_id_);
        return ret;
    }

    // Return thread utilization as a percentage. If this value is low,
    // parallelism is a fundamental limitation to the current work scheduling.
    static float get_thread_utilization(
            const exec_config_t &exec_cfg, int kg_elems, int tg_elems) {
        auto arch = convert_ngen_arch_to_dnnl(exec_cfg.hw().to_ngen());
        int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = exec_cfg.hw().eu_count() / eus_per_slice;

        int min_wg_per_slice_wave = std::max(eus_per_slice / tg_elems, 1);
        int min_wg_per_wave = slice_count * min_wg_per_slice_wave;
        return (100.f * kg_elems) / utils::rnd_up(kg_elems, min_wg_per_wave);
    }

    // Return wave utilization as a percentage. If this value is low, memory
    // latency may be an issue due to limited use of SMT to hide the latency.
    static float get_wave_utilization(
            const exec_config_t &exec_cfg, int kg_elems, int tg_elems) {
        auto arch = convert_ngen_arch_to_dnnl(exec_cfg.hw().to_ngen());
        int threads_per_eu = compute::device_info_t::threads_per_eu(
                arch, exec_cfg.regs() > 128);
        int eus_per_slice = compute::device_info_t::max_eus_per_wg(arch);
        int slice_count = exec_cfg.hw().eu_count() / eus_per_slice;

        int wgs_per_slice = eus_per_slice * threads_per_eu / tg_elems;
        ir_assert(wgs_per_slice > 0);
        int wgs_per_tile = slice_count * wgs_per_slice;
        return (100.f * kg_elems) / utils::rnd_up(kg_elems, wgs_per_tile);
    }

#define DECL_PARAM(name) \
    const name##_param_t &name##_param() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name##_param() { return name##_; } \
    const name##_param_t::value_t &name() const { \
        ir_assert(!name##_.is_undef()); \
        return name##_.get(); \
    } \
    void set_##name(const name##_param_t::value_t &value) { \
        name##_.set(value); \
    }
#define DECL_PARAM2(name) \
    const name##_param_t &name() const { \
        (void)name##_init_; \
        ir_assert(!name##_.is_undef()); \
        return name##_; \
    } \
    name##_param_t &name() { return name##_; }
    DECL_PARAM(exec_cfg)
    DECL_PARAM(kernel_grid)
    DECL_PARAM(thread_group_grid)
    DECL_PARAM2(src_layout)
    DECL_PARAM2(dst_layout)
    DECL_PARAM2(padded_dims)
    DECL_PARAM2(iter_dims)
    DECL_PARAM2(loop_dims)
    DECL_PARAM2(thread_group_dims)
#undef DECL_PARAM
#undef DECL_PARAM2

    int iter_dim(const prb_dim_t &d) const { return iter_dims().get(d); }

    int iter_dim(std::initializer_list<prb_dim_t> dims) const {
        int ret = 1;
        for (auto &dim : dims)
            ret *= iter_dim(dim);
        return ret;
    }

    int loop_dim(const prb_dim_t &d) const { return loop_dims().get(d); }

    int thread_group_dim(const prb_dim_t &d) const {
        return thread_group_dims().get(d);
    }

    int padded_dim(const prb_dim_t &d) const { return padded_dims().get(d); }

    int grid_dim(const prb_dim_t &dim) const {
        return ir_utils::safe_divide(padded_dim(dim),
                loop_dim(dim) * thread_group_dim(dim) * iter_dim(dim));
    }

    prb_tile_t dims() const { return shape(/* pad = */ false); }
    int dim(const prb_dim_t &d) const { return dims().get(d); }

    int sort_key(const param_t *param) const override;

    void init_kernel_grid(const std::array<prb_tile_t, 3> &grid) {
        std::vector<int> dims(grid.size(), 1);
        for (int i = 0; i < int(grid.size()); i++) {
            for (auto &d : grid[i]) {
                int tg_block = loop_dim(d) * thread_group_dim(d) * iter_dim(d);
                dims[i] *= ir_utils::safe_divide(padded_dim(d), tg_block);
            }
        }
        set_kernel_grid(grid_info_t(dims, "grid_idx"));
    }

    void init_thread_group_grid(const std::array<prb_tile_t, 3> &grid) {
        std::vector<int> dims(grid.size(), 1);
        for (int i = 0; i < int(grid.size()); i++) {
            for (auto &d : grid[i])
                dims[i] *= thread_group_dim(d);
        }
        set_thread_group_grid(grid_info_t(dims, "tg_idx"));
    }

protected:
    zero_points_config_t zp_cfg_;
    int params_id_ = -1;
    int bufs_hint_ = -1;

#define INIT_PARAM(name) \
    name##_param_t name##_; \
    param_init_t name##_init_ \
            = register_param([](const container_config_t *c) { \
                  return &static_cast<const prim_config_t *>(c)->name##_; \
              });
    INIT_PARAM(exec_cfg)
    INIT_PARAM(kernel_grid)
    INIT_PARAM(thread_group_grid)
    INIT_PARAM(src_layout)
    INIT_PARAM(dst_layout)
    INIT_PARAM(padded_dims)
    INIT_PARAM(iter_dims)
    INIT_PARAM(loop_dims)
    INIT_PARAM(thread_group_dims)
#undef INIT_PARAM
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
