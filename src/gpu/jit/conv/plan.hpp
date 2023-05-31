/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_PLAN_HPP
#define GPU_JIT_CONV_PLAN_HPP

#include <sstream>
#include <string>

#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/conv/plan_utils.hpp"
#include "gpu/jit/conv/zp_plan.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/send_plan.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct reorder_plan_t : public base_plan_t {
    layout_t src;
    layout_t dst;
    int split_factor = 1;

    using base_plan_t::base_plan_t;

    explicit operator bool() const { return !src.is_empty(); }

    bool can_split(int factor) const;
    void set_split(int factor = 1);
    stmt_t create_stmt(const expr_t &src_buf, const expr_t &dst_buf) const;
    int src_buf_size() const;
    int estimate_regs() const;

    std::string str(const std::string &tag = "reorder") const {
        std::ostringstream oss;
        oss << tag << ": src:" << src << " -> dst:" << dst;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct reduce_plan_t : public base_plan_t {
    layout_t src;
    layout_t dst;
    uint32_t mask;
    int split_factor = 1;

    using base_plan_t::base_plan_t;

    explicit operator bool() const { return !src.is_empty(); }
    int dst_buf_size() const;
    bool can_split(int factor) const;
    void set_split(int factor = 1);
    stmt_t create_stmt(const expr_t &src_buf, const expr_t &dst_buf) const;
    int estimate_regs() const;

    std::string str(const std::string &tag = "reduce") const {
        std::ostringstream oss;
        oss << tag << ": src:" << src << " -> dst:" << dst;
        return oss.str();
    }

    IR_DEFINE_DUMP()
};

struct slm_plan_t : public base_plan_t {
    layout_t a_layout;
    layout_t b_layout;
    send_plan_t a_g2s_load;
    send_plan_t b_g2s_load;
    tensor_t x_reduce_tile;
    layout_t x_reduce_layout;
    reduce_plan_t x_reduce;
    reorder_plan_t a_reorder;
    reorder_plan_t b_reorder;
    send_plan_t a_g2s_store;
    send_plan_t b_g2s_store;
    grid_info_t a_grid;
    grid_info_t b_grid;

    slm_plan_t(ngen::HW hw)
        : base_plan_t(hw), x_reduce(hw), a_reorder(hw), b_reorder(hw) {}

    explicit operator bool() const { return has_a() || has_b(); }
    bool has_a() const { return (bool)a_g2s_load; }
    bool has_b() const { return (bool)b_g2s_load; }
    int slm_size() const { return (int)(a_layout.size() + b_layout.size()); }
    std::string str() const;

    IR_DEFINE_DUMP()
};

struct prefetch_plan_t : public base_plan_t {
    send_plan_t a_prefetch;
    send_plan_t b_prefetch;
    grid_info_t a_grid;
    grid_info_t b_grid;

    using base_plan_t::base_plan_t;

    explicit operator bool() const { return a_prefetch || b_prefetch; }

    bool has_a() const { return (bool)a_prefetch; }
    bool has_b() const { return (bool)b_prefetch; }

    int estimate_regs(bool reuse_headers) const;
    std::string str() const;

    IR_DEFINE_DUMP()
};

struct x2r_plan_t : public base_plan_t {
    send_plan_t a_load;
    send_plan_t b_load;
    tensor_t x_reduce_tile;
    layout_t x_reduce_layout;
    reduce_plan_t x_reduce;
    reorder_plan_t a_reorder;
    reorder_plan_t b_reorder;
    layout_t a_layout;
    layout_t b_layout;
    abc_kind_t split_abc = abc_kind_t::undef;
    int split_factor = 1;

    x2r_plan_t(ngen::HW hw)
        : base_plan_t(hw), x_reduce(hw), a_reorder(hw), b_reorder(hw) {}

    bool can_split(abc_kind_t abc, int factor) const;
    void set_split(abc_kind_t abc = abc_kind_t::undef, int factor = 1);

    int a_buf_size() const {
        int a_size = a_layout.size();
        if (split_abc == abc_kind_t::a)
            a_size = utils::div_up(a_size, split_factor);
        return a_size;
    }

    int b_buf_size() const {
        int b_size = b_layout.size();
        if (split_abc == abc_kind_t::b)
            b_size = utils::div_up(b_size, split_factor);
        return b_size;
    }

    int estimate_regs(bool reuse_headers) const;
    std::string str() const;

    IR_DEFINE_DUMP()
};

struct fma_plan_t : public base_plan_t {
    layout_t a_layout;
    layout_t b_layout;
    layout_t c_layout;
    layout_t c_prb_layout;
    fma_kind_t fma_kind;
    int b_blk;
    int m_blk;
    int n_blk;
    int k_blk;
    abc_kind_t split_abc = abc_kind_t::undef;
    int split_factor = 1;

    using base_plan_t::base_plan_t;

    int max_bmn_blk() const {
        int ret = 0;
        ret = std::max(ret, b_blk);
        ret = std::max(ret, m_blk);
        ret = std::max(ret, n_blk);
        return ret;
    }

    bool can_split(abc_kind_t abc, int factor) const;
    void set_split(abc_kind_t abc, int factor);
    bool is_a_broadcast() const { return b_blk * m_blk * k_blk == 1; }
    bool is_b_broadcast() const { return b_blk * k_blk * n_blk == 1; }
    int a_buf_size() const;
    int b_buf_size() const;
    int bmnk_split_idx(bmnk_kind_t bmnk, int split_off, bool is_start) const;
    int bmnk_start_idx(bmnk_kind_t bmnk, int subtile_idx) const;
    int bmnk_stop_idx(bmnk_kind_t bmnk, int subtile_idx) const;

    int estimate_regs() const;
    std::string str() const;

    IR_DEFINE_DUMP()
};

struct conv_plan_t : public base_plan_t {
    expr_t ap_buf;
    expr_t bp_buf;
    expr_t cp_buf;
    constraint_set_t init_cset;
    gemm_schedule_t gemm_schedule;
    view_t bia_view;
    slm_plan_t slm;
    prefetch_plan_t prefetch;
    x2r_plan_t x2r;
    fma_plan_t fma;
    zp_plan_t zp;
    abc_kind_t split_abc = abc_kind_t::undef;
    int split_factor = 1;
    bool reuse_headers = false;
    int max_gmem_bufs = 0;
    int reserved_regs = -1;

    conv_plan_t(ngen::HW hw)
        : base_plan_t(hw), slm(hw), prefetch(hw), x2r(hw), fma(hw), zp(hw) {}

    const tensor_t &x_reduce_tile() const {
        if (!x2r.x_reduce_tile.is_empty()) return x2r.x_reduce_tile;
        if (!slm.x_reduce_tile.is_empty()) return slm.x_reduce_tile;
        ir_error_not_expected();
        return x2r.x_reduce_tile;
    }

    bool can_split(abc_kind_t abc, int factor) const;
    void set_split(abc_kind_t abc, int factor);
    bool uses_2d_load(abc_kind_t abc) const;
    grf_usage_t grf_usage() const;
    void reset();
    std::string str() const;

    IR_DEFINE_DUMP()
};

class conv_config_t;

status_t init_plan(conv_config_t &cfg);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
