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

#include "gpu/intel/jit/v2/conv/plan.hpp"

#include "gpu/intel/jit/v2/conv/tensor_utils.hpp"

#include <algorithm>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

pvar_coord_t<expr_t> coord_info_t::iter_coord() const {
    pvar_coord_t<expr_t> ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        ret[d] = simplify_rewrite(e.iter_size * e.iter_idx);
    }
    return ret;
}

pvar_coord_t<expr_t> coord_info_t::tg_iter_coord() const {
    pvar_coord_t<expr_t> ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        auto idx = e.iter_size * e.iter_idx;
        if (!is_const(e.thr_idx)) {
            idx = substitute(idx, e.thr_idx, expr_t(0));
        }
        ret[d] = simplify_rewrite(idx);
    }
    return ret;
}

pvar_tile_t coord_info_t::tg_iter_tile() const {
    pvar_tile_t ret;
    for (auto &d : entries_) {
        auto &e = entries_.at(d);
        ret[d] = e.tg_size * e.iter_size;
    }
    return ret;
}

class multiply_info_t {
public:
    multiply_info_t() = default;
    multiply_info_t(fma_kind_t fma, int simd, const pvar_tile_t &iter_tile,
            const pvar_map_t<char> &bmnk_map, const type_t &a_type,
            const layout_desc_t &a_desc, const type_t &b_type,
            const layout_desc_t &b_desc, const layout_desc_t &c_desc)
        : fma_(fma)
        , simd_(simd)
        , iter_tile_(iter_tile)
        , bmnk_map_(bmnk_map)
        , a_type_(a_type)
        , b_type_(b_type)
        , acc_type_(accumulator_type(a_type, b_type)) {
        if (!init(a_desc, b_desc, c_desc)) return;
        is_valid_ = true;
    }

    explicit operator bool() const { return is_valid_; }

    multiply_hint_t hint(tensor_kind_t abc) const {
        if (!utils::one_of(abc, tensor_kind_t::a, tensor_kind_t::b))
            return multiply_hint_t();
        bool src1 = (abc == tensor_kind_t::b);
        bool src2 = (abc == tensor_kind_t::a);
        multiply_hint_t hint;
        hint.fma = fma_;
        hint.simd = simd_;
        hint.src1 = src1;
        hint.src2 = src2;
        hint.bmnk_map = bmnk_map_;
        return hint;
    }

    fma_kind_t fma() const { return fma_; }
    int simd() const { return simd_; }
    const type_t &a_type() const { return a_type_; }
    const type_t &b_type() const { return b_type_; }
    const type_t &acc_type() const { return acc_type_; }

    bool has(tensor_kind_t abc, const pvar_t &dim) const {
        switch (abc) {
            case tensor_kind_t::a: return is_b(dim) || is_m(dim) || is_k(dim);
            case tensor_kind_t::b: return is_b(dim) || is_k(dim) || is_n(dim);
            case tensor_kind_t::c: return is_b(dim) || is_m(dim) || is_n(dim);
            default: gpu_error_not_expected();
        }
        return false;
    }

    bool is(const pvar_t &dim, char bmnk) const {
        gpu_assert(utils::one_of(bmnk, 'b', 'm', 'n', 'k'));
        if (!bmnk_map_.has(dim)) return false;
        return bmnk_map_[dim] == bmnk;
    }

    bool is_b(const pvar_t &dim) const { return is(dim, 'b'); }
    bool is_m(const pvar_t &dim) const { return is(dim, 'm'); }
    bool is_n(const pvar_t &dim) const { return is(dim, 'n'); }
    bool is_k(const pvar_t &dim) const { return is(dim, 'k'); }
    char to_bmnk(const pvar_t &dim) const { return bmnk_map_.at(dim); }

    pvar_tile_t inst_tile() const {
        pvar_tile_t ret;
        switch (fma_) {
            case fma_kind_t::mad: ret = b_inner_.int_dim_sizes(); break;
            case fma_kind_t::dpas: {
                auto a_tile = a_inner_.int_dim_sizes();
                auto b_tile = b_inner_.int_dim_sizes();
                ret = std::move(a_tile);
                for (auto &d : b_tile) {
                    if (ret.has(d)) gpu_assert(ret[d] == b_tile[d]);
                    ret[d] = b_tile[d];
                }
                return ret;
            }
            default: gpu_error_not_expected();
        }
        for (auto &d : iter_tile_) {
            if (!ret.has(d)) ret[d] = 1;
        }
        return ret;
    }

    bool is_compatible(tensor_kind_t abc, const layout_t &layout) const {
        if (!fma_type_supported(layout.type())) return false;
        switch (abc) {
            case tensor_kind_t::a: return layout.is_blocked_by(a_inner_);
            case tensor_kind_t::b: return layout.is_blocked_by(b_inner_);
            default: gpu_error_not_expected();
        }
        return false;
    }

    layout_t to_compatible_layout(
            tensor_kind_t abc, const layout_t &layout) const {
        auto ret = layout;
        switch (abc) {
            case tensor_kind_t::a: ret.block_by(a_inner_.blocks()); break;
            case tensor_kind_t::b: ret.block_by(b_inner_.blocks()); break;
            default: gpu_error_not_expected();
        }
        ret = get_fma_type_layout(ret);
        return ret;
    }

    layout_t acc_layout(const layout_t &a_layout, const layout_t &b_layout,
            const layout_t &c_layout) const {
        gpu_assert(a_layout.has_const_sizes());
        gpu_assert(b_layout.has_const_sizes());
        layout_t acc(c_layout.desc(), acc_type());
        for (auto &b : a_layout.blocks()) {
            if (is_k(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        for (auto &b : b_layout.blocks()) {
            if (is_k(b.dim) || is_b(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        acc.block_by(c_inner_.blocks());
        return acc;
    }

    layout_t bias_layout(
            const layout_t &b_layout, const layout_t &bias_layout) const {
        gpu_assert(b_layout.has_const_sizes());
        layout_t acc(bias_layout.desc(), acc_type());

        for (auto &b : b_layout.blocks()) {
            if (is_k(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        return acc;
    }

private:
    struct fused_dim_t {
        char mnk = '\0';
        std::vector<pvar_t> dims;
        std::vector<dim_t> sizes;

        fused_dim_t(char mnk) : mnk(mnk) {}
        int ndims() const { return (int)dims.size(); }
        dim_t size() const { return utils::array_product(sizes); }
        void add(const pvar_t &dim, dim_t size) {
            gpu_assert(size > 1);
            dims.push_back(dim);
            sizes.push_back(size);
        }

        std::pair<pvar_t, dim_t> pop(dim_t &block) {
            gpu_assert(!dims.empty());
            dim_t b = math::gcd(sizes.back(), block);
            gpu_assert(b > 1);
            auto ret = std::make_pair(dims.back(), b);
            sizes.back() /= b;
            block /= b;
            if (sizes.back() == 1) {
                dims.pop_back();
                sizes.pop_back();
            }
            return ret;
        }
    };

    bool fma_type_supported(const type_t &type) const {
        switch (fma_) {
            case fma_kind_t::mad:
                return utils::one_of(type, type_t::f32(), type_t::s16());
                break;
            case fma_kind_t::dpas:
                return utils::one_of(type, type_t::u8(), type_t::s8(),
                        type_t::f16(), type_t::bf16());
                break;
            default: gpu_error_not_expected();
        }
        return false;
    }

    layout_t get_fma_type_layout(const layout_t &layout) const {
        if (fma_ == fma_kind_t::mad) {
            auto blocks = layout.blocks();
            if (utils::one_of(layout.type(), type_t::s8(), type_t::u8())) {

                for (auto &b : blocks) {
                    b.stride *= 2;
                }
                return layout_t(
                        layout.desc(), type_t::s16(), layout.base(), blocks);
            }
            if (utils::one_of(layout.type(), type_t::f16(), type_t::bf16(),
                        type_t::f32()))
                return layout_t(
                        layout.desc(), type_t::f32(), layout.base(), blocks);
        }
        return layout;
    }

    bool init(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        switch (fma_) {
            case fma_kind_t::mad: return init_mad(a_desc, b_desc, c_desc);
            case fma_kind_t::dpas: return init_dpas(a_desc, b_desc, c_desc);
            default: gpu_error_not_expected();
        }
        return false;
    }

    bool init_mad(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        bool found = false;
        for (auto &d : iter_tile_) {
            if (iter_tile_[d] % simd_ != 0) continue;
            if (is_n(d) || is_b(d)) {
                found = true;
                block_t block;
                block.dim = d;
                block.size = simd_;
                block.stride = expr_t(1);
                b_inner_ = layout_t(b_desc, b_type_, 0, {std::move(block)});
                break;
            }
        }
        gpu_check(found) << "init_mad: cannot find dimension to vectorize.";
        c_inner_ = layout_t(c_desc, acc_type_, 0, b_inner_.blocks());
        return true;
    }

    bool init_dpas(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        fused_dim_t m_dim('m');
        fused_dim_t n_dim('n');
        fused_dim_t k_dim('k');
        for (auto &d : iter_tile_) {
            switch (to_bmnk(d)) {
                case 'm': m_dim.add(d, iter_tile_[d]); break;
                case 'n': n_dim.add(d, iter_tile_[d]); break;
                case 'k': k_dim.add(d, iter_tile_[d]); break;
                default: gpu_error_not_expected();
            }
        }
        gpu_check(m_dim.ndims() == 1 && n_dim.ndims() == 1
                && utils::one_of(k_dim.ndims(), 1, 2))
                << "init_dpas: cannot initialize MNK dimensions.";
        if (k_dim.ndims() == 2) {
            std::swap(k_dim.dims[0], k_dim.dims[1]);
            std::swap(k_dim.sizes[0], k_dim.sizes[1]);
        }
        const uint8_t sdepth = 8;
        const uint8_t rcount = 8;
        int type_size = a_type_.size();
        int dword_size = 4;
        gpu_check(m_dim.size() % rcount == 0)
                << "init_dpas: M dimension size is invalid: " << m_dim.size();
        gpu_check(n_dim.size() % simd_ == 0)
                << "init_dpas: N dimension size is invalid: " << n_dim.size();
        gpu_check((k_dim.size() * type_size) % (sdepth * dword_size) == 0)
                << "init_dpas: K dimension size is invalid: " << k_dim.size();

        auto _dpas = dpas_t::make(
                /*is_dpasw=*/false, simd_, sdepth, rcount, acc_type_, b_type_,
                a_type_);
        auto &dpas = _dpas.as<dpas_t>();
        a_inner_ = to_v2_layout(dpas.b_layout(), a_desc,
                std::vector<fused_dim_t> {k_dim, m_dim});
        b_inner_ = to_v2_layout(dpas.a_layout(), b_desc,
                std::vector<fused_dim_t> {n_dim, k_dim});
        c_inner_ = to_v2_layout(dpas.c_layout(), c_desc,
                std::vector<fused_dim_t> {n_dim, m_dim});
        return true;
    }

    static layout_t to_v2_layout(const jit::layout_t &layout,
            const layout_desc_t &desc, std::vector<fused_dim_t> dims) {
        layout_t ret(desc, layout.type());
        for (auto &b : layout.blocks()) {
            dim_t block = b.block;
            while (block > 1) {
                auto dim_block = dims[b.dim_idx].pop(block);
                ret.add_block(dim_block.first, dim_block.second);
            }
        }
        return ret;
    }

    bool is_valid_ = false;
    fma_kind_t fma_ = fma_kind_t::undef;
    int simd_ = 0;
    pvar_tile_t iter_tile_;
    pvar_map_t<char> bmnk_map_;
    type_t a_type_;
    type_t b_type_;
    type_t acc_type_;
    layout_t a_inner_;
    layout_t b_inner_;
    layout_t c_inner_;
};

class plan_builder_t {
public:
    plan_builder_t() = default;
    plan_builder_t(const kernel_desc_t &desc, const hw_t &hw)
        : desc_(desc), hw_(hw) {
        reqs_ = desc_.reqs();
    }

    const prb_reqs_t &reqs() const { return reqs_; }

    plan_t build() {
        init_dim_mapper_manager();
        init_tiles();
        if (!init_layouts()) return plan_t();
        if (!init_info()) return plan_t();
        return init_plan();
    }

private:
    static send_plan_t try_create_send_plan(const std::string &tag,
            const send_params_t &params, const view_t &view) {
        auto plan = create_send_plan(params, view, /*allow_fail=*/true);
        bool ok = [&]() {
            gpu_check(plan) << tag << ": cannot create send plan\n"
                            << params << "\n"
                            << ir_utils::add_tag("view", view.str());
            return true;
        }();
        if (!ok) return send_plan_t();
        return plan;
    }

    static bool check_compatible_layout(
            const layout_t &layout, const pvar_tile_t &tile) {
        for (auto &d : tile) {
            int inner = layout.inner_block(d, /*with_outer=*/false);
            gpu_check(tile[d] % inner == 0)
                    << "Incompatible layout and tiling. Layout: "
                    << layout.str() << ", tile: " << tile.str();
        }
        return true;
    }

    void init_dim_mapper_manager() {
        dim_mapper_manager_ = dim_mapper_manager_t(desc_.prop, reqs_);
    }

    void init_tiles() {
        tg_grid_ = create_thread_group_grid(desc_);
        thr_grid_ = create_thread_grid(desc_);
        for (auto &d : conv_index_dims(desc_.prop)) {
            bool is_loop = desc_.loop_desc.has(d);
            bool is_global_loop = desc_.loop_desc.is_global(d);
            dim_t tg_tile = desc_.thread_group_tile.get(d, 1);
            dim_t iter_tile = desc_.iter_tile.get(d, 1);
            auto thr_idx = thr_grid_.index_var(d);
            coord_info_.add_dim(d, is_loop, is_global_loop, tg_tile, thr_idx,
                    iter_tile, reqs_);
        }
    }

    bool init_layouts() {
        auto src_layout = make_conv_layout(
                tensor_kind_t::src, desc_.src_tag, desc_.is_dw, reqs_);
        auto wei_layout = make_conv_layout(
                tensor_kind_t::wei, desc_.wei_tag, desc_.is_dw, reqs_);
        auto dst_layout = make_conv_layout(
                tensor_kind_t::dst, desc_.dst_tag, desc_.is_dw, reqs_);
        gpu_check(check_compatible_layout(src_layout, desc_.iter_tile));
        gpu_check(check_compatible_layout(wei_layout, desc_.iter_tile));
        gpu_check(check_compatible_layout(dst_layout, desc_.iter_tile));
        a_layout_ = pick_a(desc_.prop, src_layout, wei_layout, dst_layout);
        b_layout_ = pick_b(desc_.prop, src_layout, wei_layout, dst_layout);
        c_layout_ = pick_c(desc_.prop, src_layout, wei_layout, dst_layout);
        if (desc_.with_bias_bwd_w()) {
            auto bias_tag = make_conv_layout_tag(
                    tensor_kind_t::bias, "a:" + desc_.bias_type.str());
            bias_layout_ = make_conv_layout(
                    tensor_kind_t::bias, bias_tag, desc_.is_dw, reqs_);
        }
        return true;
    }

    pvar_map_t<char> to_bmnk_map() const {
        pvar_map_t<char> ret;
        for (auto &d : conv_index_dims(desc_.prop)) {
            auto gemm_d = to_gemm(d, desc_.prop);
            gpu_assert(!gemm_d.is_undef());
            ret[d] = gemm_d.name()[0];
        }
        return ret;
    }

    bool init_info() {
        auto &a_mapper = dim_mapper_manager_.mapper(tensor_kind_t::a);
        auto &b_mapper = dim_mapper_manager_.mapper(tensor_kind_t::b);
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto &a_desc = a_mapper.layout_desc();
        auto &b_desc = b_mapper.layout_desc();
        auto &c_desc = c_mapper.layout_desc();
        mul_info_ = multiply_info_t(desc_.fma, desc_.simd, desc_.iter_tile,
                to_bmnk_map(), a_layout_.type(), a_desc, b_layout_.type(),
                b_desc, c_desc);
        if (!mul_info_) return false;
        return true;
    }

    plan_t init_plan() {
        plan_t plan(hw_);
        if (!try_init_plan(plan, reqs_) || !check_plan(plan)) return plan_t();

        // Re-create plan to ensure all collected requirements are cross-used
        // between sub-plans.
        plan = plan_t(hw_);
        if (!try_init_plan(plan, reqs_) || !check_plan(plan)) {
            gpu_error_not_expected();
            return plan_t();
        }
        reqs_.simplify();
        return plan;
    }

    bool try_init_plan(plan_t &plan, prb_reqs_t &reqs) const {
        plan.desc = desc_;
        plan.tg_grid = tg_grid_;
        plan.thr_grid = thr_grid_;
        plan.virt_grid = virt_grid_;
        plan.coord_info = coord_info_;
        gpu_check(init_x2r_fma_plan(plan.x2r_fma, reqs));
        gpu_check(init_prefetch_plan(
                plan.x2r_fma, plan.virt_grid, plan.prefetch));
        gpu_check(init_epilogue_plan(
                plan.x2r_fma.c_layout, plan.virt_grid, plan.epilogue, reqs));
        if (desc_.with_bias_bwd_w())
            gpu_check(init_epilogue_bias(
                    plan.x2r_fma.bias_layout, plan.epilogue, reqs));
        return true;
    }

    bool init_x_prefetch_plan(tensor_kind_t abc,
            const pvar_coord_t<expr_t> &coord, const pvar_tile_t &tile,
            const x2r_fma_plan_t &x2r_fma, virt_grid_t &virt_grid,
            send_plan_t &prefetch) const {
        auto &mapper = dim_mapper_manager_.mapper(abc);
        auto &layout = (abc == tensor_kind_t::a ? a_layout_ : b_layout_);
        grid_splitter_t grid_splitter;
        for (auto &d : thr_grid_.all_dims()) {
            grid_splitter.add(
                    thr_grid_.index_var(d), desc_.thread_group_tile[d]);
        }
        auto view = view_t::split(mapper, layout, coord, tile, grid_splitter);
        for (auto &kv : grid_splitter.virt_grid_idxs()) {
            virt_grid.add(kv.first, kv.second);
        }
        // Try 2D messages first.
        auto params = get_send_params(
                abc, send_op_t::prefetch, view, send_kind_t::_2d);
        prefetch = create_send_plan(params, view, /*allow_fail=*/true);
        if (!prefetch || !reqs_.implies(prefetch.reqs())) {
            // If 2D failed, try compressed prefetch.
            params = get_send_params(abc, send_op_t::prefetch, view,
                    send_kind_t::compressed_prefetch);
            prefetch = try_create_send_plan(__func__, params, view);
            if (!prefetch) return false;
            if (!reqs_.implies(prefetch.reqs())) return false;
        }
        return true;
    }

    bool init_prefetch_plan(const x2r_fma_plan_t &x2r_fma,
            virt_grid_t &virt_grid, prefetch_plan_t &plan) const {
        if (desc_.prefetch.a) {
            gpu_check(init_x_prefetch_plan(tensor_kind_t::a,
                    coord_info_.tg_iter_coord(), coord_info_.tg_iter_tile(),
                    x2r_fma, virt_grid, plan.a_prefetch));
        }
        if (desc_.prefetch.b) {
            gpu_check(init_x_prefetch_plan(tensor_kind_t::b,
                    coord_info_.tg_iter_coord(), coord_info_.tg_iter_tile(),
                    x2r_fma, virt_grid, plan.b_prefetch));
        }
        return true;
    }

    bool init_x2r_plan(
            tensor_kind_t abc, const view_t &view, x2r_plan_t &plan) const {
        auto params = get_send_params(abc, send_op_t::load, view);
        auto load = try_create_send_plan(__func__, params, view);
        if (!load) return false;
        reorder_plan_t reorder;
        layout_t reg_layout;
        if (mul_info_.is_compatible(abc, load.reg_layout())) {
            reg_layout = load.reg_layout();
        } else {
            auto &src = load.reg_layout();
            auto dst = mul_info_.to_compatible_layout(abc, load.reg_layout());
            reorder = reorder_plan_t(hw_, src, dst);
            reg_layout = reorder.dst;
        }
        plan = x2r_plan_t(hw_);
        plan.tensor_kind = abc;
        plan.load = std::move(load);
        plan.reorder = std::move(reorder);
        plan.layout = std::move(reg_layout);
        if (desc_.with_bias_bwd_w() && abc == tensor_kind_t::b) {
            auto bias_layout = mul_info_.bias_layout(plan.layout, bias_layout_);
            plan.bias_layout = std::move(bias_layout);
        }
        return true;
    }

    bool init_fma_plan(
            const layout_t &a, const layout_t &b, fma_plan_t &plan) const {
        auto inst_tile = mul_info_.inst_tile();
        auto acc_layout = mul_info_.acc_layout(a, b, c_layout_);
        gpu_check(!acc_layout.is_empty()) << "init_fma_plan: cannot vectorize.";
        plan = fma_plan_t(hw_);
        plan.simd = desc_.simd;
        plan.fma = desc_.fma;
        plan.a_layout = a;
        plan.b_layout = b;
        plan.c_layout = std::move(acc_layout);
        plan.inst_tile = std::move(inst_tile);
        return true;
    }

    bool init_x2r_fma_plan(x2r_fma_plan_t &plan, prb_reqs_t &reqs) const {
        auto &outer = desc_.iter_outer_tile;
        auto &tile = desc_.iter_tile;
        gpu_assert(outer.is_empty() || outer.size() == 1);
        auto outer_dim = (outer.is_empty() ? pvar_t() : *outer.begin());
        dim_t outer_size = outer.get(outer_dim, 1);
        auto sub_tile = tile;
        if (!outer_dim.is_undef()) sub_tile[outer_dim] /= outer_size;
        bool is_outer_m = mul_info_.is_m(outer_dim);
        layout_t a_prev_layout;
        layout_t b_prev_layout;
        layout_t c_prev_layout;
        layout_t bias_prev_layout;
        int c_off_elems = 0;
        int bias_off_elems = 0;
        auto &a_mapper = dim_mapper_manager_.mapper(tensor_kind_t::a);
        auto &b_mapper = dim_mapper_manager_.mapper(tensor_kind_t::b);
        for (int i = 0; i < outer_size; i++) {
            auto sub_coord = coord_info_.iter_coord();
            if (!outer_dim.is_undef()) {
                sub_coord[outer_dim] += sub_tile[outer_dim] * i;
            }
            if (is_outer_m || i == 0) {
                auto a_sub_view
                        = view_t(a_mapper, a_layout_, sub_coord, sub_tile);
                x2r_plan_t a;
                gpu_check(init_x2r_plan(tensor_kind_t::a, a_sub_view, a));
                plan.add_stage(a);
                a_prev_layout = a.layout;
            }
            if (!is_outer_m || i == 0) {
                auto b_sub_view
                        = view_t(b_mapper, b_layout_, sub_coord, sub_tile);
                x2r_plan_t b;
                gpu_check(init_x2r_plan(tensor_kind_t::b, b_sub_view, b));
                b_prev_layout = b.layout;
                if (desc_.with_bias_bwd_w()) {
                    bias_prev_layout = b.bias_layout;
                    b.bias_layout.set_base(bias_off_elems);
                    bias_off_elems += ir_utils::safe_div(
                            b.bias_layout.size(), b.bias_layout.type().size());
                }
                plan.add_stage(b);
            }

            fma_plan_t fma;
            gpu_check(init_fma_plan(a_prev_layout, b_prev_layout, fma));
            gpu_check(c_prev_layout.is_empty() || fma.c_layout == c_prev_layout)
                    << "init_x2r_fma_plan: inconsistent C layout from "
                       "subtiles.";
            c_prev_layout = fma.c_layout;
            fma.c_layout.set_base(c_off_elems);
            c_off_elems += ir_utils::safe_div(
                    fma.c_layout.size(), fma.c_layout.type().size());
            plan.add_stage(fma);
        }
        plan.c_layout = c_prev_layout;
        if (desc_.with_bias_bwd_w()) plan.bias_layout = bias_prev_layout;

        if (!outer_dim.is_undef()) {
            int stride = ir_utils::safe_div(
                    c_prev_layout.size(), c_prev_layout.type().size());
            plan.c_layout.add_block(outer_dim, outer_size, stride);
            if (desc_.with_bias_bwd_w()) {
                auto &bias_mapper
                        = dim_mapper_manager_.mapper(tensor_kind_t::bias);
                if (bias_mapper.has(outer_dim)) {
                    int bias_stride
                            = ir_utils::safe_div(bias_prev_layout.size(),
                                    bias_prev_layout.type().size());
                    plan.bias_layout.add_block(
                            outer_dim, outer_size, bias_stride);
                }
            }
        }
        reqs.add(plan.reqs());
        return true;
    }

    bool init_epilogue_store_bias(bool is_atomic,
            const layout_t &bias_reg_layout, const view_t &bias_mem_view,
            epilogue_store_plan_t &plan, prb_reqs_t &reqs) const {
        auto params = get_send_params(tensor_kind_t::undef,
                is_atomic ? send_op_t::atomic_add : send_op_t::store,
                bias_mem_view);
        auto store = try_create_send_plan(__func__, params, bias_mem_view);
        if (!store) return false;
        gpu_check(reqs.implies(store.reqs()))
                << "Bias store add needs additional requirements.";
        plan.bias_store = store;
        if (bias_reg_layout != store.reg_layout()) {
            auto store_layout = store.reg_layout();
            if (bias_reg_layout != store_layout) {
                plan.bias_reorder = reorder_plan_t(hw_);
                plan.bias_reorder.src = std::move(bias_reg_layout);
                plan.bias_reorder.dst = std::move(store_layout);
            }
        }
        return true;
    }

    bool init_epilogue_bias(const layout_t &bias_reg_layout,
            epilogue_plan_t &plan, prb_reqs_t &reqs) const {
        auto &bias_mapper = dim_mapper_manager_.mapper(tensor_kind_t::bias);
        auto bias_mem_view = view_t(
                dim_mapper_manager_.mapper(tensor_kind_t::bias), bias_layout_,
                coord_info_.iter_coord(), desc_.iter_tile);
        auto reduce_cond = expr_t(true);
        for (int i = 0; i < c_layout_.desc().ndims(); i++) {
            auto dim = c_layout_.desc().prb_dim(i);
            if (!bias_mapper.has(dim))
                reduce_cond
                        = reduce_cond & (coord_info_.iter_coord()[dim] == 0);
        }
        plan.bias_reduce_cond = std::move(reduce_cond);
        plan.bias_layout = bias_reg_layout;
        gpu_check(init_epilogue_store_bias(/*is_atomic=*/desc_.use_stream_k,
                bias_reg_layout, bias_mem_view, plan.store, reqs));
        return true;
    }

    bool init_slm_reduce_plan(const layout_t &c_layout, virt_grid_t &virt_grid,
            slm_reduce_plan_t &plan) const {
        pvar_t k_dim;
        for (auto &d : desc_.thread_group_tile) {
            if (to_gemm(d, desc_.prop) == pvars::k) {
                k_dim = d;
                break;
            }
        }
        if (k_dim.is_undef()) return true;

        dim_t k_tg = desc_.thread_group_tile.at(k_dim);
        gpu_assert(k_tg > 1);
        gpu_assert(desc_.thread_group_tile.elems() == k_tg)
                << "Local k-slicing assumes no split by M/N.";
        gpu_check(c_layout.size() % hw_.grf_size() == 0)
                << "init_slm_reduce_plan: c_layout is not aligned to a "
                   "reigster boundary.";

        // Create SLM layout to store partial reductions.
        auto mapper = extend_mapper(
                dim_mapper_manager_.mapper(tensor_kind_t::c), k_dim, 'k');
        layout_t slm_layout(mapper.layout_desc(), c_layout.type(),
                c_layout.base(), c_layout.blocks());
        slm_layout.add_block(k_dim, k_tg);
        auto c_tile = c_layout.desc().filter_dim_map(desc_.iter_tile);

        pvar_coord_t<expr_t> store_coord;
        store_coord[k_dim] = thr_grid_.index_var(k_dim);
        pvar_tile_t store_tile = c_tile;
        store_tile.unset(k_dim);

        // Store partial reductions.
        auto store_view = view_t(mapper, slm_layout, store_coord, store_tile);
        auto store_params = get_send_params(tensor_kind_t::c, send_op_t::store,
                store_view, send_kind_t::block, send_address_t::slm);
        store_params.skip_mask.push_back(k_dim);
        auto store = try_create_send_plan(__func__, store_params, store_view);
        if (!store) return false;

        // Split the original tile evenly between k_tg threads.
        grid_splitter_t grid_splitter;
        grid_splitter.add(thr_grid_.index_var(k_dim), k_tg);
        auto split_view = view_t::split(mapper, c_layout,
                pvar_coord_t<expr_t>(), c_tile, grid_splitter);
        for (auto &kv : grid_splitter.virt_grid_idxs()) {
            virt_grid.add(kv.first, kv.second);
        }

        auto &load_coord = split_view.coord();
        auto tile_with_k = split_view.tile();
        tile_with_k[k_dim] = k_tg;

        // Load partial sums and do the final reduction.
        auto load_view = view_t(mapper, slm_layout, load_coord, tile_with_k,
                grid_splitter.var_range_info());
        auto load_params = get_send_params(tensor_kind_t::c, send_op_t::load,
                load_view, send_kind_t::block, send_address_t::slm);
        load_params.skip_mask.push_back(k_dim);
        auto load = try_create_send_plan(__func__, load_params, load_view);
        if (!load) return false;

        auto &load_layout = load.reg_layout();
        auto reduced_layout = load_layout.map(split_view.tile());
        auto reduce = reduce_plan_t(hw_, load_layout, reduced_layout);
        auto c_post_layout = std::move(reduced_layout);
        c_post_layout.remove(k_dim);

        plan = slm_reduce_plan_t(hw_);
        plan.store = std::move(store);
        plan.load = std::move(load);
        plan.reduce = std::move(reduce);
        plan.c_layout = std::move(c_post_layout);
        plan.c_coord = coord_info_.iter_coord() + load_coord;

        return true;
    }

    bool init_epilogue_store_plan(bool is_atomic, const layout_t &c_reg_layout,
            const view_t &c_mem_view, epilogue_store_plan_t &plan,
            prb_reqs_t &reqs) const {
        auto params = get_send_params(tensor_kind_t::c,
                is_atomic ? send_op_t::atomic_add : send_op_t::store,
                c_mem_view);
        // TODO: Implement fallback from 2D to block/scattered messages to
        // allow partial use of 2D messages when possible.
        auto store = try_create_send_plan(__func__, params, c_mem_view);
        if (!store) return false;
        auto &tile = store.entry_tile();
        plan.tile = tile;
        plan.c_store = store;
        auto c_reg_tile_layout = c_reg_layout.map(tile);
        auto store_layout = store.reg_layout().map(tile);
        if (c_reg_tile_layout != store_layout) {
            plan.reorder = reorder_plan_t(hw_);
            plan.reorder.src = std::move(c_reg_tile_layout);
            plan.reorder.dst = std::move(store_layout);
        }
        reqs.add(plan.c_store.reqs());
        return true;
    }

    bool init_epilogue_plan(const layout_t &c_fma_layout,
            virt_grid_t &virt_grid, epilogue_plan_t &plan,
            prb_reqs_t &reqs) const {
        gpu_check(
                init_slm_reduce_plan(c_fma_layout, virt_grid, plan.slm_reduce));
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto c_reg_layout
                = (plan.slm_reduce ? plan.slm_reduce.c_layout : c_fma_layout);
        auto c_coord = (plan.slm_reduce ? plan.slm_reduce.c_coord
                                        : coord_info_.iter_coord());
        auto c_tile = c_reg_layout.int_dim_sizes();
        auto c_mem_view = view_t(c_mapper, c_layout_, c_coord, c_tile);
        plan.c_reg_layout = c_reg_layout;
        plan.c_coord = c_mem_view.coord();
        gpu_check(init_epilogue_store_plan(/*is_atomic=*/desc_.use_stream_k,
                c_reg_layout, c_mem_view, plan.store, reqs));
        return true;
    }

    bool check_plan(const plan_t &plan) const {
        int grf_bound = hw_.grf_size() * desc_.regs;
        int grf_bytes = plan.grf_usage_bytes();
        gpu_check(grf_bytes <= grf_bound)
                << "Plan:\n"
                << plan.str() << "\ncheck_plan: out of registers";
        int slm_bound = compute::device_info_t::max_slm_size_per_tg(
                convert_ngen_arch_to_dnnl(hw_.to_ngen()),
                into<int>(desc_.thread_group_tile.elems()), desc_.regs > 128);
        int slm_bytes = plan.slm_usage_bytes();
        gpu_check(slm_bytes <= slm_bound)
                << "Plan:\n"
                << plan.str() << "\ncheck_plan: out of SLM";
        return true;
    }

    send_params_t get_send_params(tensor_kind_t abc, send_op_t op,
            const view_t &view, send_kind_t send_kind = send_kind_t::undef,
            send_address_t send_address = send_address_t::a64) const {
        if (op == send_op_t::atomic_add) {
            auto &type = view.type();
            gpu_assert(type.is_f32() || type.is_s32());
            if (type.is_f32()) op = send_op_t::atomic_fadd;
        }
        send_params_t params;
        params.hw = hw_;
        params.kind = (send_kind != send_kind_t::undef
                        ? send_kind
                        : desc_.access_kind(op, abc));
        params.address = send_address;
        params.op = op;
        if (params.kind == send_kind_t::_2d)
            params.hint_2d = send_2d_hint_t(view, op, mul_info_.hint(abc));
        params.skip_mask = skip_mask(view);
        params.init_max_entry_reg_size();
        params.external_reqs = &reqs_;
        return params;
    }

    std::vector<pvar_t> skip_mask(const view_t &view) const {
        return conv::skip_mask(view, coord_info_.tg_iter_tile(), reqs_);
    }

    kernel_desc_t desc_;
    hw_t hw_;

    dim_mapper_manager_t dim_mapper_manager_;
    multiply_info_t mul_info_;
    coord_info_t coord_info_;
    grid_t tg_grid_;
    grid_t thr_grid_;
    virt_grid_t virt_grid_;
    layout_t a_layout_;
    layout_t b_layout_;
    layout_t c_layout_;
    layout_t bias_layout_;
    prb_reqs_t reqs_;
};

plan_t create_conv_plan_impl(const kernel_desc_t &desc, const hw_t &hw,
        const problem_t *prb = nullptr) {
    if (!desc.is_supported(hw, prb)) return plan_t();
    plan_builder_t builder(desc, hw);
    auto plan = builder.build();
    if (plan) {
#ifdef DNNL_DEV_MODE
        auto &plan_reqs = builder.reqs();
        auto desc_reqs = desc.reqs();
        desc_reqs.simplify();
        gpu_assert(plan_reqs.str() == desc_reqs.str())
                << "Mismatch between plan and descriptor dimension "
                   "requirements:\n== Plan:\n"
                << plan_reqs.str() << "\n== Descriptor:\n"
                << desc_reqs.str();
#endif
    }
    return plan;
}

plan_t create_conv_plan(const kernel_desc_t &desc, const hw_t &hw) {
    return create_conv_plan_impl(desc, hw);
}

plan_t create_conv_plan(const kernel_desc_t &desc, const problem_t &prb) {
    return create_conv_plan_impl(desc, prb.hw(), &prb);
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
