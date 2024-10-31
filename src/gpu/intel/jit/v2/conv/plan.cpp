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

#include "gpu/intel/jit/v2/conv/plan.hpp"

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

layout_tag_t append_groups(
        tensor_kind_t tensor_kind, const layout_tag_t &layout_tag, bool is_dw) {
    bool is_src = (tensor_kind == tensor_kind_t::src);
    bool is_dst = (tensor_kind == tensor_kind_t::dst);
    bool is_bia = (tensor_kind == tensor_kind_t::bia);
    if (!is_src && !is_dst && !is_bia) return layout_tag;
    auto xc_dim = (is_src ? pvars::ic : pvars::oc);
    auto xc_letter = dim_idx::as_tag(layout_tag.desc().dim_index(xc_dim));
    auto new_g_letter = xc_letter;
    auto new_xc_letter = into<char>(xc_letter + 1);
    auto &raw_tag = layout_tag.raw_tag();
    auto &entries = raw_tag.entries();
    layout_raw_tag_t new_raw_tag;
    for (auto &e : entries) {
        if (e.letter == xc_letter) {
            if (is_dw) {
                new_raw_tag.add_entry(new_g_letter, e.block, e.is_blocked);
                new_raw_tag.add_entry(new_xc_letter, 1, false);
            } else if (e.is_outer()) {
                new_raw_tag.add_entry(new_g_letter, 0, false);
                new_raw_tag.add_entry(new_xc_letter, e.block, e.is_blocked);
            } else {
                new_raw_tag.add_entry(new_xc_letter, e.block, e.is_blocked);
            }
        } else {
            char letter = e.letter;
            if (letter >= new_xc_letter) letter++;
            new_raw_tag.add_entry(letter, e.block, e.is_blocked);
        }
    }
    auto desc = make_conv_layout_desc(tensor_kind, /*src_dst_with_group=*/true);
    return layout_tag_t(desc, layout_tag.type(), new_raw_tag);
}

layout_t make_conv_layout(tensor_kind_t tensor_kind, const layout_tag_t &_tag,
        bool is_dw, const prb_reqs_t &reqs) {
    auto tag = append_groups(tensor_kind, _tag, is_dw);
    layout_t ret(tag.desc(), tag.type());
    pvar_map_t<int> blocks;
    auto rem_size = [&](const pvar_t &dim, const pvar_map_t<int> &blocks) {
        auto dim_size = dim.var();
        bool is_dim_1 = reqs.is_equal(dim, 1);
        if (is_dim_1) return expr_t(1);
        if (!blocks.has(dim)) return dim_size;
        return binary_op_t::make(op_kind_t::_div_up, dim_size, blocks[dim]);
    };
    auto &entries = tag.raw_tag().entries();
    for (auto it = entries.rbegin(); it != entries.rend(); it++) {
        pvar_t dim = tag.desc().prb_dim(it->index());
        int block_size = it->block;
        expr_t block_size_expr;
        if (block_size > 0) {
            blocks[dim] = blocks.get(dim, 1) * block_size;
            block_size_expr = expr_t(block_size);
        } else {
            block_size_expr = rem_size(dim, blocks);
        }
        ret.add_block(dim, block_size_expr);
    }
    return ret;
}

class dim_mapper_manager_t {
public:
    dim_mapper_manager_t() = default;
    dim_mapper_manager_t(prop_kind_t prop, const prb_reqs_t &reqs)
        : prop_(prop), reqs_(reqs) {
        src_mapper_ = init_src_mapper();
        wei_mapper_ = init_wei_mapper();
        dst_mapper_ = init_dst_mapper();
        bia_mapper_ = init_bia_mapper();
    }

    const dim_mapper_t &mapper(tensor_kind_t tensor) const {
        switch (tensor) {
            case tensor_kind_t::src: return src_mapper_;
            case tensor_kind_t::wei: return wei_mapper_;
            case tensor_kind_t::dst: return dst_mapper_;
            case tensor_kind_t::a:
                return mapper(pick_a(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            case tensor_kind_t::b:
                return mapper(pick_b(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            case tensor_kind_t::c:
                return mapper(pick_c(prop_, tensor_kind_t::src,
                        tensor_kind_t::wei, tensor_kind_t::dst));
            case tensor_kind_t::bia: return bia_mapper_;
            default: ir_error_not_expected();
        }
        return src_mapper_;
    }

private:
    expr_t kw_idx = pvars::kw.index_var();
    expr_t kh_idx = pvars::kh.index_var();
    expr_t kd_idx = pvars::kd.index_var();
    expr_t id_idx = pvars::id.index_var();
    expr_t ih_idx = pvars::ih.index_var();
    expr_t iw_idx = pvars::iw.index_var();
    expr_t od_idx = pvars::od.index_var();
    expr_t oh_idx = pvars::oh.index_var();
    expr_t ow_idx = pvars::ow.index_var();

    dim_mapper_t init_src_mapper() const {
        auto pd = reqs_.to_expr(pvars::pd);
        auto ph = reqs_.to_expr(pvars::ph);
        auto pw = reqs_.to_expr(pvars::pw);
        auto sd = reqs_.to_expr(pvars::sd);
        auto sh = reqs_.to_expr(pvars::sh);
        auto sw = reqs_.to_expr(pvars::sw);
        auto dd = reqs_.to_expr(pvars::dd);
        auto dh = reqs_.to_expr(pvars::dh);
        auto dw = reqs_.to_expr(pvars::dw);
        dim_mapper_t mapper;
        mapper.set_dim(pvars::mb);
        mapper.set_dim(pvars::g);
        mapper.set_dim(pvars::ic);
        if (utils::one_of(
                    prop_, prop_kind::forward, prop_kind::backward_weights)) {
            auto dd_inc = const_fold(dd + 1);
            auto dh_inc = const_fold(dh + 1);
            auto dw_inc = const_fold(dw + 1);
            auto neg_pd = const_fold(-pd);
            auto neg_ph = const_fold(-ph);
            auto neg_pw = const_fold(-pw);
            mapper.set_dim(pvars::id,
                    simplify_rewrite(sd * od_idx + neg_pd + kd_idx * dd_inc),
                    true);
            mapper.set_dim(pvars::ih,
                    simplify_rewrite(sh * oh_idx + neg_ph + kh_idx * dh_inc),
                    true);
            mapper.set_dim(pvars::iw,
                    simplify_rewrite(sw * ow_idx + neg_pw + kw_idx * dw_inc),
                    true);
        } else {
            mapper.set_dim(pvars::id);
            mapper.set_dim(pvars::ih);
            mapper.set_dim(pvars::iw);
        }
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::src));
        return mapper;
    }

    dim_mapper_t init_wei_mapper() const {
        dim_mapper_t mapper;
        mapper.set_dim(pvars::g);
        mapper.set_dim(pvars::oc);
        mapper.set_dim(pvars::ic);
        mapper.set_dim(pvars::kd);
        mapper.set_dim(pvars::kh);
        mapper.set_dim(pvars::kw);
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::wei));
        return mapper;
    }

    dim_mapper_t init_bia_mapper() const {
        dim_mapper_t mapper;
        mapper.set_dim(pvars::g);
        mapper.set_dim(pvars::oc);
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::bia));
        return mapper;
    }

    dim_mapper_t init_dst_mapper() const {
        dim_mapper_t mapper;
        mapper.set_dim(pvars::mb);
        mapper.set_dim(pvars::g);
        mapper.set_dim(pvars::oc);
        if (utils::one_of(
                    prop_, prop_kind::forward, prop_kind::backward_weights)) {
            mapper.set_dim(pvars::od);
            mapper.set_dim(pvars::oh);
            mapper.set_dim(pvars::ow);
        } else {
            auto pd = reqs_.to_expr(pvars::pd);
            auto ph = reqs_.to_expr(pvars::ph);
            auto pw = reqs_.to_expr(pvars::pw);
            auto sd = reqs_.to_expr(pvars::sd);
            auto sh = reqs_.to_expr(pvars::sh);
            auto sw = reqs_.to_expr(pvars::sw);
            auto dd = reqs_.to_expr(pvars::dd);
            auto dh = reqs_.to_expr(pvars::dh);
            auto dw = reqs_.to_expr(pvars::dw);

            auto dd_inc = const_fold(dd + 1);
            auto dh_inc = const_fold(dh + 1);
            auto dw_inc = const_fold(dw + 1);

            mapper.set_dim(pvars::od,
                    simplify_rewrite((id_idx + pd - (kd_idx * dd_inc)) / sd),
                    true);
            mapper.set_dim(pvars::oh,
                    simplify_rewrite((ih_idx + ph - (kh_idx * dh_inc)) / sh),
                    true);
            mapper.set_dim(pvars::ow,
                    simplify_rewrite((iw_idx + pw - (kw_idx * dw_inc)) / sw),
                    true);
        }
        mapper.set_layout_desc(
                make_conv_algo_layout_desc(prop_, tensor_kind_t::dst));
        return mapper;
    }

    prop_kind_t prop_ = prop_kind::undef;
    prb_reqs_t reqs_;
    dim_mapper_t src_mapper_;
    dim_mapper_t wei_mapper_;
    dim_mapper_t dst_mapper_;
    dim_mapper_t bia_mapper_;
};

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
        , b_type_(b_type) {
        init_acc_type();
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
            default: ir_error_not_expected();
        }
        return false;
    }

    bool is(const pvar_t &dim, char bmnk) const {
        ir_assert(utils::one_of(bmnk, 'b', 'm', 'n', 'k'));
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
                    if (ret.has(d)) ir_assert(ret[d] == b_tile[d]);
                    ret[d] = b_tile[d];
                }
                return ret;
            }
            default: ir_error_not_expected();
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
            default: ir_error_not_expected();
        }
        return false;
    }

    layout_t to_compatible_layout(
            tensor_kind_t abc, const layout_t &layout) const {
        auto ret = layout;
        switch (abc) {
            case tensor_kind_t::a: ret.block_by(a_inner_.blocks()); break;
            case tensor_kind_t::b: ret.block_by(b_inner_.blocks()); break;
            default: ir_error_not_expected();
        }
        ret = get_fma_type_layout(ret);
        return ret;
    }

    layout_t acc_layout(const layout_t &a_layout, const layout_t &b_layout,
            const layout_t &c_layout) const {
        ir_assert(a_layout.has_const_sizes());
        ir_assert(b_layout.has_const_sizes());
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

    layout_t bia_layout(
            const layout_t &b_layout, const layout_t &bia_layout) const {
        ir_assert(b_layout.has_const_sizes());
        layout_t acc(bia_layout.desc(), acc_type());

        for (auto &b : b_layout.blocks()) {
            if (is_k(b.dim)) continue;
            acc.add_block(b.dim, b.size);
        }
        return acc;
    }

private:
    void init_acc_type() {
        ir_assert(a_type_.size() == b_type_.size());
        switch (fma_) {
            case fma_kind_t::mad:
                acc_type_ = a_type_.is_fp() ? type_t::f32() : type_t::s32();
                break;
            case fma_kind_t::dpas:
                acc_type_ = a_type_.is_fp() ? type_t::f32() : type_t::s32();
                break;
            default: ir_error_not_expected();
        }
    }

    bool fma_type_supported(const type_t &type) const {
        switch (fma_) {
            case fma_kind_t::mad:
                return utils::one_of(type, type_t::f32(), type_t::s16());
                break;
            case fma_kind_t::dpas:
                return utils::one_of(type, type_t::u8(), type_t::s8(),
                        type_t::f16(), type_t::bf16());
                break;
            default: ir_error_not_expected();
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
            default: ir_error_not_expected();
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
        ir_check(found) << "init_mad: cannot find dimension to vectorize.";
        c_inner_ = layout_t(c_desc, acc_type_, 0, b_inner_.blocks());
        return true;
    }

    bool init_dpas(const layout_desc_t &a_desc, const layout_desc_t &b_desc,
            const layout_desc_t &c_desc) {
        pvar_t m_dim;
        pvar_t n_dim;
        pvar_t k_dim;
        for (auto &d : iter_tile_) {
            switch (to_bmnk(d)) {
                case 'm':
                    ir_assert(m_dim.is_undef());
                    m_dim = d;
                    break;
                case 'n':
                    ir_assert(n_dim.is_undef());
                    n_dim = d;
                    break;
                case 'k':
                    ir_assert(k_dim.is_undef());
                    k_dim = d;
                    break;
                default: ir_error_not_expected();
            }
        }
        ir_check(!m_dim.is_undef() && !n_dim.is_undef() && !k_dim.is_undef())
                << "init_dpas: cannot initialize MNK dimensions.";
        dim_t m_size = iter_tile_.at(m_dim);
        dim_t n_size = iter_tile_.at(n_dim);
        dim_t k_size = iter_tile_.at(k_dim);
        uint8_t sdepth = 8;
        uint8_t rcount = 8;
        int type_size = a_type_.size();
        ir_check(m_size % rcount == 0)
                << "init_dpas: M dimension size is invalid: " << m_size;
        ir_check(n_size % simd_ == 0)
                << "init_dpas: N dimension size is invalid: " << n_size;
        ir_check((k_size * type_size) % (sdepth * 4) == 0)
                << "init_dpas: K dimension size is invalid: " << k_size;

        auto _dpas = dpas_t::make(
                /*is_dpasw=*/false, simd_, sdepth, rcount, acc_type_, b_type_,
                a_type_);
        auto &dpas = _dpas.as<dpas_t>();
        a_inner_ = to_v2_layout(
                dpas.b_layout(), a_desc, std::vector<pvar_t> {k_dim, m_dim});
        b_inner_ = to_v2_layout(
                dpas.a_layout(), b_desc, std::vector<pvar_t> {n_dim, k_dim});
        c_inner_ = to_v2_layout(
                dpas.c_layout(), c_desc, std::vector<pvar_t> {n_dim, m_dim});
        return true;
    }

    static layout_t to_v2_layout(const jit::layout_t &layout,
            const layout_desc_t &desc, const std::vector<pvar_t> &dims) {
        layout_t ret(desc, layout.type());
        for (auto &b : layout.blocks()) {
            auto dim = dims[b.dim_idx];
            ret.add_block(dim, b.block);
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
    plan_builder_t(const kernel_desc_t &desc) : desc_(desc) {
        reqs_ = desc_.reqs;
        desc_.reqs = prb_reqs_t();
    }

    const prb_reqs_t &reqs() const { return reqs_; }

    plan_t build() {
        init_dim_mapper_manager();
        init_tiles();
        init_layouts();
        if (!init_info()) return plan_t();
        return init_plan();
    }

private:
    static send_plan_t try_create_send_plan(const std::string &tag,
            const send_params_t &params, const view_t &view) {
        auto plan = create_send_plan(params, view, /*allow_fail=*/true);
        bool ok = [&]() {
            ir_check(plan) << tag << ": cannot create send plan" << std::endl
                           << params << std::endl
                           << ir_utils::add_tag("view", view.str());
            return true;
        }();
        if (!ok) return send_plan_t();
        return plan;
    }

    bool with_bias_reduce() const {
        return (desc_.prop == prop_kind::backward_weights && desc_.with_bias);
    }

    void add_align_req(const pvar_t &dim, const type_t &type,
            const align_desc_t::align_t &align) {
        int align_bytes
                = (align.in_bytes ? align.value : align.value * type.size());
        reqs_.add(
                dim.var() % ir_utils::safe_div(align_bytes, type.size()) == 0);
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

    void init_layouts() {
        auto src_layout = make_conv_layout(
                tensor_kind_t::src, desc_.src_tag, desc_.is_dw, reqs_);
        auto wei_layout = make_conv_layout(
                tensor_kind_t::wei, desc_.wei_tag, desc_.is_dw, reqs_);
        auto dst_layout = make_conv_layout(
                tensor_kind_t::dst, desc_.dst_tag, desc_.is_dw, reqs_);
        a_layout_ = pick_a(desc_.prop, src_layout, wei_layout, dst_layout);
        b_layout_ = pick_b(desc_.prop, src_layout, wei_layout, dst_layout);
        c_layout_ = pick_c(desc_.prop, src_layout, wei_layout, dst_layout);
        if (desc_.prop == prop_kind::backward_weights && desc_.with_bias)
            bia_layout_ = make_conv_layout(
                    tensor_kind_t::bia, desc_.bia_tag, desc_.is_dw, reqs_);
        auto &align = desc_.align;
        add_align_req(src_layout.blocks()[0].dim, src_layout.type(), align.src);
        add_align_req(wei_layout.blocks()[0].dim, wei_layout.type(), align.wei);
        add_align_req(dst_layout.blocks()[0].dim, dst_layout.type(), align.dst);
    }

    pvar_map_t<char> to_bmnk_map() const {
        pvar_map_t<char> ret;
        for (auto &d : conv_index_dims(desc_.prop)) {
            auto gemm_d = to_gemm(d, desc_.prop);
            ir_assert(!gemm_d.is_undef());
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
        plan_t plan(desc_.hw);
        if (!try_init_plan(plan, reqs_) || !check_plan(plan)) return plan_t();

        // Re-create plan to ensure all collected requirements are cross-used
        // between sub-plans.
        plan = plan_t(desc_.hw);
        if (!try_init_plan(plan, reqs_) || !check_plan(plan)) {
            ir_error_not_expected();
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
        ir_check(init_x2r_fma_plan(plan.x2r_fma, reqs));
        ir_check(init_prefetch_plan(
                plan.x2r_fma, plan.virt_grid, plan.prefetch));
        ir_check(init_epilogue_plan(
                plan.x2r_fma.c_layout, plan.virt_grid, plan.epilogue, reqs));
        if (desc_.prop == prop_kind::backward_weights && desc_.with_bias)
            ir_check(init_epilogue_bia(
                    plan.x2r_fma.bia_layout, plan.epilogue, reqs));
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
            ir_check(init_x_prefetch_plan(tensor_kind_t::a,
                    coord_info_.tg_iter_coord(), coord_info_.tg_iter_tile(),
                    x2r_fma, virt_grid, plan.a_prefetch));
        }
        if (desc_.prefetch.b) {
            ir_check(init_x_prefetch_plan(tensor_kind_t::b,
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
            reorder = reorder_plan_t(desc_.hw, src, dst);
            reg_layout = reorder.dst;
        }
        plan = x2r_plan_t(desc_.hw);
        plan.tensor_kind = abc;
        plan.load = std::move(load);
        plan.reorder = std::move(reorder);
        plan.layout = std::move(reg_layout);
        if (with_bias_reduce() && abc == tensor_kind_t::b) {
            auto bia_layout = mul_info_.bia_layout(plan.layout, bia_layout_);
            plan.bia_layout = std::move(bia_layout);
        }
        return true;
    }

    bool init_fma_plan(
            const layout_t &a, const layout_t &b, fma_plan_t &plan) const {
        auto inst_tile = mul_info_.inst_tile();
        auto acc_layout = mul_info_.acc_layout(a, b, c_layout_);
        ir_check(!acc_layout.is_empty()) << "init_fma_plan: cannot vectorize.";
        plan = fma_plan_t(desc_.hw);
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
        ir_assert(outer.is_empty() || outer.size() == 1);
        auto outer_dim = (outer.is_empty() ? pvar_t() : *outer.begin());
        dim_t outer_size = outer.get(outer_dim, 1);
        auto sub_tile = tile;
        if (!outer_dim.is_undef()) sub_tile[outer_dim] /= outer_size;
        bool is_outer_m = mul_info_.is_m(outer_dim);
        layout_t a_prev_layout;
        layout_t b_prev_layout;
        layout_t c_prev_layout;
        layout_t bia_prev_layout;
        int c_off_elems = 0;
        int bia_off_elems = 0;
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
                ir_check(init_x2r_plan(tensor_kind_t::a, a_sub_view, a));
                plan.add_stage(a);
                a_prev_layout = a.layout;
            }
            if (!is_outer_m || i == 0) {
                auto b_sub_view
                        = view_t(b_mapper, b_layout_, sub_coord, sub_tile);
                x2r_plan_t b;
                ir_check(init_x2r_plan(tensor_kind_t::b, b_sub_view, b));
                b_prev_layout = b.layout;
                if (with_bias_reduce()) {
                    bia_prev_layout = b.bia_layout;
                    b.bia_layout.set_base(bia_off_elems);
                    bia_off_elems += ir_utils::safe_div(
                            b.bia_layout.size(), b.bia_layout.type().size());
                }
                plan.add_stage(b);
            }

            fma_plan_t fma;
            ir_check(init_fma_plan(a_prev_layout, b_prev_layout, fma));
            ir_check(c_prev_layout.is_empty() || fma.c_layout == c_prev_layout)
                    << "init_x2r_fma_plan: inconsistent C layout from "
                       "subtiles.";
            c_prev_layout = fma.c_layout;
            fma.c_layout.set_base(c_off_elems);
            c_off_elems += ir_utils::safe_div(
                    fma.c_layout.size(), fma.c_layout.type().size());
            plan.add_stage(fma);
        }
        plan.c_layout = c_prev_layout;
        if (with_bias_reduce()) plan.bia_layout = bia_prev_layout;

        if (!outer_dim.is_undef()) {
            int stride = ir_utils::safe_div(
                    c_prev_layout.size(), c_prev_layout.type().size());
            plan.c_layout.add_block(outer_dim, outer_size, stride);
            if (with_bias_reduce()) {
                auto &bia_mapper
                        = dim_mapper_manager_.mapper(tensor_kind_t::bia);
                if (bia_mapper.has(outer_dim)) {
                    int bia_stride = ir_utils::safe_div(bia_prev_layout.size(),
                            bia_prev_layout.type().size());
                    plan.bia_layout.add_block(
                            outer_dim, outer_size, bia_stride);
                }
            }
        }
        reqs.add(plan.reqs());
        return true;
    }

    bool init_epilogue_bia(const layout_t &bia_layout, epilogue_plan_t &plan,
            prb_reqs_t &reqs) const {
        auto &bia_mapper = dim_mapper_manager_.mapper(tensor_kind_t::bia);
        auto bia_iter_view
                = view_t(dim_mapper_manager_.mapper(tensor_kind_t::bia),
                        bia_layout_, coord_info_.iter_coord(), desc_.iter_tile);
        auto reduce_cond = expr_t(true);
        for (int i = 0; i < c_layout_.desc().ndims(); i++) {
            auto dim = c_layout_.desc().prb_dim(i);
            if (!bia_mapper.has(dim))
                reduce_cond
                        = reduce_cond & (coord_info_.iter_coord()[dim] == 0);
        }
        plan.reduce_cond = std::move(reduce_cond);
        auto bia_params = get_send_params(
                tensor_kind_t::undef, send_op_t::store, bia_iter_view);
        auto bia_store = create_send_plan(bia_params, bia_iter_view);
        ir_check(reqs.implies(bia_store.reqs()))
                << "Bias store needs additional requirements.";
        auto tile = plan.tile;
        plan.bia_store = bia_store;
        plan.bia_reduced_reg_layout = bia_layout;
        if (bia_layout != bia_store.reg_layout()) {
            auto fma_layout = bia_layout.map(tile);
            auto store_layout = bia_store.reg_layout().map(tile);
            if (fma_layout != store_layout) {
                plan.bia_reorder = reorder_plan_t(desc_.hw);
                plan.bia_reorder.src = std::move(fma_layout);
                plan.bia_reorder.dst = std::move(store_layout);
            }
        }
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
        ir_assert(k_tg > 1);
        ir_assert(desc_.thread_group_tile.elems() == k_tg)
                << "Local k-slicing assumes no split by M/N.";
        ir_check(c_layout.size() % desc_.hw.grf_size() == 0)
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
        auto reduce = reduce_plan_t(desc_.hw, load_layout, reduced_layout);
        auto c_post_layout = std::move(reduced_layout);
        c_post_layout.remove(k_dim);

        plan = slm_reduce_plan_t(desc_.hw);
        plan.store = std::move(store);
        plan.load = std::move(load);
        plan.reduce = std::move(reduce);
        plan.c_layout = std::move(c_post_layout);
        plan.c_coord = coord_info_.iter_coord() + load_coord;

        return true;
    }

    bool init_epilogue_plan(const layout_t &c_fma_layout,
            virt_grid_t &virt_grid, epilogue_plan_t &plan,
            prb_reqs_t &reqs) const {
        ir_check(
                init_slm_reduce_plan(c_fma_layout, virt_grid, plan.slm_reduce));
        auto &c_mapper = dim_mapper_manager_.mapper(tensor_kind_t::c);
        auto c_reg_layout
                = (plan.slm_reduce ? plan.slm_reduce.c_layout : c_fma_layout);
        auto c_coord = (plan.slm_reduce ? plan.slm_reduce.c_coord
                                        : coord_info_.iter_coord());
        auto c_tile = c_reg_layout.int_dim_sizes();
        auto c_mem_view = view_t(c_mapper, c_layout_, c_coord, c_tile);
        int target_elems = 128 / c_layout_.type().size();
        auto it_beg = begin(c_mem_view.layout());
        auto it_end = end(c_mem_view.layout());
        auto tile_last = it_beg;
        for (auto it = it_beg; it != it_end; ++it) {
            if (it.elems() > target_elems) break;
            tile_last = it;
        }
        auto full_tile = desc_.iter_tile;
        for (auto &d : desc_.iter_tile) {
            if (mul_info_.is_k(d)) full_tile.unset(d);
        }
        auto params = get_send_params(
                tensor_kind_t::c, send_op_t::store, c_mem_view);
        // TODO: Implement fallback from 2D to block/scattered messages to
        // allow partial use of 2D messages when possible.
        auto c_store = try_create_send_plan(__func__, params, c_mem_view);
        if (!c_store) return false;
        auto &tile = c_store.entry_tile();
        plan.tile = tile;
        plan.c_store = c_store;
        plan.c_reg_layout = c_reg_layout;
        plan.c_coord = c_coord;
        auto c_reg_tile_layout = c_reg_layout.map(tile);
        auto store_layout = c_store.reg_layout().map(tile);
        if (c_reg_tile_layout != store_layout) {
            plan.reorder = reorder_plan_t(desc_.hw);
            plan.reorder.src = std::move(c_reg_tile_layout);
            plan.reorder.dst = std::move(store_layout);
        }
        reqs.add(plan.c_store.reqs());
        return true;
    }

    bool check_plan(const plan_t &plan) const {
        int grf_bound = desc_.hw.grf_size() * desc_.regs;
        int grf_bytes = plan.grf_usage_bytes();
        ir_check(grf_bytes <= grf_bound) << "check_plan: out of registers";
        int slm_bound = compute::device_info_t::max_slm_size_per_tg(
                convert_ngen_arch_to_dnnl(desc_.hw.to_ngen()),
                into<int>(desc_.thread_group_tile.elems()), desc_.regs > 128);
        int slm_bytes = plan.slm_usage_bytes();
        ir_check(slm_bytes <= slm_bound) << "check_plan: out of SLM";
        return true;
    }

    send_params_t get_send_params(tensor_kind_t abc, send_op_t op,
            const view_t &view, send_kind_t send_kind = send_kind_t::undef,
            send_address_t send_address = send_address_t::a64) const {
        send_params_t params;
        params.hw = desc_.hw;
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
        std::vector<pvar_t> ret;
        auto &mask_desc = view.mask_desc();
        auto tg_iter_tile = coord_info_.tg_iter_tile();
        auto dim_sizes = view.base_layout().dim_sizes();
        for (int i = 0; i < mask_desc.nmasks(); i++) {
            pvar_t dim = mask_desc[i].dim;
            ir_assert(view.dim_mapper().has(dim));
            // Assume that dimensions with non-trivial mapping always require
            // masking.
            if (!view.dim_mapper().expr(dim).is_same(dim.index_var())) continue;
            // Assume global k-slciing implies masking.
            if (coord_info_.is_global_loop(dim)) continue;
            // Check if the mask can be proven with known dimension requirements.
            if (!reqs_.can_prove(dim_sizes.at(dim) % tg_iter_tile.at(dim) == 0))
                continue;
            // Mask is not required for this dimension.
            ret.push_back(dim);
        }
        return ret;
    }

    dim_mapper_t extend_mapper(const dim_mapper_t &mapper,
            const pvar_t &extra_dim, char letter) const {
        auto new_mapper = mapper;
        new_mapper.set_dim(extra_dim);
        auto &desc = mapper.layout_desc();
        auto new_letter_map = desc.letter_map();
        new_letter_map[extra_dim] = letter;
        auto new_desc = layout_desc_t(new_letter_map);
        new_mapper.set_layout_desc(new_desc);
        return new_mapper;
    }

    kernel_desc_t desc_;

    dim_mapper_manager_t dim_mapper_manager_;
    multiply_info_t mul_info_;
    coord_info_t coord_info_;
    grid_t tg_grid_;
    grid_t thr_grid_;
    virt_grid_t virt_grid_;
    layout_t a_layout_;
    layout_t b_layout_;
    layout_t c_layout_;
    layout_t bia_layout_;
    prb_reqs_t reqs_;
};

template <typename KernelDescT>
plan_t create_conv_plan_impl(KernelDescT &desc, bool finalize) {
    if (!desc.is_supported()) return plan_t();
    ir_assert(!desc.has_spec_strategy())
            << "Kernel descriptor strategies are required to be specialized "
               "before plan creation";
    if (!finalize) {
        ir_assert(desc.is_finalized)
                << "Kernel descriptor must be finalized before plan creation";
    }
    plan_builder_t builder(desc);
    auto plan = builder.build();
    if (plan) {
        if (finalize) {
            const_cast<kernel_desc_t &>(desc).finalize(builder.reqs());
        } else {
            ir_assert(desc.reqs.implies(builder.reqs()));
        }
    }
    return plan;
}

plan_t create_conv_plan(const kernel_desc_t &desc) {
    return create_conv_plan_impl(desc, /*finalize=*/false);
}

bool finalize_conv_desc_impl(kernel_desc_t &desc, const hw_t &hw,
        const problem_t *prb, plan_t *out_plan) {
    if (desc.is_empty()) return false;
    if (desc.hw_desc.hw != hw.to_ngen()) return false;
    desc.hw = hw;
    if (!desc.is_supported()) return false;
    if (desc.is_finalized) return true;
    auto plan = create_conv_plan_impl(desc, /*finalize=*/true);
    if (plan) {
        if (out_plan) *out_plan = plan;
        if (prb && !desc.matches(*prb)) return false;
    }
    return (bool)plan;
}

bool finalize_conv_desc(
        kernel_desc_t &desc, const problem_t &prb, plan_t *plan) {
    return finalize_conv_desc_impl(desc, prb.hw(), &prb, plan);
}

bool finalize_conv_desc(kernel_desc_t &desc, const hw_t &hw, plan_t *plan) {
    return finalize_conv_desc_impl(desc, hw, nullptr, plan);
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
