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

#include "gpu/jit/conv/zp_plan.hpp"

#include "common/utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/plan_utils.hpp"
#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/gemm_schedule.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/send_plan.hpp"
#include "gpu/jit/ir/tensor.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

enum class zp_comp_kind_t {
    undef,
    // z:s32, w:x8, c:s32
    // c = 0
    // dp4a(c, c, w, 0x01010101)
    // mul (c, c, z)
    wei_Xn4k_x8_zp_common,
    // z:s32, w:x8, c:s32
    // c = 0
    // mad(c, c, z, w_s16) 4x for 4i reduction
    wei_Xn4k_x8_zp_per_k,
    // z:s32, w:s16, c:s32
    // c = 0
    // mad(c, c, z, w)
    wei_Xb_s16,
    // z:s32, w:s16, c:s32
    // c = 0
    // mad(c, c, z, w)
    wei_Xn_s16,
};

inline std::string to_string(zp_comp_kind_t kind) {
    switch (kind) {
#define CASE(name) \
    case zp_comp_kind_t::name: return #name
        CASE(undef);
        CASE(wei_Xn4k_x8_zp_common);
        CASE(wei_Xn4k_x8_zp_per_k);
        CASE(wei_Xb_s16);
        CASE(wei_Xn_s16);
        default: ir_error_not_expected();
#undef CASE
    }
    return "unknown";
}

inline std::ostream &operator<<(std::ostream &out, zp_comp_kind_t kind) {
    out << to_string(kind);
    return out;
}

static bool ends_with(
        const layout_t &layout, int i0, int b0, bool strict = false) {
    if (layout.nblocks() < 1) return false;
    auto &blocks = layout.blocks();
    if (blocks[0].dim_idx != i0) return false;
    if (strict && blocks[0].block != b0) return false;
    if (!strict && blocks[0].block % b0 != 0) return false;
    return true;
}

static bool ends_with(const layout_t &layout, int i1, int b1, int i0, int b0) {
    if (!ends_with(layout, i0, b0, /*strict=*/true)) return false;
    if (layout.nblocks() < 2) return false;
    auto &blocks = layout.blocks();
    if (blocks[1].dim_idx != i1) return false;
    if (blocks[1].block % b1 != 0) return false;
    return true;
}

class zp_comp_init_plan_t : public base_plan_t {
public:
    using base_plan_t::base_plan_t;

    zp_comp_init_plan_t(const ngen::HW hw, bool is_fwd,
            const layout_t &zp_layout, const layout_t &wei_layout)
        : base_plan_t(hw), zp_layout_(zp_layout), wei_layout_(wei_layout) {
        init_idxs(is_fwd);
        init_comp_layout();
        init_comp_kind();
    }

    const layout_t &comp_layout() const { return comp_layout_; }

    int ndims() const { return comp_layout_.ndims(); }

    int comp_reg_buf_size() const {
        int ret = utils::div_up(comp_layout_.size(), split_factor_);
        ret = utils::rnd_up(ret, grf_size());
        return ret;
    }

    bool can_split(abc_kind_t abc, int factor) const {
        if (abc != abc_kind_t::b || factor == 1) return true;

        if (wei_layout_.is_empty()) return false;
        if (comp_layout_.is_empty()) return false;

        auto &b_wei = wei_layout_.blocks().back();
        auto &b_comp = comp_layout_.blocks().back();
        if (b_wei.block % factor != 0) return false;
        if (b_wei.block != b_comp.block) return false;
        if (b_wei.dim_idx != b_comp.dim_idx) return false;
        int dim_idx = b_comp.dim_idx;
        dim_t subtile_dim = comp_layout_.dim(dim_idx);
        if (dim_idx == simd_dim_idx_ && subtile_dim < simd_) return false;

        return true;
    }

    void set_split(abc_kind_t abc, int factor) {
        ir_assert(can_split(abc, factor));
        if (abc == abc_kind_t::b) split_factor_ = factor;
    }

    int estimate_regs() const {
        int ret = 0;
        ret += comp_reg_buf_size();
        switch (kind_) {
            case zp_comp_kind_t::wei_Xn4k_x8_zp_common:
                // zp_1x4 buffer.
                ret += grf_size();
                break;
            case zp_comp_kind_t::wei_Xn4k_x8_zp_per_k:
                // wei_s16 buffer.
                ret += utils::rnd_up(simd_ * 4, grf_size());
                break;
            default: break;
        }
        return utils::div_up(ret, grf_size());
    }

    stmt_t create_stmt(buffer_manager_t &buf_mgr, const expr_t &zp_buf,
            const expr_t &wei_buf, const expr_t &comp_buf,
            int subtile_idx) const {
        if (split_factor_ == 1 && subtile_idx > 0) return stmt_t();
        stmt_t stmt;
        int ck_blk = 1;
        switch (kind_) {
            case zp_comp_kind_t::wei_Xn4k_x8_zp_common:
            case zp_comp_kind_t::wei_Xn4k_x8_zp_per_k: ck_blk = 4; break;
            default: break;
        }
        comp_layout_.for_each_tile(
                get_simd_tile(), [&](const std::vector<dim_t> &start) {
                    if (!in_subtile(start, subtile_idx)) return;
                    auto comp = comp_buf[get_comp_off(start)];
                    for (int ck = 0; ck < wei_layout_.dim(ck_idx_);
                            ck += ck_blk) {
                        auto zp = zp_buf[get_zp_off(start, ck)];
                        auto wei = wei_buf[get_wei_off(start, ck)];
                        stmt = stmt.append(
                                create_tile_stmt(zp, wei, comp, buf_mgr));
                    }
                    stmt = stmt.append(create_zp_common_mul_stmt(zp_buf, comp));
                });
        auto zp_1x4 = buf_mgr.get("zp_1x4");
        if (!zp_1x4.is_empty()) {
            auto init = store_t::make(zp_1x4, 0, 0x01010101);
            stmt = init.append(stmt);
        }
        auto comp_zero_out = funcs::zero_out(comp_buf, comp_reg_buf_size());
        stmt = comp_zero_out.append(stmt);
        return stmt;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "zp_layout:   " << zp_layout_ << std::endl;
        oss << "wei_layout:  " << wei_layout_ << std::endl;
        oss << "comp_layout: " << comp_layout_ << std::endl;
        oss << "kind:        " << kind_ << std::endl;
        oss << "SIMD:        " << simd_;
        return add_indent("comp_init", oss.str());
    }

    IR_DEFINE_DUMP()

private:
    void init_idxs(bool is_fwd) {
        g_idx_ = 0;
        cn_idx_ = is_fwd ? 1 : 2;
        ck_idx_ = is_fwd ? 2 : 1;
    }

    void init_comp_layout() {
        auto blocks = wei_layout_.blocks();
        for (auto &b : blocks) {
            if (b.dim_idx == ck_idx_) b.block = 1;
        }
        comp_layout_ = layout_t(type_t::s32(), wei_layout_.ndims(), 0, blocks);
        comp_layout_ = comp_layout_.make_dense();
    }

    void init_comp_kind() {
        if (is_wei_Xn4k_x8_zp_common(simd_dim_idx_, simd_)) {
            kind_ = zp_comp_kind_t::wei_Xn4k_x8_zp_common;
        } else if (is_wei_Xn4k_x8_zp_per_k(simd_dim_idx_, simd_)) {
            kind_ = zp_comp_kind_t::wei_Xn4k_x8_zp_per_k;
        } else if (is_wei_Xb_s16(simd_dim_idx_, simd_)) {
            kind_ = zp_comp_kind_t::wei_Xb_s16;
        } else if (is_wei_Xn_s16(simd_dim_idx_, simd_)) {
            kind_ = zp_comp_kind_t::wei_Xn_s16;
        } else {
            ir_error_not_expected() << wei_layout_;
        }
    }

    bool is_zp_common() const { return zp_layout_.elems() == 1; }

    bool is_wei_Xn4k_x8_zp_common(int &vec_dim_idx, int &simd) const {
        if (!wei_layout_.type().is_x8()) return false;
        if (!is_zp_common()) return false;
        for (int cn_blk : {16, 8}) {
            if (ends_with(wei_layout_, cn_idx_, cn_blk, ck_idx_, 4)) {
                vec_dim_idx = cn_idx_;
                simd = cn_blk;
                return true;
            }
        }
        return false;
    }

    bool is_wei_Xn4k_x8_zp_per_k(int &vec_dim_idx, int &simd) const {
        if (!wei_layout_.type().is_x8()) return false;
        if (is_zp_common()) return false;
        for (int cn_blk : {16, 8}) {
            if (ends_with(wei_layout_, cn_idx_, cn_blk, ck_idx_, 4)) {
                vec_dim_idx = cn_idx_;
                simd = cn_blk;
                return true;
            }
        }
        return false;
    }

    bool is_wei_Xy_s16(int y_idx, int &vec_dim_idx, int &simd) const {
        for (int y_blk : {32, 16, 8}) {
            if (y_blk > 16 && hw < ngen::HW::XeHPC) continue;
            if (ends_with(wei_layout_, y_idx, y_blk)) {
                vec_dim_idx = y_idx;
                simd = y_blk;
                return true;
            }
        }
        return false;
    }

    bool is_wei_Xb_s16(int &vec_dim_idx, int &simd) const {
        return is_wei_Xy_s16(g_idx_, vec_dim_idx, simd);
    }

    bool is_wei_Xn_s16(int &vec_dim_idx, int &simd) const {
        return is_wei_Xy_s16(cn_idx_, vec_dim_idx, simd);
    }

    tensor_t get_simd_tile() const {
        std::vector<dim_t> tile_dims(ndims(), 1);
        tile_dims[simd_dim_idx_] = simd_;
        return tensor_t(tile_dims);
    }

    int get_comp_off(const std::vector<dim_t> &start) const {
        int off = (int)comp_layout_.offset_in_bytes(start);
        return off % comp_reg_buf_size();
    }

    int wei_reg_buf_size() const {
        int ret = utils::div_up(wei_layout_.size(), split_factor_);
        ret = utils::rnd_up(ret, grf_size());
        return ret;
    }

    int get_wei_off(const std::vector<dim_t> &_start, int ck) const {
        auto start = _start;
        start[ck_idx_] = ck;
        int off = (int)wei_layout_.offset_in_bytes(start);
        return off % wei_reg_buf_size();
    }

    int get_zp_off(const std::vector<dim_t> &_start, int ck) const {
        std::vector<dim_t> start(2);
        start[0] = _start[g_idx_];
        start[1] = is_zp_common() ? 0 : ck;
        int off = (int)zp_layout_.offset_in_bytes(start);
        return off;
    }

    bool in_subtile(const std::vector<dim_t> &start, int subtile_idx) const {
        if (split_factor_ == 1) return true;

        auto &b = comp_layout_.blocks().back();
        dim_t subtile_dim = ir_utils::safe_divide(
                comp_layout_.dim(b.dim_idx), split_factor_);
        int beg = subtile_idx * subtile_dim;
        int end = beg + subtile_dim;
        return start[b.dim_idx] >= beg && start[b.dim_idx] < end;
    }

    stmt_t create_tile_stmt(const expr_t &zp, const expr_t &wei,
            const expr_t &comp, buffer_manager_t &buf_mgr) const {
        switch (kind_) {
            case zp_comp_kind_t::wei_Xn4k_x8_zp_common:
                return create_tile_wei_Xn4k_x8_zp_common(
                        zp, wei, comp, buf_mgr);
            case zp_comp_kind_t::wei_Xn4k_x8_zp_per_k:
                return create_tile_wei_Xn4k_x8_zp_per_k(zp, wei, comp, buf_mgr);
            case zp_comp_kind_t::wei_Xb_s16:
            case zp_comp_kind_t::wei_Xn_s16:
                return create_tile_wei_Xy_s16(zp, wei, comp);
            default: ir_error_not_expected();
        }
        return stmt_t();
    }

    stmt_t create_zp_common_mul_stmt(
            const expr_t &zp, const expr_t &comp) const {
        if (kind_ != zp_comp_kind_t::wei_Xn4k_x8_zp_common) return stmt_t();

        auto zp_type = zp_layout_.type();
        auto comp_type = comp_layout_.type();
        auto comp_load = load_t::make(comp_type.with_elems(simd_), comp, 0);
        auto zp_load = load_t::make(zp_type, zp, 0);
        auto zp_bcast = shuffle_t::make_broadcast(zp_load, simd_);
        auto comp_store = store_t::make(comp, 0, comp_load * zp_bcast);
        return comp_store;
    }

    stmt_t create_tile_wei_Xn4k_x8_zp_common(const expr_t &zp,
            const expr_t &wei, const expr_t &comp,
            buffer_manager_t &buf_mgr) const {
        UNUSED(zp);
        auto comp_type = comp_layout_.type();
        auto wei_type
                = wei_layout_.type().is_s8() ? type_t::s32() : type_t::u32();
        auto _1x4_type = type_t::s32();
        auto dp4a = dpas_t::make_dp4a(simd_, comp_type, wei_type, _1x4_type);
        auto zp_1x4 = buf_mgr.get("zp_1x4", grf_size());
        return dp4a.call({comp, comp, wei, zp_1x4});
    }

    stmt_t create_tile_wei_Xn4k_x8_zp_per_k(const expr_t &zp, const expr_t &wei,
            const expr_t &comp, buffer_manager_t &buf_mgr) const {
        int zp_stride = 0;
        int wei_s16_stride = 2;
        auto zp_type = zp_layout_.type();
        auto wei_x8_type = wei_layout_.type();
        auto wei_s16_type = type_t::s16();
        auto comp_type = comp_layout_.type();
        auto mad = mad_t::make(hw, comp_type, simd_, zp_type, zp_stride,
                wei_s16_type, wei_s16_stride);
        auto wei_s16_buf = buf_mgr.get(
                "zp_wei_s16", simd_ * wei_s16_type.size() * wei_s16_stride);
        stmt_t ret;
        for (int ic = 0; ic < 4; ic++) {
            auto wei_x8
                    = load_t::make(wei_x8_type.with_elems(simd_), wei, ic, 4);
            auto wei_s16 = cast(wei_x8, wei_s16_type.with_elems(simd_));
            auto store_s16 = store_t::make(wei_s16_buf, 0, wei_s16,
                    wei_s16_stride * type_t::s16().size());
            ret = ret.append(store_s16);
            ret = ret.append(mad.call(
                    {comp, comp, zp + zp_type.size() * ic, wei_s16_buf}));
        }
        return ret;
    }

    stmt_t create_tile_wei_Xy_s16(
            const expr_t &zp, const expr_t &wei, const expr_t &comp) const {
        int zp_stride = (kind_ == zp_comp_kind_t::wei_Xb_s16 && !is_zp_common())
                ? 1
                : 0;
        int wei_stride = 2;
        auto zp_type = zp_layout_.type();
        auto wei_type = wei_layout_.type();
        auto comp_type = comp_layout_.type();

        ir_assert((int)comp_layout_.inner_stride() == 1);
        ir_assert(wei_type.is_s16());
        ir_assert((int)wei_layout_.inner_stride() == wei_stride);

        auto mad = mad_t::make(
                hw, comp_type, simd_, zp_type, zp_stride, wei_type, wei_stride);
        return mad.call({comp, comp, zp, wei});
    }

    int g_idx_ = -1;
    int cn_idx_ = -1;
    int ck_idx_ = -1;

    layout_t zp_layout_;
    layout_t wei_layout_;
    layout_t comp_layout_;

    zp_comp_kind_t kind_ = zp_comp_kind_t::undef;
    int simd_dim_idx_ = -1; // comp layout index.
    int simd_ = -1;

    int split_factor_ = 1;
};

struct texpr_t {
    texpr_t() {
        std::fill(vidxs, vidxs + max_nvargs, -1);
        std::fill(vstrides, vstrides + max_nvargs, 0);
    }

    expr_t to_expr(const std::vector<expr_t> &vvars) const {
        auto ret = base;
        for (int i = 0; i < nvargs(); i++) {
            ret += vstrides[i] * vvars[vidxs[i]];
        }
        return ret;
    }

    expr_t to_expr(const std::vector<expr_t> &vstart,
            const std::vector<dim_t> vstart_inc,
            const std::vector<expr_t> &vvars, int simd_vidx) const {
        int ndims = (int)vstart.size();
        ir_assert((int)vstart_inc.size() == ndims);
        ir_assert((int)vvars.size() == ndims);
        bool non_linear[max_nvdims] = {false};
        if (has_non_linear) {
            auto vars = find_objects<var_t>(base);
            for (auto &v : vars) {
                for (int i = 0; i < (int)vvars.size(); i++) {
                    if (vvars[i].is_same(v)) {
                        ir_assert(i < max_nvdims);
                        non_linear[i] = true;
                        break;
                    }
                }
            }
        }
        ir_assert(!non_linear[simd_vidx]);

        auto ret = base;
        for (int i = 0; i < nvargs(); i++) {
            auto &s = vstart[vidxs[i]];
            auto &s_inc = vstart_inc[vidxs[i]];
            if (!is_zero(s)) ret += s * vstrides[i];
            if (s_inc != 0) ret += s_inc * vstrides[i];
        }
        if (has_non_linear) {
            for (int i = 0; i < ndims; i++) {
                if (!non_linear[i]) continue;
                ret = substitute(ret, vvars[i], vstart[i] + vstart_inc[i]);
            }
        }
        return ret;
    }

    bool is_const() const { return nvargs() == 0 && jit::is_const(base); }

    bool is_var() const { return nvargs() == 1 && is_zero(base); }

    int64_t to_const() const {
        ir_assert(is_const());
        return to_cpp<int64_t>(base);
    }

    bool has_vidx(int vidx, const std::vector<expr_t> &vvars) const {
        for (int i = 0; i < nvargs(); i++)
            if (vidxs[i] == vidx) return true;
        if (has_non_linear) {
            auto vars = find_objects<var_t>(base);
            for (auto &v : vars) {
                if (v.is_same(vvars[vidx])) return true;
            }
        }
        return false;
    }

    int vstride(int vidx) const {
        for (int i = 0; i < nvargs(); i++)
            if (vidxs[i] == vidx) { return vstrides[i]; }
        ir_error_not_expected() << "Dimension not found: " << vidx;
        return -1;
    }

    texpr_t operator+(const texpr_t &b) const { return add(b, 1); }
    texpr_t operator-(const texpr_t &b) const { return add(b, -1); }

    texpr_t add(const texpr_t &b, int mult) const {
        auto ret = *this;
        if (mult == 1) {
            ret.base += b.base;
        } else {
            ret.base -= b.base;
        }
        int cur_idx = nvargs();
        for (int i = 0; i < b.nvargs(); i++) {
            bool found = false;
            for (int j = 0; j < nvargs(); j++) {
                if (b.vidxs[i] == vidxs[j]) {
                    found = true;
                    ret.vstrides[j] += mult * b.vstrides[i];
                    break;
                }
            }
            if (!found) {
                ir_assert(cur_idx < max_nvargs);
                ret.vidxs[cur_idx] = b.vidxs[i];
                ret.vstrides[cur_idx] += mult * b.vstrides[i];
                cur_idx++;
            }
        }
        ret.has_non_linear |= b.has_non_linear;
        return ret;
    }

    texpr_t operator*(const texpr_t &b) const {
        if (!is_const() && b.is_const()) return b * *this;
        if (!is_const()) ir_error_not_expected();

        auto c = to_const();
        auto ret = b;
        ret.base *= c;
        for (int i = 0; i < ret.nvargs(); i++)
            ret.vstrides[i] *= c;
        ret.has_non_linear |= has_non_linear;
        return ret;
    }

    int nvargs() const {
        for (int i = max_nvargs - 1; i >= 0; i--) {
            if (vidxs[i] != -1) return i + 1;
        }
        return 0;
    }

    std::string str() const {
        std::ostringstream oss;
        std::vector<std::string> parts;
        if (!is_zero(base)) parts.push_back(base.str());
        for (int i = 0; i < nvargs(); i++) {
            auto s = "_" + std::to_string(vidxs[i]);
            if (vstrides[i] != 1) s = std::to_string(vstrides[i]) + " x " + s;
            parts.push_back(s);
        }
        for (int i = 0; i < (int)parts.size(); i++) {
            if (i > 0) oss << " + ";
            oss << parts[i];
        }
        if (parts.empty()) return "0";
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static texpr_t create_from_const(const expr_t &e) {
        ir_assert(jit::is_const(e));
        texpr_t ret;
        ret.base = e;
        return ret;
    }

    static texpr_t create_from_vidx(int vidx) {
        ir_assert(vidx != -1);
        texpr_t ret;
        ret.base = expr_t(0);
        ret.vidxs[0] = vidx;
        ret.vstrides[0] = 1;
        return ret;
    }

    static const int max_nvargs = 2;
    static const int max_nvdims = 16;

    expr_t base;
    int vidxs[max_nvargs];
    int64_t vstrides[max_nvargs];
    bool has_non_linear = false;
};

class zp_mask_desc_t {
public:
    zp_mask_desc_t(const expr_t &mask, const std::vector<expr_t> &vvars,
            const std::vector<expr_t> &vstart) {
        vinfo_t vinfo(vvars, vstart);
        if (is_x_op_y(mask, vinfo, op_, lhs_, rhs_)) return;
        ir_error_not_expected() << mask;
    }

    const texpr_t &lhs() const { return lhs_; }
    const texpr_t &rhs() const { return rhs_; }

    expr_t normalize(const std::vector<expr_t> &vvars,
            const std::vector<expr_t> &vstart,
            const std::vector<dim_t> &vstart_inc, int simd,
            int simd_vidx) const {
        auto e_lhs = lhs_.to_expr(vstart, vstart_inc, vvars, simd_vidx);
        auto e_rhs = rhs_.to_expr(vstart, vstart_inc, vvars, simd_vidx);
        e_lhs = shuffle_t::make_broadcast(e_lhs, simd);
        e_rhs = shuffle_t::make_broadcast(e_rhs, simd);
        if (!lhs_.has_vidx(simd_vidx, vvars)) {
            return binary_op_t::make(op_, e_lhs, e_rhs);
        }
        int stride = lhs_.vstride(simd_vidx);
        std::vector<expr_t> off;
        for (int i = 0; i < simd; i++) {
            off.push_back(-stride * i);
        }
        return binary_op_t::make(op_, e_lhs, e_rhs + shuffle_t::make(off));
    }

    std::string str() const {
        std::ostringstream oss;
        oss << lhs_ << " " << op_ << " " << rhs_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    struct vinfo_t {
        vinfo_t(const std::vector<expr_t> &vvars,
                const std::vector<expr_t> &vstart)
            : vvars(vvars), vstart(vstart) {}

        int vidx(const expr_t &var) const {
            for (int i = 0; i < (int)vvars.size(); i++)
                if (vvars[i].is_same(var)) return i;
            ir_error_not_expected();
            return -1;
        }

        const std::vector<expr_t> &vvars;
        const std::vector<expr_t> &vstart;
    };

    static bool is_x_op_y(const expr_t &e, const vinfo_t &vinfo, op_kind_t &op,
            texpr_t &lhs, texpr_t &rhs) {
        if (auto *binary = e.as_ptr<binary_op_t>()) {
            if (utils::one_of(binary->op_kind, op_kind_t::_lt, op_kind_t::_ge,
                        op_kind_t::_eq)) {
                op = binary->op_kind;
                lhs = to_texpr(binary->a, vinfo);
                rhs = to_texpr(binary->b, vinfo);
                ir_assert(rhs.nvargs() == 0);
                return true;
            }
        }
        return false;
    }

    static texpr_t to_texpr(const expr_t &e, const vinfo_t &vinfo) {
        if (is_const(e)) return texpr_t::create_from_const(e);
        if (is_var(e)) return texpr_t::create_from_vidx(vinfo.vidx(e));

        if (auto *binary = e.as_ptr<binary_op_t>()) {
            auto a = to_texpr(binary->a, vinfo);
            auto b = to_texpr(binary->b, vinfo);
            switch (binary->op_kind) {
                case op_kind_t::_add: return a + b;
                case op_kind_t::_sub: return a - b;
                case op_kind_t::_mul: return a * b;
                case op_kind_t::_div:
                case op_kind_t::_mod: {
                    ir_assert(b.is_const());
                    auto e_a = a.to_expr(vinfo.vvars);
                    auto e_b = b.base;
                    texpr_t ret;
                    ret.base = binary_op_t::make(binary->op_kind, e_a, e_b);
                    ret.has_non_linear = true;
                    return ret;
                }
                default: ir_error_not_expected() << e;
            }
        }
        ir_error_not_expected() << e;
        return texpr_t();
    }

    texpr_t lhs_;
    texpr_t rhs_;
    op_kind_t op_;
};

class conv_config_t;
class gemm_schedule_t;

class zp_mask_init_plan_t : public base_plan_t {
public:
    using base_plan_t::base_plan_t;

    zp_mask_init_plan_t(const conv_config_t &cfg,
            const gemm_schedule_t &gemm_schedule, const layout_t &src_layout)
        : base_plan_t(cfg.hw()) {
        auto &a_view = gemm_schedule.a_view();
        auto a_thr_tile = gemm_schedule.a_thr_tile(/*is_relative=*/false);
        vvars_ = a_view.vvars();
        vstart_ = a_thr_tile.start();
        init_mask_descs(cfg, a_view);
        init_mask_layout(src_layout, a_view.vvars());
    }

    const layout_t &mask_layout() const { return mask_layout_; }

    explicit operator bool() const { return !mask_layout_.is_empty(); }

    int mask_reg_buf_size() const {
        return utils::rnd_up(mask_layout_.size(), grf_size());
    }

    int estimate_regs() const {
        int ret = 0;
        ret += mask_reg_buf_size();
        return utils::div_up(ret, grf_size());
    }

    stmt_t create_stmt(const expr_t &mask_buf, int subtile_idx) const {
        if (!*this) return stmt_t();
        if (subtile_idx > 0) return stmt_t();

        stmt_t stmt;
        mask_layout_.for_each_tile(
                get_simd_tile(), [&](const std::vector<dim_t> &start) {
                    auto mask = mask_buf[get_mask_off(start)];
                    std::vector<expr_t> e_masks;
                    for (auto &m : mask_descs_) {
                        auto e_m = m.normalize(
                                vvars_, vstart_, start, simd_, simd_dim_idx_);
                        e_masks.push_back(e_m);
                    }
                    auto cond = e_masks[0];
                    for (int i = 1; i < (int)e_masks.size(); i++)
                        cond &= e_masks[i];
                    auto s32_type = type_t::s32().with_elems(simd_);
                    auto mask_s32 = -cast(cond, s32_type);
                    auto store
                            = store_t::make(mask, 0, cast(mask_s32, s32_type));
                    stmt = stmt.append(store);
                });
        return stmt;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "mask_layout: " << mask_layout_ << std::endl;
        oss << "SIMD:        " << simd_ << " (dim_idx: " << simd_dim_idx_
            << ")";
        return add_indent("mask_init", oss.str());
    }

    IR_DEFINE_DUMP()

private:
    void init_mask_descs(const conv_config_t &cfg, const view_t &a_view) {
        for (int i = 0; i < a_view.ntdims(); i++) {
            auto &tdim = a_view.tdim(i);
            if (tdim.is_identity()) {
                int vidx = tdim.vidx(0);
                auto &vvar = a_view.vvar(vidx);
                auto &name = vvar.as<var_t>().name;
                if (utils::one_of(name, "g", "ic", "oc")) continue;
                int padded = cfg.padded_dim(
                        conv_dim_t::from_name(vvar.as<var_t>().name));
                int dim = a_view.vdims()[vidx];
                if (dim != padded) add_mask_desc(mask_descs_, vvar < dim);
                continue;
            }

            auto tmask = tdim.mask();
            if (tmask.is_empty()) continue;
            tmask = substitute(tmask, tdim_t::placeholder_var(), tdim.expr());
            add_mask_desc(mask_descs_, tmask);
        }
    }

    void init_mask_layout(
            const layout_t &src_layout, const std::vector<expr_t> &vvars) {
        if (mask_descs_.empty()) return;
        int ndims = src_layout.ndims();
        ir_assert((int)vvars.size() == ndims);
        std::vector<dim_t> dims(ndims, 1);
        for (int i = 0; i < ndims; i++) {
            for (auto &m : mask_descs_)
                if (m.lhs().has_vidx(i, vvars)) {
                    dims[i] = src_layout.dim(i);
                    break;
                }
        }
        mask_layout_ = layout_t(type_t::s32(), 0, std::vector<dim_t>(ndims, 1));
        for (int i = 0; i < ndims; i++) {
            auto &name = vvars[i].as<var_t>().name;
            if (utils::one_of(name, "mb", "ow", "osp", "iw") && dims[i] != 1) {
                for (int b : {32, 16, 8}) {
                    if (b > 16 && hw < ngen::HW::XeHPC) continue;
                    if (dims[i] >= b && is_linear_mask(i)) {
                        simd_dim_idx_ = i;
                        simd_ = b;
                        break;
                    }
                }
                if (simd_ != 1) {
                    mask_layout_ = mask_layout_.add_outer_block(i, simd_);
                    dims[i] = utils::div_up((int)dims[i], simd_);
                    break;
                }
            }
        }
        for (int i = 0; i < ndims; i++) {
            if (dims[i] != 1)
                mask_layout_ = mask_layout_.add_outer_block(i, dims[i]);
        }
    }

    bool is_linear_mask(int idx) const {
        for (auto &md : mask_descs_) {
            if (md.lhs().has_non_linear && md.lhs().has_vidx(idx, vvars_))
                return false;
            if (md.rhs().has_non_linear && md.rhs().has_vidx(idx, vvars_))
                return false;
        }
        return true;
    }

    int get_mask_off(const std::vector<dim_t> &start) const {
        int off = (int)mask_layout_.offset_in_bytes(start);
        return off;
    }

    tensor_t get_simd_tile() const {
        std::vector<dim_t> tile_dims(mask_layout_.ndims(), 1);
        tile_dims[simd_dim_idx_] = simd_;
        return tensor_t(tile_dims);
    }

    void add_mask_desc(
            std::vector<zp_mask_desc_t> &mask_descs, const expr_t &mask) const {
        auto masks = split_by_and(mask);
        for (auto &m : masks) {
            if (is_const(m)) {
                ir_assert(to_cpp<bool>(m));
                continue;
            }
            mask_descs.emplace_back(m, vvars_, vstart_);
        }
    }

    std::vector<zp_mask_desc_t> mask_descs_;
    std::vector<expr_t> vvars_;
    std::vector<expr_t> vstart_;
    layout_t mask_layout_;
    int simd_ = 1;
    int simd_dim_idx_ = 0;
};

class zp_comp_apply_plan_t : public base_plan_t {
public:
    using base_plan_t::base_plan_t;

    zp_comp_apply_plan_t(const ngen::HW hw, bool is_fwd,
            const layout_t &comp_layout, const layout_t &mask_layout,
            const layout_t &c_layout, const bmnk_mapper_t &mapper)
        : base_plan_t(hw)
        , comp_layout_(comp_layout)
        , mask_layout_(mask_layout)
        , c_layout_(c_layout) {
        if (!mask_layout_.is_empty()) {
            ir_assert(utils::one_of(mask_layout_.ndims(), 7, 9));
        }
        init_idxs(is_fwd);
        init_simd();
        init_splits(mapper);
    }

    bool can_split(abc_kind_t abc, int factor) const {
        if (factor == 1) return true;
        auto &splits = (abc == abc_kind_t::a) ? a_splits_ : b_splits_;
        if (factor >= (int)splits.size()) return false;
        return (bool)splits[factor];
    }

    void set_split(abc_kind_t abc, int factor) {
        ir_assert(can_split(abc, factor));
        if (factor == 1) {
            split_ = split_t::no_split();
            return;
        }
        auto &splits = (abc == abc_kind_t::a) ? a_splits_ : b_splits_;
        split_ = splits[factor];
    }

    stmt_t create_stmt(const expr_t &comp_buf, const expr_t &mask_buf,
            const expr_t &c_buf, int subtile_idx) const {
        int kw_dim = comp_layout_.dim(comp_kw_idx_);
        stmt_t stmt;
        c_layout_.for_each_tile(
                get_simd_tile(), [&](const std::vector<dim_t> &start) {
                    if (!split_.in_subtile(start, subtile_idx)) return;
                    for (int kw = 0; kw < kw_dim; kw++) {
                        auto comp = comp_buf[get_comp_off(start, kw)];
                        auto mask = mask_buf.is_empty()
                                ? expr_t()
                                : mask_buf[get_mask_off(start, kw)];
                        auto c = c_buf[get_c_off(start, kw)];
                        stmt = stmt.append(create_tile_stmt(comp, mask, c));
                    }
                });
        return stmt;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "comp_layout: " << comp_layout_ << std::endl;
        oss << "c_layout:    " << c_layout_ << std::endl;
        oss << "SIMD:        " << simd_ << " (dim_idx: " << simd_dim_idx_
            << ")";
        return add_indent("comp_apply", oss.str());
    }

    IR_DEFINE_DUMP()

private:
    class split_t {
    public:
        split_t() = default;
        split_t(const layout_t &c, const bmnk_mapper_t &mapper, abc_kind_t abc,
                int factor, int simd_dim_idx, int simd) {
            ir_assert(factor > 1);
            bmnk_kind_t split_mn
                    = (abc == abc_kind_t::a ? bmnk_kind_t::m : bmnk_kind_t::n);
            dim_t dim = 1;
            std::vector<block_t> mn_blocks;
            for (auto &b : c.blocks()) {
                if (b.block == 1) continue;
                auto bmnk = mapper.bmnk_kind(abc_kind_t::c, b.dim_idx);
                if (bmnk != split_mn) continue;
                dim *= b.block;
                mn_blocks.push_back(b);
            }
            if (dim % factor != 0) return;
            dim_t subtile_dim = dim / factor;
            dim_t cur_dim = 1;
            int cur_idx = -1;
            for (auto &b : mn_blocks) {
                if (cur_idx == -1) {
                    cur_idx = b.dim_idx;
                } else if (b.dim_idx != cur_idx) {
                    return;
                }
                dim_t max_block = std::min(b.block, subtile_dim / cur_dim);
                if (b.block % max_block != 0) return;
                cur_dim *= max_block;
                if (cur_dim == subtile_dim) break;
            }
            if (cur_dim != subtile_dim) return;
            if (cur_idx == simd_dim_idx && cur_dim < simd) return;
            abc_ = abc;
            factor_ = factor;
            dim_idx_ = cur_idx;
            subtile_dim_ = subtile_dim;
        }

        static split_t no_split() {
            split_t ret;
            ret.factor_ = 1;
            return ret;
        }

        abc_kind_t abc() const { return abc_; }
        int factor() const { return factor_; }

        explicit operator bool() const { return factor_ != 0; }

        bool in_subtile(
                const std::vector<dim_t> &start, int subtile_idx) const {
            if (!*this) return false;
            if (factor_ == 1) return true;
            int beg = subtile_idx * subtile_dim_;
            int end = beg + subtile_dim_;
            return start[dim_idx_] >= beg && start[dim_idx_] < end;
        }

    private:
        abc_kind_t abc_ = abc_kind_t::undef;
        int factor_ = 0;
        int dim_idx_ = -1;
        int subtile_dim_ = -1;
    };

    void init_idxs(bool is_fwd) {
        ir_assert(comp_layout_.ndims() == 6);
        comp_g_idx_ = 0;
        comp_c_idx_ = is_fwd ? 1 : 2;
        comp_kw_idx_ = 5;
        c_g_idx_ = 1;
        c_c_idx_ = 2;
    }

    void init_simd() {
        for (int b : {32, 16, 8}) {
            if (b > 16 && hw < ngen::HW::XeHPC) continue;
            if (ends_with(comp_layout_, comp_g_idx_, b)
                    && ends_with(c_layout_, c_g_idx_, b)) {
                simd_dim_idx_ = 1;
                simd_ = b;
                return;
            }
            if (ends_with(comp_layout_, comp_c_idx_, b)
                    && ends_with(c_layout_, c_c_idx_, b)) {
                simd_dim_idx_ = 2;
                simd_ = b;
                return;
            }
        }
        ir_error_not_expected()
                << "Can't initialize SIMD size, comp_layout = " << comp_layout_
                << ", c_layout = " << c_layout_;
    }

    void init_splits(const bmnk_mapper_t &mapper) {
        for (auto abc : {abc_kind_t::a, abc_kind_t::b}) {
            for (int factor : {2, 4}) {
                auto &splits = (abc == abc_kind_t::a) ? a_splits_ : b_splits_;
                if ((int)splits.size() <= factor) splits.resize(factor + 1);
                splits[factor] = split_t(
                        c_layout_, mapper, abc, factor, simd_dim_idx_, simd_);
            }
        }
    }

    tensor_t get_simd_tile() const {
        std::vector<dim_t> tile_dims(c_layout_.ndims(), 1);
        tile_dims[simd_dim_idx_] = simd_;
        return tensor_t(tile_dims);
    }

    stmt_t create_tile_stmt(
            const expr_t &comp, const expr_t &mask, const expr_t &c) const {
        auto comp_type = comp_layout_.type();
        auto c_type = c_layout_.type();
        ir_assert((int)comp_layout_.inner_stride() == 1);
        auto comp_load = load_t::make(comp_type.with_elems(simd_), comp, 0);
        auto c_load = load_t::make(c_type.with_elems(simd_), c, 0);
        stmt_t c_update;
        if (mask.is_empty()) {
            c_update = store_t::make(c, 0, c_load - comp_load);
        } else {
            auto mad = mad_t::make(
                    hw, c_type, simd_, comp_type, 1, type_t::s16(), 0);
            c_update = mad.call({c, c, comp, mask});
        }
        return c_update;
    }

    int comp_reg_buf_size() const {
        int ret = (int)comp_layout_.size();
        if (split_.abc() == abc_kind_t::b) {
            ret = utils::div_up(ret, split_.factor());
        }
        return utils::rnd_up(ret, grf_size());
    }

    int get_comp_off(const std::vector<dim_t> &_start, int kw) const {
        auto start = c_to_comp(_start);
        start[comp_kw_idx_] = kw;
        int off = (int)comp_layout_.offset_in_bytes(start);
        return off % comp_reg_buf_size();
    }

    int get_mask_off(const std::vector<dim_t> &_start, int kw) const {
        auto start = c_to_mask(_start);
        int mask_kw_idx = mask_layout_.ndims() - 1;
        start[mask_kw_idx] = kw;
        int off = (int)mask_layout_.offset_in_bytes(start);
        return off;
    }

    int get_c_off(const std::vector<dim_t> &start, int kw) const {
        int off = (int)c_layout_.offset_in_bytes(start);
        return off;
    }

    std::vector<dim_t> c_to_comp(const std::vector<dim_t> &c) const {
        // c:    ngcdhw or ngc[osp]
        // comp: goidhw
        std::vector<dim_t> ret(6);
        ret[comp_g_idx_] = c[c_g_idx_];
        ret[comp_c_idx_] = c[c_c_idx_];
        return ret;
    }

    std::vector<dim_t> c_to_mask(const std::vector<dim_t> &c) const {
        // c:    ngcdhw or ngc[osp]
        // mask: ngcdhw[kd][kh][kw] or ngc[osp][kd][kh][kw]
        std::vector<dim_t> ret(mask_layout_.ndims());
        for (int i = 0; i < (int)c.size(); i++)
            ret[i] = c[i];
        return ret;
    }

    int comp_g_idx_ = -1;
    int comp_c_idx_ = -1;
    int comp_kw_idx_ = -1;
    int c_g_idx_ = -1;
    int c_c_idx_ = -1;

    layout_t comp_layout_;
    layout_t mask_layout_;
    layout_t c_layout_;

    int simd_dim_idx_ = -1; // C layout index.
    int simd_ = -1;

    std::vector<split_t> a_splits_;
    std::vector<split_t> b_splits_;
    split_t split_ = split_t::no_split();
};

struct zp_plan_impl_t : public base_plan_t {
    send_plan_t load;
    zp_comp_init_plan_t comp_init;
    zp_mask_init_plan_t mask_init;
    zp_comp_apply_plan_t comp_apply;

    zp_plan_impl_t(ngen::HW hw)
        : base_plan_t(hw), comp_init(hw), mask_init(hw), comp_apply(hw) {}

    explicit operator bool() const { return (bool)load; }

    bool can_split(abc_kind_t abc, int factor) const {
        if (!comp_init.can_split(abc, factor)) return false;
        if (!comp_apply.can_split(abc, factor)) return false;
        return true;
    }

    void set_split(abc_kind_t abc, int factor) {
        ir_assert(can_split(abc, factor));
        comp_init.set_split(abc, factor);
        comp_apply.set_split(abc, factor);
    }

    int estimate_regs() const {
        if (!*this) return 0;
        int ret = 0;
        ret += comp_init.estimate_regs();
        ret += mask_init.estimate_regs();
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << load.str("load") << std::endl;
        oss << comp_init << std::endl;
        oss << mask_init << std::endl;
        oss << comp_apply << std::endl;
        return add_indent("zp", oss.str());
    }

    IR_DEFINE_DUMP()
};

zp_plan_t::zp_plan_t(ngen::HW hw)
    : impl(utils::make_unique<zp_plan_impl_t>(hw)) {};

zp_plan_t::~zp_plan_t() = default;

void zp_plan_t::init(const conv_config_t &cfg,
        const gemm_schedule_t &gemm_schedule, const view_t &zp_view,
        const layout_t &src_layout, const layout_t &wei_layout,
        const layout_t &dst_layout) {
    auto &exec_cfg = cfg.exec_cfg();
    auto load_params = get_send_params(
            exec_cfg, send_op_t::load, send_address_t::a64, zp_view);
    impl->load = create_send_plan(exec_cfg, zp_view, load_params);
    impl->comp_init = zp_comp_init_plan_t(
            cfg.hw(), cfg.prb().is_fwd, impl->load.reg_layout(), wei_layout);
    impl->mask_init = zp_mask_init_plan_t(cfg, gemm_schedule, src_layout);
    impl->comp_apply = zp_comp_apply_plan_t(cfg.hw(), cfg.prb().is_fwd,
            impl->comp_init.comp_layout(), impl->mask_init.mask_layout(),
            dst_layout, gemm_schedule.bmnk_mapper());
}

zp_plan_t::operator bool() const {
    return (bool)*impl;
}

int zp_plan_t::load_reg_buf_size() const {
    return impl->load.reg_buf_size();
}

int zp_plan_t::mask_reg_buf_size() const {
    return impl->mask_init.mask_reg_buf_size();
}

int zp_plan_t::comp_reg_buf_size() const {
    return impl->comp_init.comp_reg_buf_size();
}

stmt_t zp_plan_t::load_create_stmt(
        const expr_t &mem_buf, const expr_t &reg_buf, int subtile_idx) const {
    return subtile_idx > 0
            ? stmt_t()
            : impl->load.create_stmt(mem_buf, reg_buf, subtile_idx);
}

stmt_t zp_plan_t::comp_init_create_stmt(buffer_manager_t &buf_mgr,
        const expr_t &zp_buf, const expr_t &wei_buf, const expr_t &comp_buf,
        int subtile_idx) const {
    return impl->comp_init.create_stmt(
            buf_mgr, zp_buf, wei_buf, comp_buf, subtile_idx);
}

stmt_t zp_plan_t::mask_init_create_stmt(
        const expr_t &mask_buf, int subtile_idx) const {
    return impl->mask_init.create_stmt(mask_buf, subtile_idx);
}

stmt_t zp_plan_t::comp_apply_create_stmt(const expr_t &comp_buf,
        const expr_t &mask_buf, const expr_t &c_buf, int subtile_idx) const {
    return impl->comp_apply.create_stmt(comp_buf, mask_buf, c_buf, subtile_idx);
}

bool zp_plan_t::can_split(abc_kind_t abc, int factor) const {
    return impl->can_split(abc, factor);
}

void zp_plan_t::set_split(abc_kind_t abc, int factor) {
    impl->set_split(abc, factor);
}

int zp_plan_t::estimate_regs() const {
    return impl->estimate_regs();
}

std::string zp_plan_t::str() const {
    return impl->str();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
