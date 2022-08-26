/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "gpu/jit/conv/kernel_builder.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gpu/jit/conv/builder_utils.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/cse.hpp"
#include "gpu/jit/conv/epilogue.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/gemm_schedule.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/pass/pass.hpp"
#include "gpu/jit/conv/pipeline.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/reduce_support.hpp"
#include "gpu/jit/conv/reorder_support.hpp"
#include "gpu/jit/conv/slm_reduce_builder.hpp"
#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class buffer_access_verifier_t : public ir_visitor_t {
public:
    void _visit(const alloc_t &obj) override {
        buf_sizes_.emplace(obj.buf, obj.size);
        ir_visitor_t::_visit(obj);
        buf_sizes_.erase(obj.buf);
    }

    void _visit(const func_call_t &obj) override {
        auto &func = obj.func;
        if (auto *dpas = func.as_ptr<dpas_t>()) {
            auto &dst = dpas_t::arg_dst(obj);
            auto &src0 = dpas_t::arg_src0(obj);
            auto &src1 = dpas_t::arg_src1(obj);
            auto &src2 = dpas_t::arg_src2(obj);
            check_access(dst, dpas->dst_size(), obj);
            if (!is_zero(src0)) check_access(src0, dpas->src0_size(), obj);
            check_access(src1, dpas->src1_size(), obj);
            check_access(src2, dpas->src2_size(), obj);
        } else if (func.is<eltwise_t>()) {
            auto &elems = eltwise_t::arg_elems(obj);
            auto &data = eltwise_t::arg_data(obj);
            int size = to_cpp<int>(elems) * sizeof(float);
            check_access(data, size, obj);
        } else if (auto *mad = func.as_ptr<mad_t>()) {
            auto &dst = mad_t::arg_dst(obj);
            auto &src0 = mad_t::arg_src0(obj);
            auto &src1 = mad_t::arg_src1(obj);
            auto &src2 = mad_t::arg_src2(obj);
            check_access(dst, mad->dst_size(), obj);
            if (!is_zero(src0)) check_access(src0, mad->src0_size(), obj);
            check_access(src1, mad->src1_size(), obj);
            check_access(src2, mad->src2_size(), obj);
        } else if (auto *reduce = func.as_ptr<reduce_t>()) {
            auto &dst_buf = reduce_t::arg_dst_buf(obj);
            auto &src_buf = reduce_t::arg_src_buf(obj);
            check_access(dst_buf, reduce->dst_layout.size(), obj);
            check_access(src_buf, reduce->src_layout.size(), obj);
        } else if (auto *reorder = func.as_ptr<reorder_t>()) {
            auto &dst_buf = reorder_t::arg_dst_buf(obj);
            auto &src_buf = reorder_t::arg_src_buf(obj);
            check_access(dst_buf, reorder->dst_layout.size(), obj);
            check_access(src_buf, reorder->src_layout.size(), obj);
            return;
        } else if (auto *send = func.as_ptr<send_t>()) {
            if (!send->is_prefetch() && !send->is_prefetch_2d()) {
                auto &reg_buf = send_t::arg_reg_buf(obj);
                int size = send->payload_size();
                check_access(reg_buf, size, obj);
            }
            return;
        } else if (func.is<builtin_t>()) {
            // No buffers to check.
        } else {
            ir_error_not_expected() << "Unhandled function: " << obj;
        }

        ir_visitor_t::_visit(obj);
    }

    void _visit(const load_t &obj) override {
        auto elem_type = obj.type.scalar();
        int stride_bytes
                = (obj.has_default_stride() ? elem_type.size() : obj.stride);
        int off = to_cpp<int>(obj.off);
        auto stride = (obj.type.elems() - 1) * stride_bytes;
        check_access(obj.buf + off, stride + elem_type.size(), obj);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const store_t &obj) override {
        auto elem_type = obj.value.type().scalar();
        int stride_bytes
                = (obj.has_default_stride() ? elem_type.size() : obj.stride);
        int off = to_cpp<int>(obj.off);
        auto stride = (obj.value.type().elems() - 1) * stride_bytes;
        check_access(obj.buf + off, stride + elem_type.size(), obj);
        ir_visitor_t::_visit(obj);
    }

private:
    void check_access(const expr_t &buf, int size, const object_t &obj) {
        auto &base = (is_var(buf) ? buf : buf.as<ptr_t>().base);
        int off = (is_var(buf) ? 0 : to_cpp<int>(buf.as<ptr_t>().off));
        auto it = buf_sizes_.find(base);
        ir_assert(it != buf_sizes_.end())
                << "Can't find allocation for buffer: " << buf;
        int buf_size = it->second;
        ir_assert(off + size <= buf_size)
                << "Invalid access:\n    " << obj << "\n    Buffer " << base
                << " has size: " << buf_size;
    }

    object_map_t<expr_t, int> buf_sizes_;
};

void verify_buffer_access(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    buffer_access_verifier_t verifier;
    verifier.visit(s);
    trace_pass("verify_buffer_access", s, ir_ctx);
}

class multiply_builder_t {
public:
    multiply_builder_t() = default;

    multiply_builder_t(const conv_config_t &cfg,
            const bmnk_mapper_t &bmnk_mapper, const view_t &a_view,
            const view_t &b_view, const expr_t &a_buf, const expr_t &b_buf,
            const expr_t &c_buf)
        : hw_(cfg.hw())
        , simd_size_(cfg.simd_size())
        , bmnk_mapper_(bmnk_mapper)
        , a_view_(a_view)
        , b_view_(b_view)
        , a_buf_(a_buf)
        , b_buf_(b_buf)
        , c_buf_(c_buf) {
        switch (cfg.fma_kind) {
            case fma_kind_t::dp4a:
            case fma_kind_t::dpas:
            case fma_kind_t::dpasw:
                if (try_build_dpas()) return;
                break;
            case fma_kind_t::mad:
                if (try_build_mad()) return;
                break;
            default: ir_error_not_expected() << "Unknown FMA kind.";
        }

        ir_error_not_expected()
                << "Can't decompose into multiplication instructions. A view: "
                << a_view << ". B view: " << b_view;
    }

    const stmt_t &stmt() const { return stmt_; }

    const layout_t &c_layout() const { return c_layout_; }

    bool do_transpose() const { return do_transpose_; }

    std::string str() const {
        std::ostringstream oss;
        oss << "A view:    " << a_view_ << std::endl;
        oss << "B view:    " << b_view_ << std::endl;
        oss << "C layout:  " << c_layout_ << std::endl;
        oss << "Statement: " << std::endl << stmt_;
        return oss.str();
    }

private:
    struct loop_info_t {
        loop_info_t() = default;

        loop_info_t(const expr_t &var, bmnk_kind_t bmnk_kind, int dim)
            : var(var), bmnk_kind(bmnk_kind), dim(dim) {}

        expr_t var;
        bmnk_kind_t bmnk_kind;

        int dim;
        int a_idx = -1;
        int b_idx = -1;
        int c_idx = -1;
        int block = 1;
    };

    bool try_build_dpas() {
        ir_assert(a_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";
        ir_assert(b_view_.can_convert_to_vlayout())
                << "Views are not supported with dpas/dpasw.";

        auto a_layout = a_view_.create_vlayout();
        auto b_layout = b_view_.create_vlayout();

        check_k_blocks_order(a_layout, b_layout);

        bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper_);
        from_bmnk_mapper.push_blocks(abc_kind_t::a, a_layout.blocks());
        from_bmnk_mapper.push_blocks(abc_kind_t::b, b_layout.blocks());

        // Convert to MNK layouts.
        a_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::a, {bmnk_kind_t::m, bmnk_kind_t::k}, a_layout);
        b_layout = bmnk_mapper_.map_to_bmnk(
                abc_kind_t::b, {bmnk_kind_t::k, bmnk_kind_t::n}, b_layout);

        multiply_desc_t desc(a_layout, b_layout, /*force_c_upconvert=*/true);
        if (!dpas_t::matches_types(
                    hw_, desc.a_type(), desc.b_type(), desc.c_type()))
            return false;

        int sdepth = 8;
        int rcount = std::min(utils::rnd_up_pow2(desc.n()), 8);
        auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size_, sdepth,
                rcount, desc.c_type(), desc.a_type(), desc.b_type());
        if (_dpas.as<dpas_t>().matches(desc)) {
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }

        // Try to transpose and flip: C += A * B -> C^T = B^T * A^T.
        rcount = std::min(desc.m(), 8);
        desc = multiply_desc_t(
                b_layout.transpose(), a_layout.transpose(), true);
        _dpas = dpas_t::make(/*is_dpasw=*/false, /*exec_size=*/simd_size_,
                sdepth, rcount, desc.c_type(), desc.a_type(), desc.b_type());

        if (_dpas.as<dpas_t>().matches(desc)) {
            do_transpose_ = true;
            build_dpas(from_bmnk_mapper, _dpas.as<dpas_t>(), desc);
            return true;
        }
        return false;
    }

    void check_k_blocks_order(const layout_t &a, const layout_t &b) const {
        object_map_t<expr_t, int> k_vars;
        auto k_sub_layout = [&](abc_kind_t abc_kind, const layout_t &l) {
            layout_t k_layout = layout_t(type_t::u8(), 0,
                    std::vector<dim_t>(layout_t::max_ndims, 1));
            for (auto &b : l.blocks()) {
                auto bmnk_kind = bmnk_mapper_.bmnk_kind(abc_kind, b.dim_idx);
                if (bmnk_kind != bmnk_kind_t::k) continue;
                auto &var = bmnk_mapper_.var(abc_kind, b.dim_idx);
                auto ret = k_vars.emplace(var, (int)k_vars.size());
                k_layout = k_layout.add_outer_block(ret.first->second, b.block);
            }
            return k_layout;
        };
        auto a_k = k_sub_layout(abc_kind_t::a, a);
        auto b_k = k_sub_layout(abc_kind_t::b, b);
        ir_assert(a_k == b_k)
                << "Order of K dimensions doesn't match in A and B. A layout: "
                << a << ", B layout: " << b;
    }

    void build_dpas(const bmnk_block_mapper_t &from_bmnk_mapper,
            const dpas_t &dpas, const multiply_desc_t &desc) {
        int m_blk = dpas.exec_size;
        int n_blk = dpas.rcount;
        int k_blk = dpas.sdepth * 4 / dpas.src1_type.size();

        c_layout_ = compute_dpas_c_layout(m_blk, n_blk, dpas.c_layout(), desc);

        expr_t a_buf = a_buf_;
        expr_t b_buf = b_buf_;
        if (do_transpose_) std::swap(a_buf, b_buf);
        auto dpas_tail = dpas_t::make(/*is_dpasw=*/false, dpas.exec_size,
                dpas.sdepth, desc.n() > n_blk ? desc.n() % n_blk : n_blk,
                dpas.dst_type, dpas.src1_type, dpas.src2_type);

        for (int i_k = 0; i_k < desc.k(); i_k += k_blk) {
            for (int i_m = 0; i_m < desc.m(); i_m += m_blk) {
                for (int i_n = 0; i_n < desc.n(); i_n += n_blk) {
                    std::vector<int> a_args = {i_m, i_k};
                    std::vector<int> b_args = {i_k, i_n};
                    std::vector<int> c_args = {i_m, i_n};
                    auto a = a_buf[desc.a_layout()(a_args)
                            * desc.a_type().size()];
                    auto b = b_buf[desc.b_layout()(b_args)
                            * desc.b_type().size()];
                    auto c = c_buf_[c_layout_(c_args) * desc.c_type().size()];
                    auto &_dpas = (i_n + n_blk > desc.n())
                            ? dpas_tail.as<dpas_t>()
                            : dpas;
                    stmt_ = stmt_.append(_dpas(c, c, a, b));
                }
            }
        }

        // Transpose C layout back if needed.
        if (do_transpose_) c_layout_ = c_layout_.transpose();

        // Convert C layout back to problem notation.
        c_layout_ = from_bmnk_mapper.map_from_bmnk(
                abc_kind_t::c, {bmnk_kind_t::m, bmnk_kind_t::n}, c_layout_);
    }

    static layout_t compute_dpas_c_layout(int m_blk, int n_blk,
            const layout_t &blk_layout, const multiply_desc_t &desc) {
        auto c_layout = blk_layout;
        auto new_blocks = c_layout.blocks();
        if (new_blocks.size() > 1) new_blocks[1].block = desc.n();
        c_layout = layout_t(c_layout.type(), c_layout.ndims(),
                c_layout.offset(), new_blocks);
        c_layout = c_layout.add_outer_block(0, desc.m() / m_blk);
        return c_layout;
    }

    bool try_build_mad() {
        auto loops = create_loop_nest();

        if (try_build_mad_kmn_block_by_n(loops)) return true;
        if (try_build_mad_kmn_block_by_b(loops)) return true;

        return false;
    }

    std::vector<loop_info_t> create_loop_nest() const {
        object_map_t<expr_t, loop_info_t> loops;
        for (auto *view : {&a_view_, &b_view_}) {
            abc_kind_t abc_kind
                    = (view == &a_view_ ? abc_kind_t::a : abc_kind_t::b);
            for (int i = 0; i < view->nvdims(); i++) {
                auto &var = bmnk_mapper_.var(abc_kind, i);
                int dim = int(view->vdims()[i]);
                if (dim == 1) continue;

                if (loops.count(var) > 0) continue;
                loops[var] = loop_info_t(var, bmnk_mapper_.bmnk_kind(var), dim);
            }
        }

        std::vector<loop_info_t> ret;
        for (auto &kv : loops) {
            auto &loop = kv.second;
            loop.a_idx = bmnk_mapper_.dim_idx(abc_kind_t::a, loop.var);
            loop.b_idx = bmnk_mapper_.dim_idx(abc_kind_t::b, loop.var);
            loop.c_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loop.var);
            ret.push_back(kv.second);
        }
        return ret;
    }

    // Order of loops: BKMN, block by N.
    bool try_build_mad_kmn_block_by_n(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::n);
    }

    // Order of loops: BKMN, block by B.
    bool try_build_mad_kmn_block_by_b(std::vector<loop_info_t> &_loops) {
        return try_build_mad_impl(_loops,
                {bmnk_kind_t::b, bmnk_kind_t::k, bmnk_kind_t::m,
                        bmnk_kind_t::n},
                bmnk_kind_t::b);
    }

    bool try_build_mad_impl(std::vector<loop_info_t> &_loops,
            const std::vector<bmnk_kind_t> &loop_order,
            bmnk_kind_t block_bmnk_kind) {
        auto loops = _loops;
        int nloops = int(loops.size());
        std::sort(loops.begin(), loops.end(),
                [&](const loop_info_t &a, const loop_info_t &b) {
                    int a_key = ir_utils::find_index(loop_order, a.bmnk_kind);
                    int b_key = ir_utils::find_index(loop_order, b.bmnk_kind);
                    ir_assert(a_key != -1);
                    ir_assert(b_key != -1);
                    return a_key < b_key;
                });

        int block_idx = -1;
        for (int i = 0; i < nloops; i++) {
            auto &l = loops[i];
            if (l.bmnk_kind == block_bmnk_kind) {
                ir_assert(block_idx == -1) << "Can't block 2+ dimensions.";
                block_idx = i;
            }
        }

        // Couldn't find N dimension, try different blocking scheme.
        if (block_idx == -1) return false;

        auto &block_loop = loops[block_idx];

        int block = simd_size_;
        while (block >= 1) {
            if (block_loop.dim % block == 0) break;
            block /= 2;
        }

        ir_assert(block >= 1) << "Invalid block size.";
        block_loop.block = block;

        int a_stride = 0;
        int b_stride = 0;

        // Ensure that A tile is dense.
        if (block_loop.a_idx != -1) {
            std::vector<dim_t> tile_dims(a_view_.nvdims(), 1);
            tile_dims[block_loop.a_idx] = block;
            auto layout = a_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            a_stride = tile.blocks()[0].stride;
        }

        // Ensure that B tile is dense.
        if (block_loop.b_idx != -1) {
            std::vector<dim_t> tile_dims(b_view_.nvdims(), 1);
            tile_dims[block_loop.b_idx] = block;
            auto layout = b_view_.create_pseudo_vlayout();
            auto tile = layout.map(tensor_t(tile_dims));
            if (!is_1d_strided(tile)) return false;
            b_stride = tile.blocks()[0].stride;
        }

        build_mad(loops, block_loop, a_stride, b_stride);
        return true;
    }

    static bool is_1d_strided(const layout_t &layout) {
        auto &blocks = layout.blocks();
        if (blocks.size() > 1) return false;
        return true;
    }

    void build_mad(const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop, int a_stride, int b_stride) {
        ir_assert(utils::one_of(
                block_loop.bmnk_kind, bmnk_kind_t::b, bmnk_kind_t::n))
                << "Unsupported blocking (expected blocking by B or N).";

        auto &a_type = a_view_.type();
        auto &b_type = b_view_.type();
        auto c_type = multiply_desc_t::get_c_type(a_type, b_type,
                /*force_c_upconvert=*/false);

        int block = block_loop.block;
        auto _mad = mad_t::make(
                hw_, c_type, block, a_type, a_stride, b_type, b_stride);
        auto &mad = _mad.as<mad_t>();

        c_layout_ = compute_mad_c_layout(c_type, loops, block_loop);

        int nloops = int(loops.size());
        std::vector<int> bounds(loops.size());
        for (int i = 0; i < nloops; i++) {
            bounds[i] = loops[i].dim / loops[i].block;
        }
        std::vector<int> a_idx(a_view_.nvdims());
        std::vector<int> b_idx(b_view_.nvdims());
        std::vector<int> c_idx(c_layout_.ndims());
        ir_utils::for_each(bounds, [&](const std::vector<int> &idx) {
            for (int i = 0; i < nloops; i++) {
                int full_idx = idx[i] * loops[i].block;
                auto &loop = loops[i];
                if (loop.a_idx != -1) a_idx[loop.a_idx] = full_idx;
                if (loop.b_idx != -1) b_idx[loop.b_idx] = full_idx;
                if (loop.c_idx != -1) c_idx[loop.c_idx] = full_idx;
            }
            int a_off = a_view_(a_idx) * a_type.size();
            int b_off = b_view_(b_idx) * b_type.size();
            int c_off = c_layout_(c_idx) * c_type.size();
            stmt_ = stmt_.append(mad(c_buf_[c_off], c_buf_[c_off],
                    a_buf_[a_off], b_buf_[b_off]));
            // XXX: Workaround for fp64 correctness issues with mad.
            if (c_type.is_f64()) stmt_ = stmt_.append(funcs::swsb_long_sync());
        });
    }

    layout_t compute_mad_c_layout(const type_t &c_type,
            const std::vector<loop_info_t> &loops,
            const loop_info_t &block_loop) const {
        layout_t c_layout(c_type, bmnk_mapper_.ndims(abc_kind_t::c), 0, {});

        int c_dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, block_loop.var);
        c_layout = c_layout.add_outer_block(c_dim_idx, block_loop.block);

        for (size_t i = 0; i < loops.size(); i++) {
            if (loops[i].bmnk_kind == bmnk_kind_t::k) continue;
            int dim_idx = bmnk_mapper_.dim_idx(abc_kind_t::c, loops[i].var);
            int bound = loops[i].dim / loops[i].block;
            c_layout = c_layout.add_outer_block(dim_idx, bound);
        }
        return c_layout;
    }

    ngen::HW hw_;
    int simd_size_;
    bmnk_mapper_t bmnk_mapper_;

    bool do_transpose_ = false;

    view_t a_view_;
    view_t b_view_;
    layout_t c_layout_;

    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    stmt_t stmt_;
};

class fma_helper_t {
public:
    fma_helper_t(int simd_size, fma_kind_t fma_kind, const type_t &a_type,
            const type_t &b_type, bool allow_a_grf_reorder,
            bool allow_b_grf_reorder, bool is_src1_broadcast)
        : simd_size_(simd_size)
        , fma_kind_(fma_kind)
        , a_type_(a_type)
        , b_type_(b_type)
        , allow_a_grf_reorder_(allow_a_grf_reorder)
        , allow_b_grf_reorder_(allow_b_grf_reorder)
        , is_src1_broadcast_(is_src1_broadcast) {}

    fma_kind_t fma_kind() const { return fma_kind_; }

    layout_t convert_to_fma_friendly_layout(const layout_t &layout,
            abc_kind_t abc_kind, bool is_slm, const bmnk_mapper_t &bmnk_mapper,
            bool *changed = nullptr) const {
        bool allow_grf_reorder
                = (abc_kind == abc_kind_t::a ? allow_a_grf_reorder_
                                             : allow_b_grf_reorder_);
        if (changed) *changed = false;
        if (!allow_grf_reorder) return layout;

        // GRF reorder is only supported with dpas/dpasw.
        if (fma_kind_ == fma_kind_t::mad) {
            if (is_slm) return layout;
            // mad may require type conversion, supported for GRF layouts only.
            return convert_to_fma_friendly_type(layout, abc_kind, changed);
        }

        std::vector<bmnk_kind_t> bmnk_kinds;
        if (abc_kind == abc_kind_t::a) {
            bmnk_kinds.push_back(bmnk_kind_t::m);
            bmnk_kinds.push_back(bmnk_kind_t::k);
        } else {
            bmnk_kinds.push_back(bmnk_kind_t::k);
            bmnk_kinds.push_back(bmnk_kind_t::n);
        }

        auto bmnk_layout
                = bmnk_mapper.map_to_bmnk(abc_kind, bmnk_kinds, layout);

        auto dpas_layout = get_dpas_friendly_layout(bmnk_layout, abc_kind);
        if (dpas_layout == bmnk_layout) return layout;

        if (changed) *changed = true;

        bmnk_block_mapper_t from_bmnk_mapper(bmnk_mapper);
        from_bmnk_mapper.push_blocks(abc_kind, layout.blocks());

        auto fma_layout = from_bmnk_mapper.map_from_bmnk(
                abc_kind, bmnk_kinds, dpas_layout);
        fma_layout = fma_layout.make_dense();
        return fma_layout;
    }

private:
    layout_t convert_to_fma_friendly_type(const layout_t &layout,
            abc_kind_t abc_kind, bool *changed = nullptr) const {
        if (changed) *changed = false;
        if (fma_kind_ != fma_kind_t::mad) return layout;

        // mad with s8/u8 is not supported, promote to strided s16.
        if (a_type_.is_x8() && b_type_.is_x8()) {
            if (changed) *changed = true;
            return layout.retype(type_t::s16()).make_strided(2);
        }

        // bf16 mixed mode mad requires src2 to be f32.
        if (abc_kind == abc_kind_t::b && a_type_.is_bf16()) {
            if (changed) *changed = true;
            return layout.retype(type_t::f32()).make_dense();
        }

        // bf16 mixed mode mad requires src1 to be packed, when src1 is
        // broadcasted it needs to be converted to f32.
        if (abc_kind == abc_kind_t::a && a_type_.is_bf16()
                && is_src1_broadcast_) {
            if (changed) *changed = true;
            return layout.retype(type_t::f32()).make_dense();
        }

        // Ensure the layout is dense to align regioning.
        if (!layout.is_dense()) {
            if (changed) *changed = true;
            return layout.make_dense();
        }

        return layout;
    }

    layout_t get_dpas_friendly_layout(
            const layout_t &bmnk_layout, abc_kind_t abc_kind) const {
        bool is_a = (abc_kind == abc_kind_t::a);
        int mn_idx = (is_a ? 0 : 1);
        int k_idx = (is_a ? 1 : 0);

        dim_t mn_blk = bmnk_layout.dim(mn_idx);
        dim_t k_blk = bmnk_layout.dim(k_idx);

        // Cannot calculate correct r_count when !is_a, but rcount is effectively
        // ignored in that case as rcount mainly affects b_layout.
        // Also note that rcount used here may not be supported in hardware and is used soley to compute layout.
        int rcount = is_a ? mn_blk : 8;
        auto _dpas = dpas_t::make(/*is_dpasw=*/false, simd_size_, /*sdepth=*/8,
                rcount, type_t::undef(), b_type_, a_type_);
        auto &dpas = _dpas.as<dpas_t>();

        auto dpas_layout = (is_a ? dpas.b_layout() : dpas.a_layout());
        dpas_layout = dpas_layout.transpose();

        auto default_layout = bmnk_layout.retype(is_a ? a_type_ : b_type_);
        if (dpas_layout <= default_layout) return default_layout;

        dim_t dpas_mn_blk = dpas_layout.dim(mn_idx);
        dim_t dpas_k_blk = dpas_layout.dim(k_idx);
        ir_assert(k_blk % dpas_k_blk == 0);

        dim_t k_outer = ir_utils::safe_divide(k_blk, dpas_k_blk);
        dim_t mn_outer = ir_utils::safe_divide(mn_blk, dpas_mn_blk);
        dpas_layout = dpas_layout.add_outer_block(k_idx, k_outer);
        dpas_layout = dpas_layout.add_outer_block(mn_idx, mn_outer);
        return dpas_layout;
    }

    int simd_size_;
    fma_kind_t fma_kind_;
    type_t a_type_;
    type_t b_type_;
    bool allow_a_grf_reorder_;
    bool allow_b_grf_reorder_;
    bool is_src1_broadcast_;
};

class b_reduce_context_t {
public:
    b_reduce_context_t(ir_context_t &ir_ctx, const conv_config_t &cfg)
        : ir_ctx_(ir_ctx), cfg_(cfg), reduce_condition_(true) {
        if (cfg.do_b_reduction) b_reduced_reg_buf_ = make_buffer("b_reduced");
    }

    // Setters for B reduced memory buffer/view.
    void set_b_reduced_mem_buf(const expr_t &buf) { b_reduced_mem_buf_ = buf; }
    void set_b_reduced_view(const view_t &v) { b_reduced_view_ = v; }

    // Sets the condition to update B reduced output. Reduction is done across
    // K for B (KxN tensor) so M dimension should be checked before the update.
    void set_reduce_condition(const expr_t &cond) { reduce_condition_ = cond; }

    // Global memory buffer.
    const expr_t &b_reduced_mem_buf() const { return b_reduced_mem_buf_; }

    // Register buffer.
    const expr_t &b_reduced_reg_buf() const { return b_reduced_reg_buf_; }
    int b_reduced_size() const { return b_reduced_size_; }

    // Memory view.
    const view_t &b_reduced_thr_view() const { return b_reduced_thr_view_; }

    // Register layout.
    const layout_t &b_reduced_reg_layout() const {
        return b_reduced_reg_layout_;
    }

    void init_reduced_thr_view(
            const tensor_t &b_thr_tile, const expr_t &cond = expr_t()) {
        ir_assert(b_reduced_thr_view_.is_empty()) << "Can't initialize twice.";

        auto b_reduced_thr_tile = b_to_b_reduced_tile(b_thr_tile);
        b_reduced_thr_view_
                = b_reduced_view_.create_sub_view(b_reduced_thr_tile);
        b_reduced_reg_layout_ = b_reduced_thr_view_.create_dense_vlayout();
        b_reduced_size_ = b_reduced_reg_layout_.size();
        b_reduced_size_ = utils::rnd_up(b_reduced_size_, cfg_.grf_size());

        if (!cond.is_empty()) reduce_condition_ &= cond;
    }

    stmt_t create_reduce_stmt(const layout_t &b_layout, const expr_t &b_buf,
            const tensor_t &sub_tile = tensor_t()) {
        auto reduction_stmt
                = jit::create_reduce_stmt(b_layout, b_reduced_reg_layout_,
                        b_buf, b_reduced_reg_buf_, sub_tile, reduction_mask_);
        return reduction_stmt;
    }

    stmt_t create_store_stmt() const {
        auto r2g = make_access_builder(ir_ctx_, b_reduced_thr_view_,
                b_reduced_mem_buf_, b_reduced_reg_buf_, send_op_t::atomic_fadd,
                send_address_t::a64);
        // TODO: Check that layouts match.
        auto ret = r2g.stmt();
        if (!reduce_condition_.is_empty()) {
            ret = if_t::make(reduce_condition_, ret);
        }
        return ret;
    }

private:
    tensor_t b_to_b_reduced_tile(const tensor_t &b_tile) const {
        std::vector<dim_t> dims;
        std::vector<expr_t> start;
        for (int i = 0; i < b_tile.ndims(); i++) {
            if ((reduction_mask_ & (1 << i)) != 0) {
                dims.push_back(b_tile(i));
                start.push_back(b_tile.start(i));
            }
        }
        return tensor_t(dims, start);
    }

    ir_context_t &ir_ctx_;
    const conv_config_t &cfg_;

    expr_t reduce_condition_;

    expr_t b_reduced_mem_buf_;
    expr_t b_reduced_reg_buf_;

    view_t b_reduced_view_;
    view_t b_reduced_thr_view_;

    layout_t b_reduced_reg_layout_;
    int b_reduced_size_ = 0;

    uint32_t reduction_mask_ = (1 << 1) | (1 << 2);
};

class sub_tile_info_t {
public:
    using post_load_func_t = std::function<stmt_t(
            const layout_t &, const expr_t &, const tensor_t &)>;

    sub_tile_info_t(ir_context_t &ir_ctx, const gemm_schedule_t &gemm_schedule,
            const fma_helper_t &fma_helper, abc_kind_t abc_kind, bool use_slm,
            bool load_buffered, bool allow_2d_load, int idx,
            const view_t &mem_view, const tensor_t &sub_tile,
            const expr_t &mem_buf, const expr_t &slm_buf, const expr_t &reg_buf,
            const expr_t &tmp_buf)
        : ir_ctx_(ir_ctx)
        , gemm_schedule_(gemm_schedule)
        , fma_helper_(fma_helper)
        , abc_kind_(abc_kind)
        , use_slm_(use_slm)
        , load_buffered_(load_buffered)
        , allow_2d_load_(allow_2d_load)
        , idx_(idx)
        , mem_view_(mem_view)
        , sub_tile_(sub_tile)
        , mem_buf_(mem_buf)
        , slm_buf_(slm_buf)
        , reg_buf_(reg_buf)
        , tmp_buf_(tmp_buf) {}

    bool is_loaded() const { return is_loaded_; }

    void set_loaded() { is_loaded_ = true; }

    const view_t &reg_view() const { return reg_view_; }

    int reg_buf_size() const {
        return utils::rnd_up(reg_layout_.size(), ir_ctx_.hw_cfg().grf_size());
    }

    int tmp_buf_size() const { return tmp_buf_size_; }

    const stmt_t &s2r_load() const { return s2r_load_; }

    const stmt_t &g2r_load() const { return g2r_load_; }

    const send_hint_t &send_hint() const { return send_hint_; }

    void load(const post_load_func_t &post_load = post_load_func_t()) {
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();

        layout_t load_layout;
        stmt_t &stmt = (use_slm_ ? s2r_load_ : g2r_load_);
        load_impl(ir_ctx_, load_layout, reg_view_, send_hint_, stmt);

        if (post_load) {
            stmt = stmt.append(post_load(load_layout, reg_buf_, sub_tile_));
        }

        reg_layout_ = load_layout;

        bool changed;
        auto fma_layout = fma_helper_.convert_to_fma_friendly_layout(
                reg_layout_, abc_kind_,
                /*is_slm=*/false, bmnk_mapper, &changed);

        if (changed) {
            bool is_reorder_nop
                    = fma_layout.retype(reg_layout_.type()) == reg_layout_
                    && reg_layout_.type().is_bitwise_compatible(
                            fma_layout.type());

            if (fma_layout.type() != reg_layout_.type()) {
                reg_view_ = reg_view_.retype(fma_layout.type());
            }
            reg_layout_ = fma_layout;
            reg_view_.set_tlayout(reg_layout_);
            if (!is_reorder_nop) {
                stmt = substitute(stmt, reg_buf_, tmp_buf_);
                stmt = stmt.append(create_reorder_stmt(
                        load_layout, reg_layout_, tmp_buf_, reg_buf_));
                int load_reg_size = int(load_layout.size());
                load_reg_size = utils::rnd_up(
                        load_reg_size, ir_ctx_.hw_cfg().grf_size());
                tmp_buf_size_ = std::max(tmp_buf_size_, load_reg_size);
            }
        }
    }

private:
    void load_impl(ir_context_t &ir_ctx, layout_t &load_layout,
            view_t &load_view, send_hint_t &send_hint, stmt_t &stmt) const {
        view_t mem_view = mem_view_;
        if (load_buffered_)
            mem_view_.try_create_buffer_view(mem_view, load_view);

        send_op_t send_op = send_op_t::load;
        send_hint = get_send_hint(ir_ctx_.hw_cfg(), send_op_t::load,
                fma_helper_.fma_kind(), abc_kind_, mem_view, gemm_schedule_,
                allow_2d_load_);
        auto read = make_access_builder(ir_ctx, mem_view,
                use_slm_ ? slm_buf_ : mem_buf_, reg_buf_, send_op,
                use_slm_ ? send_address_t::slm : send_address_t::a64,
                send_hint);
        ir_trace() << (abc_kind_ == abc_kind_t::a ? "A" : "B")
                   << " GMEM/SLM to GRF load #" << idx_ << ":\n"
                   << read.str() << std::endl;

        load_layout = read.reg_layout();
        if (!load_view.is_empty()) {
            load_view.set_tlayout(load_layout);
        } else {
            load_view = view_t(load_layout);
        }
        stmt = read.stmt();
    }

    ir_context_t &ir_ctx_;
    const gemm_schedule_t &gemm_schedule_;
    const fma_helper_t &fma_helper_;
    abc_kind_t abc_kind_;
    bool use_slm_;
    bool load_buffered_;
    bool allow_2d_load_;
    int idx_;
    view_t mem_view_;
    tensor_t sub_tile_;

    expr_t mem_buf_;
    expr_t slm_buf_;
    expr_t reg_buf_;
    expr_t tmp_buf_;

    bool is_loaded_ = false;
    view_t reg_view_;
    layout_t reg_layout_;
    int tmp_buf_size_ = 0;
    stmt_t s2r_load_;
    stmt_t g2r_load_;
    send_hint_t send_hint_;
};

class load_multiply_builder_t {
public:
    load_multiply_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const gemm_schedule_t &gemm_schedule,
            const fma_helper_t &fma_helper, b_reduce_context_t &b_reduce_ctx,
            const expr_t &ap_buf, const expr_t &a_slm_buf, const expr_t &bp_buf,
            const expr_t &b_slm_buf, const view_t &ap_x_view,
            const view_t &bp_x_view, const kernel_info_t &kernel_info)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , gemm_schedule_(gemm_schedule)
        , fma_helper_(fma_helper)
        , b_reduce_ctx_(b_reduce_ctx)
        , ap_buf_(ap_buf)
        , a_slm_buf_(a_slm_buf)
        , bp_buf_(bp_buf)
        , b_slm_buf_(b_slm_buf)
        , kernel_info_(kernel_info) {
        ir_assert(cfg_.a_sub_tiles == 1 || cfg_.b_sub_tiles == 1)
                << "At most one tensor can be tiled.";

        ab_tmp_buf_ = make_buffer("ab_tmp");
        a_buf_ = make_buffer("a");
        b_buf_ = make_buffer("b");
        c_buf_ = make_buffer("c");

        // Views to multiply by a thread.
        a_thr_view_ = ap_x_view.create_sub_view(gemm_schedule_.a_thr_tile());
        b_thr_view_ = bp_x_view.create_sub_view(gemm_schedule_.b_thr_tile());

        // Initialize view for reduced B.
        if (cfg_.do_b_reduction && !cfg_.use_b_slm) {
            b_reduce_ctx_.init_reduced_thr_view(
                    gemm_schedule_.b_thr_tile(/*is_relative=*/false));
        }

        // TODO: Specify loops over sub-tiles in the schedule, use unrolling.
        // Sub-tile indices.
        a_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "a_idx");
        b_idx_ = ir_ctx_.create_tmp_var(type_t::s32(), "b_idx");

        // Sub-tile views.
        a_i_view_ = create_sub_tile_view(abc_kind_t::a, a_thr_view_,
                cfg_.a_sub_tiles, a_idx_, bmnk_kind_t::m, &a_i_outer_blocks_,
                a_i_tile_);
        b_j_view_ = create_sub_tile_view(abc_kind_t::b, b_thr_view_,
                cfg_.b_sub_tiles, b_idx_, bmnk_kind_t::n, &b_j_outer_blocks_,
                b_j_tile_);

        build();
    }

    const std::vector<stmt_t> &allocs() const { return allocs_; }

    const stmt_t &load_mul_stmt() const { return load_mul_stmt_; }

    const expr_t &c_buf() const { return c_buf_; }

    const layout_t &c_reg_layout() const { return c_reg_layout_; }

private:
    view_t create_sub_tile_view(abc_kind_t abc_kind, const view_t &thr_view,
            int sub_tiles, const expr_t &idx, bmnk_kind_t bmnk_kind,
            std::vector<block_t> *outer_blocks, tensor_t &sub_tile) const {
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        auto layout = thr_view.create_pseudo_vlayout();
        dim_t mn_dim = 1;
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) mn_dim *= b.block;
        }

        std::vector<dim_t> sub_tile_dims(thr_view.nvdims(), 1);
        dim_t mn_sub_tile_dim = ir_utils::safe_divide(mn_dim, dim_t(sub_tiles));
        for (auto &b : layout.blocks()) {
            auto b_bmnk_kind = bmnk_mapper.bmnk_kind(abc_kind, b.dim_idx);
            if (b_bmnk_kind == bmnk_kind) {
                if (mn_sub_tile_dim == 1) continue;
                dim_t next_block;
                if (mn_sub_tile_dim % b.block == 0) {
                    next_block = b.block;
                } else {
                    ir_assert(b.block % mn_sub_tile_dim == 0);
                    next_block = mn_sub_tile_dim;
                }
                sub_tile_dims[b.dim_idx] *= next_block;
                mn_sub_tile_dim /= next_block;
            } else {
                sub_tile_dims[b.dim_idx] *= b.block;
            }
        }
        grid_info_t grid({sub_tiles}, {idx});
        sub_tile = layout.split(tensor_t(sub_tile_dims), grid, outer_blocks);
        return thr_view.create_sub_view(sub_tile);
    }

    void build() {
        int max_iters = 2;
        bool load_ok = false;
        for (int iter = 0; iter < max_iters; iter++) {
            if (try_load_sub_tiles(/*allow_2d_load=*/iter == 0)) {
                load_ok = true;
                break;
            }
        }
        ir_assert(load_ok) << "Can't generate load statements for sub-tiles.";

        for (int i = 0; i < cfg_.a_sub_tiles; i++) {
            for (int j = 0; j < cfg_.b_sub_tiles; j++) {
                build_sub_tile(i, j);
            }
        }

        if (zp_buf_size_ > 0)
            register_buffer(zp_buf_, zp_buf_size_, alloc_kind_t::grf);

        // Handle temporary buffer in case of GRF reorders.
        int tmp_buf_size = 0;
        for (int i = 0; i < cfg_.a_sub_tiles; i++)
            tmp_buf_size
                    = std::max(tmp_buf_size, a_sub_tiles_[i].tmp_buf_size());
        for (int j = 0; j < cfg_.b_sub_tiles; j++)
            tmp_buf_size
                    = std::max(tmp_buf_size, b_sub_tiles_[j].tmp_buf_size());
        if (tmp_buf_size > 0)
            register_buffer(ab_tmp_buf_, tmp_buf_size, alloc_kind_t::grf);

        // C layout in problem notation.
        auto c_layout = c_sub_tile_layout_;

        // Add outer blocks coming from A/B sub-tiles.
        auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
        for (auto &b : a_i_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::a, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }
        for (auto &b : b_j_outer_blocks_) {
            auto &var = bmnk_mapper.var(abc_kind_t::b, b.dim_idx);
            int c_dim_idx = bmnk_mapper.dim_idx(abc_kind_t::c, var);
            c_layout = c_layout.add_outer_block(c_dim_idx, b.block);
        }

        c_reg_layout_ = c_layout;
    }

    bool can_use_2d_load(const abc_kind_t &abc_kind, const view_t &view) const {
        bool is_blocked = view.tlayout().innermost_block_layout().elems() > 1;
        if (!is_blocked) return true;

        // In general we want to skip expensive logic to check requirements for
        // 2D block messages with block layouts as performance with 1D messages
        // is good enough. However there are a few cases (backward by weights
        // with dpas) when 2D block messages give boost even for block layouts
        // due to VNNI/transpose features.
        if (cfg_.is_bwd_w && cfg_.is_dp_fma()) {
            auto &bmnk_mapper = gemm_schedule_.bmnk_mapper();
            auto &blocks = view.tlayout().blocks();
            if (blocks.size() < 2) return false;
            int b1_dim_idx = blocks[1].dim_idx;
            return bmnk_mapper.bmnk_kind(abc_kind, b1_dim_idx)
                    == bmnk_kind_t::k;
        }
        return false;
    }

    bool try_load_sub_tiles(bool allow_2d_load) {
        a_sub_tiles_.clear();
        b_sub_tiles_.clear();
        for (int i = 0; i < cfg_.a_sub_tiles; i++) {
            auto view = a_i_view_.substitute(a_idx_, i);
            auto tile = a_i_tile_.substitute(a_idx_, i);
            // Using buffered view is enabled only when:
            // - Loading directly from global memory
            // - FMA kind is mad (dpas implementation is more strict and requires
            //   layouts, not views)
            // - Loading A tensor (A - activations for FWD/BWD_D where we may have
            //   overlapping when applying KW blocking )
            bool load_buffered = cfg_.use_ow_kw_grf_cache && !cfg_.use_a_slm
                    && cfg_.fma_kind == fma_kind_t::mad;
            a_sub_tiles_.emplace_back(ir_ctx_, gemm_schedule_, fma_helper_,
                    abc_kind_t::a, cfg_.use_a_slm, load_buffered,
                    allow_2d_load && can_use_2d_load(abc_kind_t::a, a_i_view_),
                    i, view, tile, ap_buf_, a_slm_buf_, a_buf_, ab_tmp_buf_);
            a_sub_tiles_.back().load();
        }
        sub_tile_info_t::post_load_func_t b_post_load;
        if (!cfg_.use_b_slm && cfg_.do_b_reduction) {
            b_post_load = [&](const layout_t &reg_layout, const expr_t &reg_buf,
                                  const tensor_t &tile) {
                return b_reduce_ctx_.create_reduce_stmt(
                        reg_layout, reg_buf, tile);
            };
        }
        for (int j = 0; j < cfg_.b_sub_tiles; j++) {
            auto view = b_j_view_.substitute(b_idx_, j);
            auto tile = b_j_tile_.substitute(b_idx_, j);
            b_sub_tiles_.emplace_back(ir_ctx_, gemm_schedule_, fma_helper_,
                    abc_kind_t::b, cfg_.use_b_slm,
                    /*load_buffered=*/false,
                    allow_2d_load && can_use_2d_load(abc_kind_t::b, b_j_view_),
                    j, view, tile, bp_buf_, b_slm_buf_, b_buf_, ab_tmp_buf_);

            b_sub_tiles_.back().load(b_post_load);
        }

        // Validate sub-tile loads, when VNNI permutation is applied, both A/B
        // have to use the same pattern.
        int vnni_permute_factor
                = a_sub_tiles_[0].send_hint().hint_2d.vnni_permute_factor;
        for (int i = 1; i < cfg_.a_sub_tiles; i++) {
            int f = a_sub_tiles_[i].send_hint().hint_2d.vnni_permute_factor;
            if (f != vnni_permute_factor) return false;
        }
        for (int j = 0; j < cfg_.b_sub_tiles; j++) {
            int f = b_sub_tiles_[j].send_hint().hint_2d.vnni_permute_factor;
            if (f != vnni_permute_factor) return false;
        }
        return true;
    }

    class src_zp_mask_info_t {
    public:
        src_zp_mask_info_t() = delete;
        src_zp_mask_info_t(load_multiply_builder_t &lmb, int m_blk, int k_blk,
                int desc_m, int desc_n, int channels, int a_stride, bool is_mad,
                const view_t &a_view)
            : lmb_(lmb)
            , is_const_(true)
            , is_simd_(true)
            , is_scalar_(false)
            , is_wide_(m_blk < 16) {
            const auto tile
                    = lmb_.gemm_schedule_.a_thr_tile(/*is_relative=*/false);
            const auto a_thr_view
                    = lmb_.gemm_schedule_.a_view().create_sub_view(tile);
            const auto ic_dim = (!is_mad) ? 2 : 1;
            ic_start_ = a_thr_view.vstart(ic_dim);

            // 0. Are the masks at all required?
            const auto &cfg = lmb_.cfg_;
            const auto dims = tile.dims()[3] * tile.dims()[4] * tile.dims()[5];
            const auto is_scalar = !is_mad && (dims <= 1);

            const auto has_pad = (cfg.pd + 1) * (cfg.ph + 1) * (cfg.pw + 1) > 1;
            const auto has_stride_bd
                    = cfg.is_bwd_d && (cfg.sd * cfg.sh * cfg.sw > 1);

            // 1. Get the raw representation of the buffer`s masks
            auto mask_tensor
                    = a_thr_view.create_mask_tensor(lmb_.ir_ctx_.cset());

            // 2. Collect the masks, transforming the dimensions as needed
            int channels_blk = std::min(channels,
                    (int)a_thr_view.tlayout().normalize().blocks()[0].block);
            if (channels_blk > 32) channels_blk = 32;
            const auto c_blk = std::min(channels_blk, m_blk);
            auto a_tdims = a_view.tlayout().dims();
            auto mask_blk = is_mad ? c_blk : channels_blk;
            size_ = ((cfg.kd * cfg.kh * cfg.kw > 1) || has_pad || has_stride_bd)
                    * ((!is_scalar) ? accumulate(a_tdims.begin(), a_tdims.end(),
                                              1, std::multiplies<dim_t>())
                                            / mask_blk
                                    : 1);
            if (size_ == 0) return;

            mask_tensor_t masks(
                    layout_t(type_t::_bool(), 0, std::vector<dim_t> {size_}));
            std::vector<dim_t> a_dims(a_view.tlayout().ndims(), 1);
            a_dims[ic_dim] = mask_blk;
            int i = 0;
            a_view.tlayout().for_each_tile(
                    tensor_t(a_dims), [&](const std::vector<dim_t> &start) {
                        std::vector<dim_t> a_st(a_thr_view.nvdims(), 0);
                        for (int idx = 0; idx < (int)start.size(); idx++) {
                            auto tdim_to_vdims = [&](int idx) {
                                auto tdim = a_view.tdim(idx);
                                auto vidx0 = tdim.vidx(0);
                                auto vidx1 = -1;
                                int vdim1 = 0;
                                ir_assert(tdim.nvargs() <= 2);
                                for (int vdim0 = 0;
                                        vdim0 < a_thr_view.vdims()[vidx0];
                                        vdim0++) {
                                    auto tdim_expr = substitute(tdim.expr(),
                                            a_view.vvars()[vidx0],
                                            to_expr(vdim0));
                                    if (tdim.nvargs() == 2) {
                                        vidx1 = tdim.vidx(1);
                                        for (vdim1 = 0; vdim1
                                                < a_thr_view.vdims()[vidx1];
                                                vdim1++) {
                                            auto tdim_expr2 = substitute(
                                                    tdim_expr,
                                                    a_view.vvars()[vidx1],
                                                    to_expr(vdim1));
                                            if (to_cpp<dim_t>(
                                                        simplify(tdim_expr2))
                                                    == start[idx]) {
                                                a_st[vidx1] = vdim1;
                                                a_st[vidx0] = vdim0;
                                                return;
                                            }
                                        }
                                        tdim_expr = substitute(tdim_expr,
                                                a_view.vvars()[vidx1],
                                                to_expr(0));
                                    }
                                    if (to_cpp<dim_t>(simplify(tdim_expr))
                                            == start[idx]) {
                                        a_st[vidx0] = vdim0;
                                        break;
                                    }
                                }
                            };
                            tdim_to_vdims(idx);
                        }
                        auto off = a_thr_view.create_pseudo_vlayout()
                                           .make_dense()
                                           .offset<dim_t>(a_st);
                        if (i >= size_) return;
                        masks.set_mask(i, mask_tensor.mask(off));
                        i++;
                    });

            // 3. Compute some basic properties of the masks just collected
            for (int n = 0; n < size_; n++) {
                auto *sh = masks.mask(n).as_ptr<shuffle_t>();
                is_simd_ &= !sh || sh->is_broadcast();
                is_const_ &= !!sh;
                for (int v = (sh) ? 0 : c_blk; v < c_blk; v++)
                    is_const_ &= sh->vec[sh->idx[v]].is<bool_imm_t>();
            }

            // 4. Scalarize if the masks permit, transform to shorts otherwise
            for (int n = 0; n < size_; n++)
                if (is_simd_) {
                    object_map_t<expr_t, std::vector<expr_t>> vars;
                    expr_scalarizer_t sc(c_blk, 0, vars);
                    masks.set_mask(n, sc.mutate(masks.mask(n)));
                } else if (is_const_) {
                    uint16_t mask = 0;
                    auto &sh = masks.mask(n).as<shuffle_t>();
                    for (int v = c_blk; v; v--)
                        mask = mask * 2
                                + sh.vec[sh.idx[v - 1]].as<bool_imm_t>().value;
                    masks.set_mask(n, mask);
                } else {
                    ir_error_not_expected() << "Non-SIMD non-constant masks!";
                }

            // 5. Assume lack of masks if they all are true
            bool all_true = true;
            for (int n = 0; all_true && (n < size_); n++)
                all_true &= masks.mask(n).is_equal(expr_t(true));
            if (all_true) {
                is_const_ = true;
                is_simd_ = true;
                size_ = 0;
                return;
            }
            is_scalar_ = is_scalar;

            // 6. The masks need to be created; allocate the buffers
            zp_mask_ = lmb_.ir_ctx_.create_tmp_var(
                    type_t::byte_ptr(), "zp_mask");
            var_mask_ = lmb_.ir_ctx_.create_tmp_var(
                    (!is_bool()) ? type_t::s16() : type_t::_bool(16));

            // 7. Vectorize everything for easier computation and emit the IR
            if (!is_scalar) {
                std::vector<expr_t> exprs;
                object_eq_map_t<expr_t, expr_t> vars;

                // Here we assume two important things:
                // - C has exactly one N block like 4c16f8c (where f is ow)
                // - The innermost block is by M and it matches the SIMD size

                std::vector<expr_t> proto(size_);
                if (is_wide_) {
                    for (int n = 0; n < int(proto.size()); n++) {
                        const auto r = (n / 2 / k_blk) * 2 * k_blk
                                + (n / 2) % k_blk + (n & 1) * k_blk;
                        proto[n] = masks.mask(r % size_);
                    }
                } else {
                    for (int n = 0; n < int(proto.size()); n++)
                        proto[n] = masks.mask(n % size_);
                }
                for (; size_ >= m_blk * 2; size_ /= 2) {
                    auto c = [](expr_t &a, expr_t &b) { return a.is_equal(b); };
                    auto half = proto.begin() + size_ / 2;
                    if (!std::equal(proto.begin(), half, half, c)) break;
                }

                const auto blk
                        = (size_ > m_blk) ? std::min(m_blk * 2, 16) : m_blk;
                for (int n = 0; n < size_; n += blk) {
                    std::vector<expr_t> e(blk);
                    for (int m = 0; m < blk; m++) {
                        e[m] = proto[n + m % (size_ - n)];
                    }
                    int ntrue = 0, nfalse = 0;
                    for (int m = 0; m < blk; m++) {
                        if (e[m].is<bool_imm_t>())
                            ((e[m].as<bool_imm_t>().value) ? ntrue : nfalse)++;
                    }
                    ir_assert((ntrue == 0) || (ntrue + nfalse == blk));
                    if ((ntrue == 0) && (nfalse > 0) && (nfalse < blk)) {
                        auto nb = *std::find_if(e.begin(), e.end(),
                                [](expr_t &x) { return !x.is<bool_imm_t>(); });
                        for (int m = 0; m < blk; m++) {
                            e[m] = (e[m].is<bool_imm_t>())
                                    ? (nb & expr_t(false))
                                    : (e[m] & expr_t(true));
                        }
                    }
                    exprs.emplace_back(vector2expr(e, vars));
                }

                const auto real_size = std::max(
                        size_ * ((is_wide_) ? 8 : 1), int(exprs.size()) * blk);
                lmb_.register_buffer(
                        zp_mask_, real_size * w_stride(), alloc_kind_t::grf);
                for (int i = 0; i < int(exprs.size()); i++) {
                    auto expr = cast_t::make(w_type(blk), exprs[i]);
                    stmt_ = stmt_.append(
                            store_t::make(zp_mask_, i * blk * w_stride(),
                                    (is_simd_) ? -expr : expr, w_stride()));
                }
                if (is_wide_) {
                    auto wide_scalar = [&](const expr_t &a, const expr_t &b) {
                        std::vector<int> idx(16, 1);
                        for (int i = 0; i < 8; i++)
                            idx[i] = 0;
                        return shuffle_t::make(std::vector<expr_t> {a, b}, idx);
                    };
                    for (int i = size_ - 2; i > 0; i -= 2) {
                        auto load_l = load_t::make(
                                type_t::s16(), zp_mask_, i * w_stride());
                        auto load_h = load_t::make(
                                type_t::s16(), zp_mask_, (i + 1) * w_stride());
                        auto load = wide_scalar(load_l, load_h);
                        load = cast_t::make(type_t::s16(16), load);
                        stmt_ = stmt_.append(store_t::make(zp_mask_,
                                i * 8 * w_stride(), load, w_stride()));
                    }
                    if (size_ % 2 == 0) {
                        auto l0h = load_t::make(
                                type_t::s16(), zp_mask_, w_stride());
                        stmt_ = stmt_.append(store_t::make(zp_mask_,
                                8 * w_stride(),
                                shuffle_t::make_broadcast(l0h, 8), w_stride()));
                    }
                    auto l0l = load_t::make(type_t::s16(), zp_mask_, 0);
                    stmt_ = stmt_.append(store_t::make(zp_mask_, 0,
                            shuffle_t::make_broadcast(l0l, 8), w_stride()));
                }

                for (auto &v : vars)
                    stmt_ = let_t::make(v.second, v.first, stmt_);
            } else { // is_scalar == true
                lmb_.register_buffer(
                        zp_mask_, type_t::s16().size(), alloc_kind_t::grf);
                auto expr = cast_t::make(type_t::s16(), masks.mask(0));
                if (is_simd_) expr = cast(-expr, type_t::s16());
                stmt_ = stmt_.append(store_t::make(zp_mask_, 0, expr));
            }
        }

        const stmt_t &stmt() const { return stmt_; }
        expr_t ic_start() const { return ic_start_; }
        bool is_scalar() const { return is_scalar_; }
        bool is_simd() const { return is_simd_; }
        bool is_const() const { return is_const_; }
        bool is_bool() const { return !size_ || !is_simd() || is_scalar(); }

        expr_t gen_mask(int base) const {
            auto null_mask = (is_bool()) ? expr_t() : expr_t(-1);
            if (!size_ || is_scalar_) return (size_) ? var_mask_ : null_mask;
            return word2bool(base % size_);
        }

        expr_t maybe_gen_mask_let(const stmt_t &loop) const {
            return (size_ && is_scalar_)
                    ? let_t::make(var_mask_, word2bool(0), loop)
                    : loop;
        }

    private:
        type_t w_type(int width = 1) const {
            return (is_bool()) ? type_t::u16(width) : type_t::s32(width);
        }
        int w_stride() const { return w_type().size(); }

        expr_t word2bool(int off) const {
            auto type = (is_bool()) ? type_t::u16() : type_t::s16(16);
            expr_t load;
            if (is_wide_ && !is_bool()) {
                load = load_t::make(
                        type, zp_mask_, off * 8 * w_stride(), w_stride());
            } else {
                load = load_t::make(
                        type.scalar(), zp_mask_, off * w_stride(), w_stride());
                if (!is_bool()) load = shuffle_t::make_broadcast(load, 16);
            }
            return (is_bool()) ? cast_t::make(type_t::_bool(16), load) : load;
        }

        expr_t vector2expr(const std::vector<expr_t> &expr,
                object_eq_map_t<expr_t, expr_t> &vars) const {
            constexpr size_t mask = 0x8000;
            auto hash = [&](const binary_op_t &b) -> size_t {
                return size_t(b.op_kind) | ((b.b.is<int_imm_t>()) ? mask : 0UL);
            };
            auto fetch_var = [this, &vars](expr_t e) {
                if (vars.find(e) == vars.end()) {
                    auto var = lmb_.ir_ctx_.create_tmp_var(
                            type_t::s32(e.type().elems()), "zp_mask");
                    vars.emplace(e, var);
                }
                return vars[e];
            };
            if (expr.empty()) return expr_t();
            // Can only vectorize if the element count is a power of 2
            ir_assert(math::is_pow2(expr.size())) << "Cannot vectorize.";

            std::unordered_map<size_t, size_t> kind;
            for (const expr_t &e : expr)
                if (const auto *bin = e.as_ptr<binary_op_t>())
                    kind[hash(*bin)]++;
            if (!kind.empty()) {
                using k_type = decltype(kind)::value_type;
                auto k = std::max_element(
                        kind.begin(), kind.end(), [](k_type &a, k_type &b) {
                            return a.second < b.second;
                        });
                const auto k_raw = op_kind_t(k->first & (mask - 1));
                std::vector<expr_t> a, b;
                for (const expr_t &e : expr) {
                    const auto *bin = e.as_ptr<binary_op_t>();
                    if (bin && (hash(*bin) == k->first)) {
                        a.emplace_back(bin->a);
                        b.emplace_back(bin->b);
                    } else {
                        const int is_mul = (k_raw == op_kind_t::_mul);
                        ir_assert(is_mul || (k_raw == op_kind_t::_add));
                        a.emplace_back(e);
                        b.emplace_back(is_mul);
                    }
                }
                auto a_new = vector2expr(a, vars);
                auto b_new = vector2expr(b, vars);
                if (auto *a_bin = a_new.as_ptr<binary_op_t>())
                    if ((a_bin->op_kind == op_kind_t::_add) && is_var(a_bin->b)
                            && is_cmp_op(k_raw) && is_shuffle_const(b_new))
                        for (auto &v : vars)
                            if (v.second.is_equal(a_bin->b)) {
                                auto fold = const_fold_non_recursive(
                                        b_new - v.first);
                                return binary_op_t::make(negate_cmp_op(k_raw),
                                        fetch_var(fold), a_bin->a);
                            }
                return binary_op_t::make(k_raw, a_new, b_new);
            }

            size_t num_ints = 0;
            for (const expr_t &e : expr)
                num_ints += e.is<int_imm_t>();
            ir_assert((num_ints == 0) || (num_ints == expr.size()));
            if (num_ints == expr.size()) {
                auto offs = shuffle_t::make(expr);
                if (offs.as<shuffle_t>().is_broadcast()) return offs;
                return fetch_var(offs);
            }

            size_t num_bools = 0;
            for (const expr_t &e : expr)
                num_bools += e.is<bool_imm_t>();
            ir_assert((num_bools == 0) || (num_bools == expr.size()));
            if (num_bools == expr.size()) return shuffle_t::make(expr);

            ir_assert(expr.front().is<var_t>());
            for (const expr_t &e : expr)
                ir_assert(e.is_same(expr.front()));
            return shuffle_t::make_broadcast(expr.front(), int(expr.size()));
        }

        load_multiply_builder_t &lmb_;
        bool is_const_;
        bool is_simd_;
        bool is_scalar_;
        bool is_wide_;
        int size_;
        expr_t ic_start_;
        expr_t var_mask_;
        expr_t zp_mask_;
        stmt_t stmt_;
    };

    stmt_t maybe_add_src_zps(const view_t &a_view, const view_t &b_view,
            const multiply_builder_t &mul_builder, int i_buf, int j_buf) {
        if (!cfg_.zp_cfg.do_src_compensation) return mul_builder.stmt();
        const bool is_runtime = cfg_.zp_cfg.is_runtime_src_zero_points;
        const bool is_scalar = cfg_.zp_cfg.is_common_src_zero_point;
        const bool is_mad = (cfg_.fma_kind == fma_kind_t::mad);
        const int channels = utils::rnd_up_pow2(
                (!is_mad) ? (cfg_.is_fwd) ? cfg_.ic : cfg_.oc : cfg_.g);
        const int c_blk = (channels < 32) ? channels : 32;
        const int k_blk = ((channels > 4)) ? 32 / c_blk : 1;
        const int m_blk = cfg_.simd_size();

        const type_t s_type = a_view.type();
        const type_t i_type = type_t::s32(); // x32 type that is always signed
        auto has_sign = [&]() {
            if (is_runtime) return s_type.is_signed();
            ir_assert(is_scalar);
            return cfg_.zp_cfg.common_src_zero_point < 0;
        };
        const type_t d_type = (has_sign()) ? type_t::s32() : type_t::u32();
        ir_assert((is_mad) ? s_type.is_x16() : s_type.is_x8());

        const int a_stride
                = s_type.size() * int(a_view.tlayout().blocks()[0].stride);
        int desc_m = 0, desc_n = 0;

        if (!is_mad) {
            auto &mapper = gemm_schedule_.bmnk_mapper();
            auto a_layout = mapper.map_to_bmnk(abc_kind_t::a,
                    {bmnk_kind_t::m, bmnk_kind_t::k}, a_view.create_vlayout());
            auto b_layout = mapper.map_to_bmnk(abc_kind_t::b,
                    {bmnk_kind_t::k, bmnk_kind_t::n}, b_view.create_vlayout());
            if (mul_builder.do_transpose()) {
                a_layout = a_layout.transpose();
                b_layout = b_layout.transpose();
                std::swap(a_layout, b_layout);
            }
            multiply_desc_t desc(a_layout, b_layout, true);
            desc_m = desc.m();
            desc_n = desc.n();
        } else {
            desc_n = a_view.tlayout().size() / m_blk / a_stride;
            desc_m = m_blk;
        }
        src_zp_mask_info_t masks(*this, m_blk, k_blk, desc_m, desc_n, channels,
                a_stride, is_mad, a_view);
        stmt_t data = masks.stmt();

        const int simd_per_ic = utils::div_up(
                std::min((!is_scalar) ? channels : 1, 32), m_blk);
        const std::vector<dim_t> dims
                = {m_blk * std::min((is_mad) ? 1 : 2, simd_per_ic)};
        const bool sc_ic = is_scalar || (channels <= 32);
        expr_t offs = (!sc_ic) ? masks.ic_start() * d_type.size() : 0;

        if (is_runtime && !sc_ic && !cfg_.do_pipeline_unroll
                && (cfg_.slm_bufs > 1)) {
            auto buf = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "zp_mask");
            register_buffer(buf, type_t::u32().size(), alloc_kind_t::grf);
            data = data.append(store_t::make(buf, 0, offs));
            offs = load_t::make(type_t::u32(), buf, 0);
        }

        auto get_src_zp_size = [](bool scalar, bool runtime, bool mad, int b) {
            if (scalar) return (!mad) ? b * 2 : ((runtime) ? b : 0);
            return (!mad) ? std::max(b * 2, 32) : 32;
        };
        const int m_blk_x2 = std::min(m_blk * 2, 16);
        const int src_zp_size = get_src_zp_size(
                is_scalar, is_runtime, is_mad, m_blk_x2 * k_blk);
        if (zp_buf_.is_empty())
            zp_buf_ = ir_ctx_.create_tmp_var(type_t::byte_ptr(), "zp_buf");
        zp_buf_size_ = std::max(zp_buf_size_, src_zp_size * d_type.size());

        for (int i = (is_runtime) ? 0 : std::numeric_limits<int>::max();
                i < m_blk * simd_per_ic; i += dims[0]) {
            const int b = i * d_type.size();
            view_t zpv(layout_t(d_type, 0, dims));
            auto read = make_access_builder(ir_ctx_, zpv,
                    kernel_info_.find_arg("src_zero_points")[offs + b],
                    zp_buf_[b], send_op_t::load, send_address_t::a64);
            data = data.append(read.stmt());
        }

        if (is_mad) {
            // TODO: for now, only b-blocking (per G) of the MAD loop is ready;
            //       please implement n-blocking (per OC) as well!
            ir_assert(a_view.tlayout().size() % a_stride == 0);
            ir_assert(cfg_.ic == 1);
            ir_assert(masks.is_simd());

            std::vector<stmt_t> loop(std::max(1, 32 / m_blk));
            for (int a_off = 0; a_off < a_view.tlayout().size();
                    a_off += m_blk * a_stride) {
                int iter = (a_off / m_blk / a_stride) % loop.size();
                type_t sv_type(s_type.kind(), m_blk);
                type_t b_type(s_type.kind(), (!is_scalar) ? m_blk : 1);
                auto a = load_t::make(sv_type, a_buf_, a_off, a_stride);
                auto b_off
                        = (!is_scalar && (channels > m_blk)) ? iter * m_blk : 0;
                auto b = (is_runtime) // '4'-s mean '(|i32| / |i16|) * |i16|'
                        ? load_t::make(b_type, zp_buf_, b_off * 4, 4)
                        : cfg_.zp_cfg.common_src_zero_point;
                auto mask = masks.gen_mask(
                        (utils::div_up(k_blk, 2)) * a_off / m_blk / a_stride);
                auto mad = (masks.is_bool())
                        ? binary_op_t::make(op_kind_t::_sub, a, b, sv_type)
                        : ternary_op_t::make(
                                op_kind_t::_mad, a, mask, b, sv_type);
                loop[iter] = loop[iter].append(store_t::make(a_buf_, a_off, mad,
                        a_stride, (masks.is_bool()) ? mask : expr_t()));
            }
            for (size_t i = 1; i < loop.size(); i++)
                loop[0] = loop[0].append(loop[i]);
            return data.append(masks.maybe_gen_mask_let(
                    loop[0].append(mul_builder.stmt())));
        }

        if (is_scalar) {
            expr_t expr = (!is_runtime)
                    ? (cfg_.zp_cfg.common_src_zero_point & 0xFF) * 0x01010101
                    : cast_t::make(type_t::s8(4),
                            shuffle_t::make_broadcast(
                                    load_t::make(s_type, zp_buf_, 0), 4));
            data = data.append(store_t::make(zp_buf_, 0, expr));
        } else {
            data = data.append(store_t::make(zp_buf_, 0,
                    load_t::make(type_t::u8(m_blk_x2), zp_buf_, 0, 4)));
            if (channels > 16)
                data = data.append(store_t::make(zp_buf_, 16,
                        load_t::make(type_t::u8(m_blk_x2), zp_buf_, 64, 4)));
            if (m_blk_x2 != m_blk)
                data = data.append(store_t::make(zp_buf_, 32,
                        load_t::make(type_t::u32(4), zp_buf_, 4, 8), 8));
        }
        std::vector<stmt_t> parts;

        auto wide_scalar = [m_blk](const expr_t &a, const expr_t &b, int blk) {
            if (blk == m_blk) return shuffle_t::make_broadcast(a, m_blk);
            std::vector<int> index(blk, 1);
            for (int i = 0; i < m_blk; i++)
                index[i] = 0;
            return shuffle_t::make(std::vector<expr_t> {a, b}, index);
        };
        auto wide_vector = [m_blk, i_type](const expr_t &a, int blk) {
            if (blk == m_blk)
                return load_t::make(type_t(i_type.kind(), m_blk), a, 0);
            std::vector<expr_t> vec(m_blk);
            std::vector<int> index(blk);
            for (int i = 0; i < m_blk; i++) {
                vec[i] = load_t::make(i_type, a, i * i_type.size());
                index[i + m_blk] = index[i] = i;
            }
            return shuffle_t::make(vec, index);
        };
        std::vector<expr_t> acc;
        for (int i = 1; i <= 2 * k_blk; i++)
            acc.emplace_back(
                    zp_buf_[(src_zp_size
                                    - utils::div_up(
                                              i, m_blk != m_blk_x2 ? 1 : 2)
                                            * m_blk)
                            * d_type.size()]);
        for (int i_m = 0; i_m < desc_m; i_m += m_blk) {
            const int blk
                    = (masks.is_simd() || masks.is_scalar()) ? m_blk_x2 : m_blk;
            for (int i = 0; i < k_blk; i++) {
                for (int i_k = i * (c_blk / 4); i_k
                        < ((channels > 4) ? (c_blk + i * c_blk) / 4 : cfg_.kw);
                        i_k += m_blk_x2 / m_blk) {
                    type_t vi(i_type.kind(), m_blk_x2);
                    const int szp_off = (is_scalar) ? 0 : (i_k * d_type.size());
                    auto b0 = load_t::make(d_type, zp_buf_, szp_off);
                    auto b1 = load_t::make(
                            d_type, zp_buf_, szp_off + m_blk * d_type.size());
                    auto b = (is_scalar) ? b0 : wide_scalar(b0, b1, m_blk_x2);
                    auto c = load_t::make(vi, b_buf_,
                            (i_m * (32 / 4) + i_k * m_blk) * d_type.size());
                    if (is_scalar) std::swap(b, c);
                    auto a = (i_k != i * (c_blk / 4))
                            ? load_t::make(vi, acc[i * 2 + 1], 0)
                            : expr_t(0);
                    parts.emplace_back(store_t::make(acc[i * 2 + 1], 0,
                            ternary_op_t::make(op_kind_t::_dp4a, a, b, c, vi)));
                }

                if (m_blk_x2 != m_blk) {
                    type_t vi(i_type.kind(), m_blk);
                    auto a = load_t::make(vi, acc[i * 2 + 1], 0);
                    auto b = load_t::make(vi, acc[i * 2 + 0], 0);
                    parts.emplace_back(store_t::make(acc[i * 2 + 1], 0, a + b));
                    if (!masks.is_bool() && (blk != m_blk))
                        parts.emplace_back(store_t::make(acc[i * 2 + 0], 0, a));
                }
            }
            for (int i_n = 0; i_n < desc_n; i_n += blk / m_blk) {
                int off_n = i_m / m_blk * desc_n + i_n;
                const int ij_buf = i_buf * cfg_.b_sub_tiles + j_buf;
                auto dst = c_buf_ + off_n * m_blk * d_type.size()
                        + ij_buf * mul_builder.c_layout().size();
                type_t vi(i_type.kind(), blk);
                auto a = load_t::make(vi, dst, 0);
                for (int i = 0; i < k_blk; i++) {
                    auto mask = masks.gen_mask(off_n * k_blk + i * blk / m_blk);
                    if (!masks.is_bool()) {
                        auto b = load_t::make(vi, acc[i * 2 + 1], 0);
                        auto mad = ternary_op_t::make(
                                op_kind_t::_mad, a, b, mask);
                        parts.emplace_back(store_t::make(dst, 0, mad));
                    } else {
                        auto b = wide_vector(acc[i * 2 + 1], blk);
                        auto sub = binary_op_t::make(op_kind_t::_sub, a, b);
                        parts.emplace_back(store_t::make(
                                dst, 0, sub, store_t::default_stride, mask));
                    }
                }
            }
        }
        // Stick the compensations between DPASes for better GPU utilization
        auto raw_dpas = flatten_statements(mul_builder.stmt());
        std::vector<stmt_t> dpas;
        stmt_t full;
        expr_t src1;
        for (auto &r : raw_dpas) {
            ir_assert(is_func_call<dpas_t>(r));
            auto &this_src1 = dpas_t::arg_src1(r);
            if (this_src1.is_equal(src1)) {
                dpas.back() = dpas.back().append(r);
            } else {
                src1 = this_src1;
                dpas.emplace_back(r);
            }
        }
        ir_assert(parts.size() % dpas.size() == 0);
        const int loop_size = int(parts.size()) / int(dpas.size());
        for (int i = 0; i < int(dpas.size()); i++) {
            full = full.append(dpas[i]);
            const auto k = (i + int(dpas.size()) / 2) % int(dpas.size());
            for (int j = k * loop_size; j < (k + 1) * loop_size; j++)
                full = full.append(parts[j]);
        }
        return data.append(masks.maybe_gen_mask_let(full));
    }

    void build_sub_tile(int i, int j) {
        bool is_first = (i == 0 && j == 0);

        stmt_t ab_s2r_load;
        stmt_t ab_g2r_load;
        if (!a_sub_tiles_[i].is_loaded()) {
            ab_s2r_load = ab_s2r_load.append(a_sub_tiles_[i].s2r_load());
            ab_g2r_load = ab_g2r_load.append(a_sub_tiles_[i].g2r_load());
            a_sub_tiles_[i].set_loaded();
        }
        if (!b_sub_tiles_[j].is_loaded()) {
            ab_s2r_load = ab_s2r_load.append(b_sub_tiles_[j].s2r_load());
            ab_g2r_load = ab_g2r_load.append(b_sub_tiles_[j].g2r_load());
            b_sub_tiles_[j].set_loaded();
        }
        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::g2r_load(i + j), ab_g2r_load));
        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::s2r_load(i + j), ab_s2r_load));

        auto &a_i_view = a_sub_tiles_[i].reg_view();
        auto &b_j_view = b_sub_tiles_[j].reg_view();

        // Multiply C_i_j += A_i x B_j in GEMM notation.
        multiply_builder_t mul_builder(cfg_, gemm_schedule_.bmnk_mapper(),
                a_i_view, b_j_view, a_buf_, b_buf_, c_buf_[c_buf_off_]);
        c_sub_tile_layout_ = mul_builder.c_layout();

        auto mul_total
                = maybe_add_src_zps(a_i_view, b_j_view, mul_builder, i, j);

        c_buf_off_ += c_sub_tile_layout_.size();
        ir_trace() << "Multiply (" << i << ", " << j << "):\n"
                   << mul_total.str() << std::endl;

        load_mul_stmt_ = load_mul_stmt_.append(
                stmt_group_t::make(stmt_label_t::mul(i + j), mul_total));

        if (!is_first) {
            ir_assert(mul_builder.c_layout() == c_sub_tile_layout_)
                    << "Sub-tile layouts must be equal.";
            return;
        }

        register_buffer(
                a_buf_, a_sub_tiles_[i].reg_buf_size(), alloc_kind_t::grf);
        register_buffer(
                b_buf_, b_sub_tiles_[j].reg_buf_size(), alloc_kind_t::grf);
    }
    void register_buffer(const stmt_t &alloc) {
        ir_assert(alloc.is<alloc_t>());
        allocs_.push_back(alloc);
    }

    void register_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        register_buffer(alloc_t::make(buf, size, kind, attr));
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    const gemm_schedule_t &gemm_schedule_;
    const fma_helper_t &fma_helper_;
    b_reduce_context_t &b_reduce_ctx_;

    expr_t ap_buf_;
    expr_t a_slm_buf_;

    expr_t bp_buf_;
    expr_t b_slm_buf_;

    expr_t zp_buf_;
    int zp_buf_size_ = 0;

    layout_t c_reg_layout_;

    expr_t ab_tmp_buf_;
    expr_t a_buf_;
    expr_t b_buf_;
    expr_t c_buf_;

    // Per-thread views to multiply.
    view_t a_thr_view_;
    view_t b_thr_view_;

    // Sub-tile indices.
    expr_t a_idx_;
    expr_t b_idx_;

    // Sub-tile views.
    view_t a_i_view_;
    view_t b_j_view_;

    tensor_t a_i_tile_;
    tensor_t b_j_tile_;

    std::vector<sub_tile_info_t> a_sub_tiles_;
    std::vector<sub_tile_info_t> b_sub_tiles_;

    std::vector<block_t> a_i_outer_blocks_;
    std::vector<block_t> b_j_outer_blocks_;

    std::vector<stmt_t> allocs_;

    stmt_t load_mul_stmt_;

    int c_buf_off_ = 0;
    layout_t c_sub_tile_layout_;

    const kernel_info_t &kernel_info_;
};

class compute_builder_t {
public:
    compute_builder_t(const conv_config_t &cfg, ir_context_t &ir_ctx,
            const kernel_info_t &kernel_info)
        : cfg_(cfg)
        , ir_ctx_(ir_ctx)
        , b_reduce_ctx_(ir_ctx, cfg)
        , g2s_ctx_(ir_ctx)
        , fma_helper_(cfg.simd_size(), cfg.fma_kind, cfg.a_data_type,
                  cfg.b_data_type, cfg.allow_a_grf_reorder,
                  cfg.allow_b_grf_reorder, !cfg.is_dw)
        , kernel_info_(kernel_info) {}

    int ab_slm_size() const { return ab_slm_size_; }

    const stmt_t &c_zero_out_stmt() const { return c_zero_out_stmt_; }
    const stmt_t &b_reduced_zero_out_stmt() const {
        return b_reduced_zero_out_stmt_;
    }

    stmt_t zero_out_stmt() const {
        stmt_t ret;
        ret = ret.append(c_zero_out_stmt());
        ret = ret.append(b_reduced_zero_out_stmt());
        return ret;
    }

    stmt_t iter_stmt() const {
        stmt_t stmt;
        bool use_prefetch = !prefetch_stmt_.is_empty();
        bool use_slm = !g2s_load_stmt_.is_empty();
        if (use_prefetch) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::prefetch(), prefetch_stmt_));
        } else if (use_slm) {
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_load(), g2s_load_stmt_));
            stmt = stmt.append(funcs::barrier());
            stmt = stmt.append(stmt_group_t::make(
                    stmt_label_t::g2s_store(), g2s_store_stmt_));
            stmt = stmt.append(funcs::barrier());
        }
        stmt = stmt.append(load_mul_stmt_);
        return stmt;
    }

    const stmt_t &c_store_stmt() const { return c_store_stmt_; }
    const stmt_t &b_reduced_store_stmt() const { return b_reduced_store_stmt_; }

    stmt_t inject_compute_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, compute_allocs_);
    }

    stmt_t inject_out_alloc_stmts(const stmt_t &stmt) const {
        return jit::inject_alloc_stmts(stmt, out_allocs_);
    }

    stmt_t inject_let_stmts(const stmt_t &stmt) const {
        return jit::inject_let_stmts(stmt, g2s_ctx_.grid_idx_lets);
    }

    void set_gemm_schedule(const gemm_schedule_t &gemm_schedule) {
        gemm_schedule_ = gemm_schedule;
    }

    // Setters for original AP/BP/CP buffers (P - problem notation).
    void set_ap_buf(const expr_t &buf) { ap_buf_ = buf; }
    void set_bp_buf(const expr_t &buf) { bp_buf_ = buf; }
    void set_cp_buf(const expr_t &buf) { cp_buf_ = buf; }
    void set_b_reduced_mem_buf(const expr_t &buf) {
        b_reduce_ctx_.set_b_reduced_mem_buf(buf);
    }

    void set_b_reduced_view(const view_t &v) {
        b_reduce_ctx_.set_b_reduced_view(v);
    }

    void set_post_op_context(const post_op_context_t &post_op_ctx) {
        post_op_ctx_ = post_op_ctx;
    }

    void set_reduce_condition(const expr_t &cond) {
        b_reduce_ctx_.set_reduce_condition(cond);
    }

    void build() {
        // Initialize SLM buffers.
        expr_t a_slm_buf = make_buffer("a_slm");
        expr_t b_slm_buf = make_buffer("b_slm");

        view_t ap_gmem_view = gemm_schedule_.a_tg_view();
        view_t bp_gmem_view = gemm_schedule_.b_tg_view();

        // Views to multiply by a thread group (either GMEM or SLM).
        view_t ap_x_view;
        view_t bp_x_view;
        prepare_gmem_to_slm("A", cfg_.use_a_slm, gemm_schedule_.a_tg_tile(),
                ap_gmem_view, ap_buf_, a_slm_buf, ap_x_view, g2s_ctx_);
        prepare_gmem_to_slm("B", cfg_.use_b_slm, gemm_schedule_.b_tg_tile(),
                bp_gmem_view, bp_buf_, b_slm_buf, bp_x_view, g2s_ctx_);
        prepare_prefetch("A", cfg_.use_prefetch, ap_gmem_view, ap_buf_);
        prepare_prefetch("B", cfg_.use_prefetch, bp_gmem_view, bp_buf_);

        if (ap_x_view.is_empty()) ap_x_view = ap_gmem_view;
        if (bp_x_view.is_empty()) bp_x_view = bp_gmem_view;

        for (auto &bi : g2s_ctx_.bufs) {
            register_compute_buffer(bi.buf, bi.size, alloc_kind_t::grf);
        }

        load_multiply_builder_t load_mul_builder(cfg_, ir_ctx_, gemm_schedule_,
                fma_helper_, b_reduce_ctx_, ap_buf_, a_slm_buf, bp_buf_,
                b_slm_buf, ap_x_view, bp_x_view, kernel_info_);

        load_mul_stmt_ = load_mul_builder.load_mul_stmt();
        compute_allocs_.insert(compute_allocs_.end(),
                load_mul_builder.allocs().begin(),
                load_mul_builder.allocs().end());

        auto c_buf = load_mul_builder.c_buf();
        int c_size = load_mul_builder.c_reg_layout().size();
        int c_size_grf_rounded = utils::rnd_up(c_size, cfg_.hw_cfg.grf_size());
        register_out_buffer(c_buf, c_size_grf_rounded, alloc_kind_t::grf);

        auto c_thr_reg_layout = load_mul_builder.c_reg_layout();
        auto thr_tile = gemm_schedule_.c_thr_tile(/*is_relative=*/false);

        auto reduce_cond = expr_t();
        if (gemm_schedule_.with_thread_group_k_slicing()) {
            slm_reduce_builder_t slm_reduce_builder(ir_ctx_,
                    gemm_schedule_.tg_grid(), c_buf, c_thr_reg_layout,
                    thr_tile);
            c_store_stmt_ = c_store_stmt_.append(slm_reduce_builder.stmt());
            c_thr_reg_layout = slm_reduce_builder.reg_layout();
            thr_tile = slm_reduce_builder.thr_tile();
            reduce_cond = slm_reduce_builder.reduce_cond();
        }

        auto c_thr_mem_view = gemm_schedule_.c_view().create_sub_view(thr_tile);
        auto c_m2g_stmt = create_epilogue_stmt(cfg_, ir_ctx_, gemm_schedule_,
                post_op_ctx_, thr_tile, c_thr_mem_view, c_thr_reg_layout,
                cp_buf_, c_buf);
        if (!reduce_cond.is_empty())
            c_m2g_stmt = if_t::make(reduce_cond, c_m2g_stmt);
        ir_trace() << "C GRF to GMEM store:\n" << c_m2g_stmt << std::endl;

        c_zero_out_stmt_ = stmt_group_t::make(stmt_label_t::c_zero_out(),
                create_zero_out_stmt(ir_ctx_, c_buf, c_size));
        c_store_stmt_ = c_store_stmt_.append(c_m2g_stmt);

        if (cfg_.do_b_reduction) {
            auto &ctx = b_reduce_ctx_;
            b_reduced_zero_out_stmt_ = create_zero_out_stmt(
                    ir_ctx_, ctx.b_reduced_reg_buf(), ctx.b_reduced_size());
            b_reduced_store_stmt_ = ctx.create_store_stmt();
            register_out_buffer(ctx.b_reduced_reg_buf(), ctx.b_reduced_size(),
                    alloc_kind_t::grf);
        }

        // Replace DPAS by DPASW when applicable.
        if (cfg_.fma_kind == fma_kind_t::dpasw) {
            alloc_updater_t alloc_updater;
            inject_dpasw(cfg_.hw(), load_mul_stmt_, c_buf, c_store_stmt_,
                    alloc_updater, gemm_schedule_.tg_grid().idx(0));
            for (auto &a : compute_allocs_) {
                a = alloc_updater.update(a);
            }
            for (auto &a : out_allocs_) {
                a = alloc_updater.update(a);
            }
        }

        // Assign {Atomic} for DPAS(W) when applicable.
        load_mul_stmt_ = inject_atomic(load_mul_stmt_);
    }

private:
    struct buf_info_t {
        buf_info_t(const std::string &tag, const expr_t &buf)
            : tag(tag), buf(buf) {}

        std::string tag;
        expr_t buf;
        int size = 0;
    };

    struct g2s_context_t {
        g2s_context_t(ir_context_t &ir_ctx) : ir_ctx(ir_ctx) {}

        expr_t create_buf(const char *tag, bool force_reuse = false) {
            if (reuse_buffers || force_reuse) {
                for (auto &bi : bufs) {
                    if (bi.tag == tag) return bi.buf;
                }
            }
            auto buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), tag);
            bufs.emplace_back(tag, buf);
            return buf;
        }

        void set_buf_size(const expr_t &buf, int size) {
            for (auto &bi : bufs) {
                if (bi.buf.is_same(buf)) bi.size = std::max(bi.size, size);
            }
        }

        expr_t create_tmp_grid_idx() {
            auto var = ir_ctx.create_tmp_var(type_t::s32(), "idx");
            tmp_grid_idxs.insert({var, expr_t()});
            return var;
        }

        void set_grid_idx_value(const expr_t &idx, const expr_t &value) {
            auto &old = tmp_grid_idxs[idx];
            ir_assert(old.is_empty());
            old = substitute_grid_idx_value(value);
        }

        expr_t substitute_grid_idx_value(const expr_t &_e) {
            auto e = _e;
            auto vars = find_unique_objects<var_t>(e);
            for (auto &v : vars) {
                auto it = tmp_grid_idxs.find(v);
                if (it == tmp_grid_idxs.end()) continue;
                e = substitute(e, v, it->second);
            }
            return e;
        }

        void register_grid(const grid_info_t &grid) {
            for (int i = 0; i < grid.ndims(); i++) {
                auto &idx = grid.idx(i);
                auto it = tmp_grid_idxs.find(idx);
                if (it == tmp_grid_idxs.end()) continue;
                grid_idx_lets.emplace_back(let_t::make(idx, it->second));
            }
        }

        ir_context_t &ir_ctx;
        grid_info_t prev_load_grid;
        bool reuse_buffers = false;
        std::vector<buf_info_t> bufs;

        object_map_t<expr_t, expr_t> tmp_grid_idxs;
        std::vector<stmt_t> grid_idx_lets;
    };

    void register_compute_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        compute_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    void register_out_buffer(const expr_t &buf, int size, alloc_kind_t kind,
            const alloc_attr_t &attr = {}) {
        out_allocs_.push_back(alloc_t::make(buf, size, kind, attr));
    }

    // Handles GMEM to SLM load for A and B. Done in two steps:
    // 1. Load: GMEM -> GRF (temporary)
    // 2. Store: GRF (temporary) -> SLM
    void prepare_gmem_to_slm(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            g2s_context_t &g2s_ctx) {
        if (!use_x_slm) return;

        grid_info_t load_grid = gemm_schedule_.tg_grid();
        for (;;) {
            bool ok = prepare_gmem_to_slm_impl(tag, use_x_slm, tg_tile,
                    x_gmem_view, xp_buf, x_slm_buf, x_slm_view, load_grid,
                    g2s_ctx);
            if (ok) {
                g2s_ctx.prev_load_grid = load_grid;
                g2s_ctx.register_grid(load_grid);
                return;
            }

            // Reduce grid and try again.
            auto grid_idx = g2s_ctx.create_tmp_grid_idx();
            int dim_idx;
            expr_t grid_idx_value;
            auto new_load_grid
                    = load_grid.halven(grid_idx, dim_idx, grid_idx_value);
            if (new_load_grid.is_empty()) break;

            if (new_load_grid == g2s_ctx.prev_load_grid) {
                new_load_grid = load_grid.halven(
                        grid_idx, dim_idx, grid_idx_value, /*first=*/false);
                g2s_ctx.reuse_buffers = true;
            }
            g2s_ctx.set_grid_idx_value(grid_idx, grid_idx_value);

            ir_ctx_.add_constraint(grid_idx >= 0);
            ir_ctx_.add_constraint(grid_idx < new_load_grid.dim(dim_idx));

            load_grid = new_load_grid;
        }
        ir_error_not_expected() << "Can't create GMEM -> SLM loads/stores.";
    }

    bool prepare_gmem_to_slm_impl(const char *tag, bool use_x_slm,
            const tensor_t &tg_tile, const view_t &x_gmem_view,
            const expr_t &xp_buf, const expr_t &x_slm_buf, view_t &x_slm_view,
            const grid_info_t &load_grid, g2s_context_t &g2s_ctx) {
        bool is_a = (tag[0] == 'A');
        abc_kind_t ab_kind = (is_a ? abc_kind_t::a : abc_kind_t::b);

        auto xp_slm_layout = create_slm_layout(x_gmem_view, ab_kind, load_grid);

        auto grid_cond = load_grid.slice_condition();

        // Per-thread tile and view to load from GMEM and store to SLM.
        tensor_t thr_tile;
        view_t x_g2s_view;
        if (cfg_.allow_slm_tg_slicing) {
            x_g2s_view = x_gmem_view.split(load_grid, thr_tile);
        } else {
            thr_tile = xp_slm_layout.split(load_grid);
            x_g2s_view = x_gmem_view.create_sub_view(thr_tile);
        }

        auto bound_cond = expr_t();
        if (is_a && !cfg_.fuse_spatial
                && thr_tile.elems() * load_grid.elems()
                        != xp_slm_layout.elems()) {
            for (int i = 0; i < x_gmem_view.nvdims(); i++) {
                if (!x_g2s_view.vstart(i).is_equal(x_gmem_view.vstart(i))) {
                    auto dim_expr
                            = x_g2s_view.vstart(i) - x_gmem_view.vstart(i);
                    if (bound_cond.is_empty())
                        bound_cond = dim_expr < x_gmem_view.vdims()[i];
                    else
                        bound_cond &= dim_expr < x_gmem_view.vdims()[i];
                }
            }
        }
        if (!bound_cond.is_empty()) {
            if (!grid_cond.is_empty())
                grid_cond = grid_cond & bound_cond;
            else
                grid_cond = bound_cond;
        }

        auto slm_thr_layout = xp_slm_layout.map(thr_tile);

        // Ensure that each thread writes a dense region to SLM. If the layout
        // is not dense, return and try with smaller grid.
        if (!slm_thr_layout.is_dense()) return false;

        register_compute_buffer(
                x_slm_buf, xp_slm_layout.size(), alloc_kind_t::slm);
        ab_slm_size_ += xp_slm_layout.size();

        // Temporary GRF buffer.
        expr_t x_g2s_reg_buf = g2s_ctx.create_buf("g2s");

        // GMEM -> GRF load.
        auto x_read = make_access_builder(ir_ctx_, x_g2s_view, xp_buf,
                x_g2s_reg_buf, send_op_t::load, send_address_t::a64);
        ir_trace() << tag << " GMEM to GRF load:\n"
                   << x_read.str() << std::endl;

        g2s_ctx.set_buf_size(x_g2s_reg_buf, x_read.reg_buf_size());

        auto load_stmt = x_read.stmt();
        if (!grid_cond.is_empty()) load_stmt = if_t::make(grid_cond, load_stmt);
        g2s_load_stmt_ = g2s_load_stmt_.append(load_stmt);

        // GRF -> SLM store.
        auto x_write = make_access_builder(ir_ctx_, view_t(slm_thr_layout),
                x_slm_buf, x_g2s_reg_buf, send_op_t::store,
                send_address_t::slm);
        ir_trace() << tag << " GRF to SLM store:\n"
                   << x_write.str() << std::endl;
        auto store_stmt = x_write.stmt();

        auto &read_layout = x_read.reg_layout();
        auto &write_layout = x_write.reg_layout();
        if (read_layout != write_layout) {
            if (is_a ? cfg_.allow_a_grf_reorder : cfg_.allow_b_grf_reorder) {
                // Temporary GRF buffer.
                expr_t tmp_buf
                        = g2s_ctx.create_buf("g2s_tmp", /*force_reuse=*/true);
                auto reorder_stmt = create_reorder_stmt(
                        read_layout, write_layout, x_g2s_reg_buf, tmp_buf);
                g2s_ctx.set_buf_size(tmp_buf, x_write.reg_buf_size());
                store_stmt = substitute(store_stmt, x_g2s_reg_buf, tmp_buf);
                store_stmt = reorder_stmt.append(store_stmt);
            } else {
                ir_error_not_expected() << "Requested register layouts for "
                                        << tag << " do not match: "
                                        << "read: " << read_layout
                                        << ", write: " << write_layout;
            }
        }
        // Generate reduction statement for B.
        if (!is_a && cfg_.do_b_reduction) {
            auto absolute_thr_tile = tg_tile.create_sub_tensor(thr_tile);
            b_reduce_ctx_.init_reduced_thr_view(absolute_thr_tile, grid_cond);
            auto reduce_stmt = b_reduce_ctx_.create_reduce_stmt(
                    read_layout, x_g2s_reg_buf);
            store_stmt = reduce_stmt.append(store_stmt);
        }
        if (!grid_cond.is_empty())
            store_stmt = if_t::make(grid_cond, store_stmt);
        g2s_store_stmt_ = g2s_store_stmt_.append(store_stmt);

        x_slm_view = view_t(xp_slm_layout);

        return true;
    }

    void prepare_prefetch(const char *tag, bool use_prefetch,
            const view_t &x_gmem_view, const expr_t &xp_buf) {
        if (!use_prefetch) return;

        // Per-thread view to prefetch from GMEM.
        auto thr_view = x_gmem_view.split(gemm_schedule_.tg_grid());

        auto send_hint = get_send_hint(cfg_.hw_cfg, send_op_t::prefetch,
                (tag[0] == 'A') ? abc_kind_t::a : abc_kind_t::b, thr_view,
                gemm_schedule_);

        // GMEM prefetch.
        auto x_prefetch = make_access_builder(ir_ctx_, thr_view, xp_buf,
                expr_t(), send_op_t::prefetch, send_address_t::a64, send_hint);

        // too many prefetches degrades performance
        if (find_objects<func_call_t>(x_prefetch.stmt()).size() > 16) {
            ir_warning() << "Dropping excessive prefetches." << std::endl;
            prefetch_stmt_ = stmt_t();
        } else {
            ir_trace() << tag << " GMEM prefetch:\n"
                       << x_prefetch.str() << std::endl;
            prefetch_stmt_ = prefetch_stmt_.append(x_prefetch.stmt());
        }
    }

    layout_t create_slm_layout(const view_t &tg_view, abc_kind_t abc_kind,
            const grid_info_t &load_grid) const {
        auto layout = tg_view.create_dense_vlayout();
        auto ret = fma_helper_.convert_to_fma_friendly_layout(layout, abc_kind,
                /*is_slm=*/true, gemm_schedule_.bmnk_mapper());
        if (cfg_.pad_slm) ret = pad_slm_layout(ret, load_grid);
        return ret.normalize();
    }

    // SLM has 65 dword-granularity banks (Xe_HP):
    //      banks:   [bank 0] [bank 1] [bank 2] ... [bank 0]
    // byte offsets: | 0      | 4      | 8      ... | 4 * 65
    // SLM reads don't have conflicts. During SLM writes each fused EU writes
    // 64 bytes (in total 128 bytes per clock). If there are repeating banks
    // between 128 bytes the write takes 2 clocks to complete.
    // Assume that every X-axis thread (across tg_dim[0]) writes the
    // corresponding outer block of the layout. The goal is to ensure that the
    // stride between outer blocks allows to avoid duplicated banks.
    layout_t pad_slm_layout(
            const layout_t &layout, const grid_info_t &load_grid) const {
        // EUs are not fused in XeHPC+ so no need to pad SLM.
        if (cfg_.hw() >= ngen::HW::XeHPC) return layout;
        auto tg_dim0 = load_grid.dim(0);
        auto tg_dim1 = load_grid.dim(1);
        int type_size = layout.type().size();

        ir_assert(layout.elems() % tg_dim0 == 0) << layout;
        dim_t inner_block = layout.elems() / tg_dim0;

        ir_assert((inner_block * type_size) % tg_dim1 == 0) << layout;
        dim_t per_thr_bytes = (inner_block * type_size) / tg_dim1;

        std::vector<dim_t> multi_blocks = {inner_block, tg_dim0};
        auto l = layout.split_into_multi_blocks(multi_blocks);

        if (l.is_empty()) {
            ir_warning() << "Couldn't split layout for SLM padding."
                         << std::endl;
            return layout;
        }
        auto padded_blocks = l.blocks();
        dim_t stride = -1;
        dim_t remaining_elems = inner_block;
        bool past_inner_block = remaining_elems == 1;
        for (auto &b : padded_blocks) {
            if (past_inner_block) {
                if (stride == -1) {
                    dim_t stride_bytes = find_min_stride_without_conflicts(
                            per_thr_bytes, dim_t(b.stride) * type_size);
                    ir_assert(stride_bytes % type_size == 0);
                    stride = stride_bytes / type_size;
                }
                b.stride = stride;
                stride = b.stride * b.block;
                continue;
            }
            ir_assert(remaining_elems % b.block == 0);
            remaining_elems /= b.block;
            if (remaining_elems == 1) past_inner_block = true;
        }
        return layout_t(
                layout.type(), layout.ndims(), layout.offset(), padded_blocks);
    }

    dim_t find_min_stride_without_conflicts(
            dim_t inner_bytes, dim_t dense_stride_bytes) const {
        int write_step = 64;
        int stride_step = 16;
        dim_t stride_beg = dense_stride_bytes;
        dim_t stride_end = 2 * dense_stride_bytes;
        auto arch = convert_ngen_arch_to_dnnl(cfg_.hw());
        const int slm_banks
                = compute::device_info_t::slm_memory_bank_count(arch);
        const int bank_granularity
                = compute::device_info_t::slm_memory_bank_granularity(arch);
        for (dim_t s = stride_beg; s < stride_end; s += stride_step) {
            bool ok = true;
            for (dim_t off0 = 0; off0 < inner_bytes; off0 += write_step) {
                // Check banks for a single SLM write.
                std::vector<bool> found(slm_banks, false);
                for (dim_t off = off0; off < off0 + write_step;
                        off += bank_granularity) {
                    int bank0 = (off / bank_granularity) % slm_banks;
                    int bank1 = ((off + s) / bank_granularity) % slm_banks;
                    if (found[bank0]) {
                        ok = false;
                        break;
                    }
                    found[bank0] = true;
                    if (found[bank1]) {
                        ok = false;
                        break;
                    }
                    found[bank1] = true;
                }
                if (ok) return s;
            }
        }

        ir_warning()
                << "Couldn't find stride without conflicts for SLM padding."
                << std::endl;

        return dense_stride_bytes;
    }

    const conv_config_t &cfg_;
    ir_context_t &ir_ctx_;
    post_op_context_t post_op_ctx_;
    b_reduce_context_t b_reduce_ctx_;

    g2s_context_t g2s_ctx_;
    fma_helper_t fma_helper_;

    gemm_schedule_t gemm_schedule_;

    expr_t ap_buf_;
    expr_t bp_buf_;
    expr_t cp_buf_;

    std::vector<stmt_t> compute_allocs_;
    std::vector<stmt_t> out_allocs_;
    int ab_slm_size_ = 0;

    stmt_t g2s_load_stmt_;
    stmt_t g2s_store_stmt_;
    stmt_t prefetch_stmt_;
    stmt_t load_mul_stmt_;

    stmt_t c_zero_out_stmt_;
    stmt_t c_store_stmt_;

    stmt_t b_reduced_zero_out_stmt_;
    stmt_t b_reduced_store_stmt_;

    const kernel_info_t &kernel_info_;
};

class compute_loop_label_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const for_t &obj) override {
        if (injected_) return obj;

        bool found_continue = false;
        auto calls = find_objects<func_call_t>(obj);
        for (auto &_c : calls) {
            auto &c = _c.as<func_call_t>();
            if (c.func.is_equal(funcs::continue_func())) found_continue = true;
        }

        if (!found_continue) {
            injected_ = true;
            return stmt_group_t::make(stmt_label_t::compute_loop(), obj);
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    bool injected_ = false;
};

// Injects compute_loop statement label to the outermost loop that can be
// pipelined. If a loop contains a "continue" function call it can't be
// pipelined because of conditional flow.
stmt_t inject_compute_loop_label(const stmt_t &s) {
    return compute_loop_label_injector_t().mutate(s);
}

void init_kernel_grid(const std::array<int, 3> &kernel_grid_dims,
        const std::array<int, 3> &tg_grid_dims, int simd_size,
        constraint_set_t &cset, grid_info_t &kernel_grid, grid_info_t &tg_grid,
        std::array<expr_t, 3> &local_id, std::vector<stmt_t> &init_stmts) {
    int grid_ndims = 3;
    kernel_grid = grid_info_t(grid_ndims);
    tg_grid = grid_info_t(grid_ndims);
    for (int i = 0; i < grid_ndims; i++) {
        local_id[i]
                = var_t::make(type_t::u16(), "local_id" + std::to_string(i));
        kernel_grid.dim(i) = kernel_grid_dims[i];
        kernel_grid.idx(i)
                = var_t::make(type_t::s32(), "grid_idx" + std::to_string(i));
        tg_grid.dim(i) = tg_grid_dims[i];
        tg_grid.idx(i)
                = var_t::make(type_t::s32(), "tg_idx" + std::to_string(i));

        int local_id_bound = tg_grid_dims[i];
        if (i == 0) local_id_bound *= simd_size;
        cset.add_constraint(local_id[i] >= 0);
        cset.add_constraint(local_id[i] < local_id_bound);

        cset.add_constraint(kernel_grid.idx(i) >= 0);
        cset.add_constraint(kernel_grid.idx(i) < kernel_grid_dims[i]);
        cset.add_constraint(tg_grid.idx(i) >= 0);
        cset.add_constraint(tg_grid.idx(i) < tg_grid_dims[i]);
    }

    for (int i = 0; i < grid_ndims; i++) {
        auto value = local_id[i];
        if (i == 0) value /= simd_size;
        auto &type = tg_grid.idx(i).type();
        init_stmts.push_back(let_t::make(tg_grid.idx(i), cast(value, type)));
    }
}

void kernel_builder_t::build() {
    constraint_set_t init_cset;

    trace_reset();

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(cfg_.kernel_grid_dim, cfg_.tg_grid_dim, cfg_.simd_size(),
            init_cset, kernel_grid_, tg_grid_, local_id_, init_stmts);

    gemm_schedule_t gemm_schedule(init_cset, kernel_grid_, tg_grid_);

    // Initialize memory buffers.
    std::vector<stmt_t> inner_lets;

    view_t a_view;
    view_t b_view;
    view_t c_view;
    view_t bp_reduced_view;

    expr_t ap_buf;
    expr_t bp_buf;
    expr_t cp_buf;
    expr_t b_reduced_mem_buf;
    expr_t b_reduction_condition;

    if (cfg_.is_fwd) {
        init_fwd(gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_d) {
        init_bwd_d(
                gemm_schedule, a_view, b_view, c_view, ap_buf, bp_buf, cp_buf);
    } else if (cfg_.is_bwd_w) {
        init_bwd_w(gemm_schedule, a_view, b_view, c_view, bp_reduced_view,
                ap_buf, bp_buf, cp_buf, b_reduced_mem_buf,
                b_reduction_condition);
    } else {
        ir_error_not_expected();
    }

    gemm_schedule.finalize();

    trace_stamp("GEMM Schedule");

    ir_context_t ir_ctx(cfg_.hw_cfg, init_cset);
    post_op_context_t post_op_ctx(pd_, cfg_, gemm_schedule, kernel_info_);
    compute_builder_t cb(cfg_, ir_ctx, kernel_info_);

    cb.set_gemm_schedule(gemm_schedule);
    cb.set_ap_buf(ap_buf);
    cb.set_bp_buf(bp_buf);
    cb.set_cp_buf(cp_buf);
    cb.set_b_reduced_mem_buf(b_reduced_mem_buf);
    cb.set_b_reduced_view(bp_reduced_view);
    cb.set_post_op_context(post_op_ctx);
    cb.set_reduce_condition(b_reduction_condition);

    cb.build();

    trace_stamp("Compute Builder");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    // Create IR statements.
    stmt_t loop_stmt = cb.iter_stmt();
    loop_stmt = gemm_schedule.create_loop_nest(loop_stmt);
    loop_stmt = inject_compute_loop_label(loop_stmt);
    loop_stmt = cb.inject_compute_alloc_stmts(loop_stmt);

    stmt_t c_store_stmt;
    c_store_stmt = c_store_stmt.append(cb.b_reduced_store_stmt());
    c_store_stmt = c_store_stmt.append(cb.c_store_stmt());
    c_store_stmt = stmt_group_t::make(stmt_label_t::c_store(), c_store_stmt);

    stmt_ = loop_stmt;
    stmt_ = stmt_seq_t::make(cb.zero_out_stmt(), stmt_);
    stmt_ = stmt_seq_t::make(stmt_, c_store_stmt);

    stmt_ = cb.inject_out_alloc_stmts(stmt_);
    stmt_ = cb.inject_let_stmts(stmt_);

    stmt_ = gemm_schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    trace_stop("Create Inital IR");

    stmt_ = inject_external_var_let(stmt_, ir_ctx);
    stmt_ = merge_slm_buffers(stmt_, ir_ctx);
    if (!cfg_.do_pipeline_unroll && (cfg_.use_a_slm || cfg_.use_b_slm)) {
        stmt_ = inject_simple_slm_buffering(
                stmt_, ir_ctx, cfg_, cb.ab_slm_size());
    } else if (!cfg_.do_pipeline_unroll && cfg_.use_prefetch) {
        // Simplify to remove loops with only 1 iteration
        stmt_ = simplify(stmt_, ir_ctx);
        stmt_ = inject_prefetch_pipeline(stmt_, ir_ctx, cfg_);
    }
    stmt_ = inject_slm_reorder(stmt_, ir_ctx, cfg_, tg_grid_);
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = lift_alloc(stmt_, ir_ctx, cfg_);
    stmt_ = lift_send_2d_header_store(stmt_, ir_ctx);
    stmt_ = hoist_send_masks(stmt_, ir_ctx, stmt_label_t::c_store(), false);
    stmt_ = eliminate_common_subexprs(stmt_, ir_ctx, cfg_);
    stmt_ = hoist_exprs(stmt_, ir_ctx);
    if (cfg_.do_pipeline_unroll) stmt_ = loop_strength_reduce(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    if (cfg_.do_pipeline_unroll) {
        stmt_ = update_loops_for_unrolling(stmt_, ir_ctx, cfg_);
        stmt_ = inject_unrolling(stmt_, ir_ctx, cfg_, cb.ab_slm_size());
    }
    if (cfg_.hoist_masks_from_compute_loop) {
        stmt_ = hoist_send_masks(
                stmt_, ir_ctx, stmt_label_t::compute_loop(), true);
    }
    stmt_ = fixup_if_conditions(stmt_, ir_ctx);
    stmt_ = unroll_loops(stmt_, ir_ctx);
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = maybe_strip_prefetches(stmt_, ir_ctx, cfg_);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    if (cfg_.hoist_masks_from_compute_loop) {
        stmt_ = remove_spurious_send_mask_cast(stmt_, ir_ctx);
    }
    stmt_ = fix_int32_overflow(stmt_, ir_ctx);
    stmt_ = optimize_peephole(stmt_, ir_ctx);
    stmt_ = optimize_barrier(stmt_, ir_ctx);
    if (cfg_.fma_kind == fma_kind_t::dp4a) stmt_ = inject_dp4a(stmt_, ir_ctx);
    stmt_ = inject_bank_conflict_attribute(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

#if !defined(NDEBUG) || defined(GEN_CONV_DEBUG)
    verify_buffer_access(stmt_, ir_ctx);
#endif

    ir_trace() << "Convolution kernel body:\n" << stmt_ << std::endl;
    trace_perf();
}

class tile_helper_t {
public:
    tile_helper_t(const layout_t &l)
        : l_(l)
        , running_blocks_(l.blocks().size(), 1)
        , blocks_(l.blocks().size(), 1) {
        const auto &l_blocks = l.blocks();
        const auto size = l_blocks.size();
        while (block_idx_ < size && l_blocks[block_idx_].block == 1)
            block_idx_++;
    }

    bool has_next() const { return block_idx_ < running_blocks_.size(); }

    dim_t size() const {
        dim_t ret = l_.size();
        for (auto b : blocks_)
            ret *= b;
        return ret;
    }

    bool is_dense() const {
        bool is_end = false;
        for (size_t i = 0; i < blocks_.size(); i++) {
            if (blocks_[i] == l_.blocks()[i].block) continue;
            if (blocks_[i] != 1 && is_end) return false;
            is_end = true;
        }
        return true;
    }

    tensor_t tile() const {
        std::vector<dim_t> dims(l_.ndims(), 1);
        for (size_t i = 0; i < blocks_.size(); i++) {
            int dim_idx = l_.blocks()[i].dim_idx;
            dims[dim_idx] *= blocks_[i];
        }
        return tensor_t(dims);
    }

    tensor_t next() {
        dim_t l_block = l_.blocks()[block_idx_].block;
        for (dim_t b = running_blocks_[block_idx_] + 1; b <= l_block; b++) {
            if (l_block % b == 0) {
                running_blocks_[block_idx_] = b;
                return running_tile();
            }
        }
        block_idx_++;
        if (has_next()) return next();
        return tensor_t();
    }

    void accept() { blocks_[block_idx_] = running_blocks_[block_idx_]; }

    static bool can_be_mapped(const layout_t &l, const tensor_t &t) {
        std::vector<dim_t> rem_dims = t.dims();
        for (auto &b : l.blocks()) {
            auto &rem_dim = rem_dims[b.dim_idx];
            if (rem_dim >= b.block) {
                if (rem_dim % b.block != 0) return false;
                rem_dim /= b.block;
                continue;
            }
            if (b.block % rem_dim != 0) return false;
            rem_dim = 1;
        }
        for (auto d : rem_dims)
            ir_assert(d == 1);
        return true;
    }

    static tensor_t merge(const tensor_t &a, const tensor_t &b) {
        std::vector<dim_t> dims(a.ndims());
        for (int i = 0; i < a.ndims(); i++) {
            dims[i] = std::max(a(i), b(i));
        }
        return tensor_t(dims);
    }

private:
    tensor_t running_tile() const {
        std::vector<dim_t> dims(l_.ndims(), 1);
        for (size_t i = 0; i < block_idx_; i++) {
            int dim_idx = l_.blocks()[i].dim_idx;
            dims[dim_idx] *= blocks_[i];
        }
        int dim_idx = l_.blocks()[block_idx_].dim_idx;
        dims[dim_idx] *= running_blocks_[block_idx_];
        return tensor_t(dims);
    }

    const layout_t &l_;
    std::vector<dim_t> running_blocks_;
    std::vector<dim_t> blocks_;
    size_t block_idx_ = 0;
};

void reorder_kernel_builder_t::compute_blocks(const layout_t &src,
        const layout_t &dst, std::vector<int> &iter_blocks,
        std::vector<int> &loop_blocks, std::vector<int> &tg_blocks,
        int max_iter_tile_bytes, int max_thr_tile_bytes) {
    if (max_iter_tile_bytes <= 0)
        max_iter_tile_bytes = default_max_iter_tile_bytes;
    if (max_thr_tile_bytes <= 0)
        max_thr_tile_bytes = default_max_thr_tile_bytes;

    ir_assert(src.ndims() == dst.ndims());
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    // Pad src/dst layouts to match each other.
    auto pad_layout = [&](const layout_t &l) {
        std::vector<block_t> padded_blocks;
        for (auto &eb : l.enumerated_blocks()) {
            auto b = eb.second;
            if (l.is_outermost(eb)) {
                dim_t inner = l.dim(b.dim_idx) / b.block;
                b.block = ir_utils::safe_divide(dims[b.dim_idx], inner);
            }
            padded_blocks.push_back(b);
        }
        return layout_t(
                l.type(), ndims, 0, padded_blocks, /*do_normalize=*/false);
    };
    layout_t padded_src = pad_layout(src);
    layout_t padded_dst = pad_layout(dst);
    ir_assert(ir_utils::is_equal(padded_src.dims(), padded_dst.dims()));

    int elems = padded_src.elems();
    int max_type_size = std::max(src.type().size(), dst.type().size());
    dim_t max_iter_tile_elems
            = std::min(max_iter_tile_bytes / max_type_size, elems);
    dim_t max_thr_tile_elems
            = std::min(max_thr_tile_bytes / max_type_size, elems);

    tile_helper_t src_th(padded_src);
    tile_helper_t dst_th(padded_dst);

    // Incrementally increase subtiles in src and dst. The goal is to find the
    // maximum src/dst tiles so that the final combined tile covers dense
    // regions as big as possible in src/dst layouts.
    std::vector<tensor_t> candidate_tiles;
    // To ensure there is at least one candidate.
    candidate_tiles.emplace_back(std::vector<dim_t>(ndims, 1));
    for (;;) {
        if (!src_th.has_next() || !dst_th.has_next()) break;
        tile_helper_t *th = &src_th;
        bool src_dense = src_th.is_dense();
        bool dst_dense = dst_th.is_dense();
        // When both sublayouts are dense, try to increase the smallest tile.
        // Otherwise, if there is a dense sublayout try to increase it.
        if (src_dense && dst_dense && dst_th.size() < src_th.size()) {
            th = &dst_th;
        } else if (dst_dense && !src_dense) {
            th = &dst_th;
        }

        auto tile = th->next();
        auto &other_th = (th == &src_th ? dst_th : src_th);
        tile = tile_helper_t::merge(tile, other_th.tile());
        if (tile_helper_t::can_be_mapped(padded_src, tile)
                && tile_helper_t::can_be_mapped(padded_dst, tile)) {
            th->accept();
            candidate_tiles.push_back(tile);
        }
        if (tile.elems() >= max_thr_tile_elems) break;
    }

    std::sort(candidate_tiles.begin(), candidate_tiles.end(),
            [](const tensor_t &a, const tensor_t &b) {
                return a.elems() > b.elems();
            });

    const tensor_t *thr_tile = nullptr;
    const tensor_t *iter_tile = nullptr;
    for (size_t i = 0; i < candidate_tiles.size(); i++) {
        auto &t = candidate_tiles[i];
        if (!thr_tile && t.elems() <= max_thr_tile_elems) thr_tile = &t;
        if (thr_tile && !iter_tile && t.elems() <= max_iter_tile_elems
                && thr_tile->is_divisible(t)) {
            iter_tile = &t;
        }
        if (thr_tile && iter_tile) break;
    }

    ir_assert(thr_tile);
    ir_assert(iter_tile);
    std::vector<int> thr_blocks(
            thr_tile->dims().begin(), thr_tile->dims().end());
    iter_blocks.assign(iter_tile->dims().begin(), iter_tile->dims().end());

    ir_assert(utils::array_product(iter_blocks) <= max_iter_tile_elems);
    ir_assert(utils::array_product(thr_blocks) <= max_thr_tile_elems);

    // Initialize loop blocks.
    loop_blocks.resize(ndims, 1);
    for (int i = 0; i < ndims; i++) {
        loop_blocks[i] = ir_utils::safe_divide(thr_blocks[i], iter_blocks[i]);
    }

    // Initialize thread group blocks.
    // Heuristic: try to split outer dimension and assign its
    // inner part to the thread group. This may give better
    // bandwidth utilization on XeHP/XeHPG.
    tg_blocks.resize(ndims, 1);
    const int tg_factor = 2;
    for (int i = 0; i < ndims; i++) {
        int outer = utils::div_up(dims[i], thr_blocks[i]);
        if (outer % tg_factor == 0) {
            tg_blocks[i] = tg_factor;
            break;
        }
    }
}

void reorder_kernel_builder_t::compute_blocks(const layout_t &src,
        const layout_t &dst, std::vector<int> &tile_blocks,
        std::vector<int> &tg_blocks) {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    compute_blocks(src, dst, iter_blocks, loop_blocks, tg_blocks);
    size_t n = iter_blocks.size();
    tile_blocks.resize(n);
    for (size_t i = 0; i < n; i++) {
        tile_blocks[i] = iter_blocks[i] * loop_blocks[i];
    }
}

void reorder_kernel_builder_t::compute_grid(const layout_t &src,
        const layout_t &dst, const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks, const std::vector<int> &tg_blocks,
        std::array<int, 3> &kernel_grid, std::array<int, 3> &tg_grid,
        std::vector<int> *dim2grid) {
    int ndims = src.ndims();
    std::vector<dim_t> dims(ndims);
    for (int i = 0; i < ndims; i++) {
        dims[i] = std::max(src.dim(i), dst.dim(i));
    }

    if (dim2grid) dim2grid->resize(ndims, -1);

    const int grid_ndims = 3;
    for (int i = 0; i < grid_ndims; i++) {
        kernel_grid[i] = 1;
        tg_grid[i] = 1;
    }

    int grid_idx = 0;
    int max_grid_idx = grid_ndims - 1;
    for (int i = 0; i < ndims; i++) {
        if (dim2grid) (*dim2grid)[i] = grid_idx;
        int outer = utils::div_up(
                dims[i], iter_blocks[i] * loop_blocks[i] * tg_blocks[i]);
        tg_grid[grid_idx] *= tg_blocks[i];
        kernel_grid[grid_idx] *= outer;
        if (outer != 1 && grid_idx != max_grid_idx) grid_idx++;
    }
}

compute::nd_range_t reorder_kernel_builder_t::nd_range(
        int simd, const layout_t &src, const layout_t &dst) {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    compute_blocks(src, dst, iter_blocks, loop_blocks, tg_blocks);
    std::array<int, 3> kernel_grid;
    std::array<int, 3> tg_grid;
    compute_grid(src, dst, iter_blocks, loop_blocks, tg_blocks, kernel_grid,
            tg_grid);
    std::array<size_t, 3> global;
    std::array<size_t, 3> local;
    for (size_t i = 0; i < kernel_grid.size(); i++) {
        global[i] = kernel_grid[i] * tg_grid[i];
        local[i] = tg_grid[i];
        if (i == 0) {
            global[i] *= simd;
            local[i] *= simd;
        }
    }
    return compute::nd_range_t(global.data(), local.data());
}

void reorder_kernel_builder_t::build() {
    std::vector<int> iter_blocks;
    std::vector<int> loop_blocks;
    std::vector<int> tg_blocks;
    compute_blocks(
            src_layout_, dst_layout_, iter_blocks, loop_blocks, tg_blocks);

    int max_iters = 10;
    int cur_iter_bytes = default_max_iter_tile_bytes;
    for (int i = 0; i < max_iters; i++) {
        if (try_build(iter_blocks, loop_blocks, tg_blocks)) {
            ir_info() << "Reorder configuration:" << std::endl;
            ir_info() << "  Source layout:              " << src_layout_
                      << std::endl;
            ir_info() << "  Destination layout:         " << dst_layout_
                      << std::endl;
            ir_info() << "  Iteration blocks:           "
                      << ir_utils::make_seq_print_helper(iter_blocks, " x ")
                      << std::endl;
            ir_info() << "  Loop blocks:                "
                      << ir_utils::make_seq_print_helper(loop_blocks, " x ")
                      << std::endl;
            ir_info() << "  Thread group blocks:        "
                      << ir_utils::make_seq_print_helper(tg_blocks, " x ")
                      << std::endl;
            return;
        }

        cur_iter_bytes /= 2;
        while (cur_iter_bytes >= 1) {
            std::vector<int> new_iter_blocks;
            compute_blocks(src_layout_, dst_layout_, new_iter_blocks,
                    loop_blocks, tg_blocks, cur_iter_bytes);
            if (!ir_utils::is_equal(new_iter_blocks, iter_blocks)) {
                iter_blocks = new_iter_blocks;
                break;
            }
            cur_iter_bytes /= 2;
        }
    }
    ir_error_not_expected();
}

bool reorder_kernel_builder_t::try_build(const std::vector<int> &iter_blocks,
        const std::vector<int> &loop_blocks,
        const std::vector<int> &tg_blocks) {
    constraint_set_t init_cset;

    int ndims = src_layout_.ndims();
    std::vector<expr_t> vars;
    for (int i = 0; i < ndims; i++) {
        char letter = 'a' + i;
        vars.push_back(var_t::make(type_t::s32(), std::string(1, letter)));
    }

    std::array<int, 3> kernel_grid_dims;
    std::array<int, 3> tg_grid_dims;
    std::vector<int> dim2grid;
    compute_grid(src_layout_, dst_layout_, iter_blocks, loop_blocks, tg_blocks,
            kernel_grid_dims, tg_grid_dims, &dim2grid);

    std::vector<stmt_t> init_stmts;
    init_kernel_grid(kernel_grid_dims, tg_grid_dims, hw_cfg_.simd_size(),
            init_cset, kernel_grid_, tg_grid_, local_id_, init_stmts);

    auto &x = view_t::placeholder_var();

    std::vector<dim_t> vdims(ndims);
    for (int i = 0; i < ndims; i++) {
        vdims[i] = std::max(src_layout_.dim(i), dst_layout_.dim(i));
    }

    view_t src_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = src_layout_.dim(i);
        src_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        src_view.set_tdim(i, vars[i], mask);
    }
    src_view.set_tlayout(src_layout_);

    view_t dst_view(vars, ndims);
    for (int i = 0; i < ndims; i++) {
        int dim = dst_layout_.dim(i);
        dst_view.set_vdim(vars[i], vdims[i]);
        expr_t mask(true);
        if (dim != vdims[i]) mask = x < dim;
        dst_view.set_tdim(i, vars[i], mask);
    }
    dst_view.set_tlayout(dst_layout_);

    gemm_schedule_t schedule(init_cset, kernel_grid_, tg_grid_);

    schedule.set_view(src_view);
    schedule.set_view(dst_view);

    std::array<std::vector<expr_t>, 3> fused_idxs;
    for (int i = 0; i < ndims; i++) {
        std::vector<expr_t> ordered;
        auto v = vars[i];
        if (iter_blocks[i] != 1) {
            expr_t outer, inner;
            schedule.split(v, iter_blocks[i], outer, inner);
            schedule.tensorize(inner);
            v = outer;
            ordered.insert(ordered.begin(), outer);
        }
        if (loop_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, loop_blocks[i], outer, inner);
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        if (tg_blocks[i] != 1) {
            if (!ordered.empty()) ordered.erase(ordered.begin());
            expr_t outer, inner;
            schedule.split(v, tg_blocks[i], outer, inner);
            schedule.bind(inner, tg_grid_.idx(dim2grid[i]));
            v = outer;
            ordered.insert(ordered.begin(), inner);
            ordered.insert(ordered.begin(), outer);
        }
        fused_idxs[dim2grid[i]].push_back(v);
        schedule.reorder(ordered);
    }

    for (int i = 0; i < (int)fused_idxs.size(); i++) {
        auto &vec = fused_idxs[i];
        if (vec.empty()) continue;
        auto var = (vec.size() == 1 ? vec[0] : schedule.fuse(vec));
        schedule.bind(var, kernel_grid_.idx(i));
    }

    schedule.finalize();

    auto thr_tile = schedule.thr_view_tile(src_view, /*is_relative=*/false);

    auto src_thr_view = src_view.create_sub_view(thr_tile);
    auto dst_thr_view = dst_view.create_sub_view(thr_tile);

    auto src_buf = kernel_info_.arg_var(0);
    auto dst_buf = kernel_info_.arg_var(1);

    ir_context_t ir_ctx(hw_cfg_, init_cset);
    auto reg_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "reg");

    std::vector<stmt_t> allocs;
    for (int i = 0; i < kernel_info_.nargs(); i++) {
        auto &var = kernel_info_.arg_var(i);
        if (!var.type().is_ptr()) continue;
        allocs.push_back(alloc_t::make(var, 0, alloc_kind_t::global));
    }

    auto read = make_access_builder(ir_ctx, src_thr_view, src_buf, reg_buf,
            send_op_t::load, send_address_t::a64);
    auto read_stmt = read.stmt();

    auto write = make_access_builder(ir_ctx, dst_thr_view, dst_buf, reg_buf,
            send_op_t::store, send_address_t::a64);
    auto write_stmt = write.stmt();

    auto read_layout = read.reg_layout();
    auto write_layout = write.reg_layout();
    allocs.push_back(
            alloc_t::make(reg_buf, read_layout.size(), alloc_kind_t::grf));

    if (read_layout != write_layout) {
        auto tmp_buf = ir_ctx.create_tmp_var(type_t::byte_ptr(), "tmp");
        allocs.push_back(
                alloc_t::make(tmp_buf, write_layout.size(), alloc_kind_t::grf));

        auto reorder_stmt = create_reorder_stmt(
                read_layout, write_layout, reg_buf, tmp_buf);
        write_stmt = substitute(write_stmt, reg_buf, tmp_buf);
        write_stmt = reorder_stmt.append(write_stmt);
    }

    stmt_ = stmt_t();
    stmt_ = stmt_.append(read_stmt);
    stmt_ = stmt_.append(write_stmt);

    stmt_ = schedule.create_loop_nest(stmt_);
    stmt_ = schedule.create_bind_stmt(stmt_);
    stmt_ = inject_let_stmts(stmt_, init_stmts);
    stmt_ = inject_alloc_stmts(stmt_, allocs);
    stmt_ = inject_external_var_let(stmt_, ir_ctx);

    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = lift_buffer_offsets_in_send(stmt_, ir_ctx);
    stmt_ = inject_send(stmt_, ir_ctx);
    stmt_ = split_wide_stores(stmt_, ir_ctx);
    stmt_ = eliminate_common_subexprs(
            stmt_, ir_ctx, hw_cfg_.regs() * hw_cfg_.grf_size());
    stmt_ = simplify(stmt_, ir_ctx);
    stmt_ = optimize_alloc_let(stmt_, ir_ctx);
    stmt_ = stmt_group_t::make(stmt_label_t::kernel(), stmt_);

    int ir_usage = get_peak_grf_usage(stmt_, hw_cfg_.grf_size());
    int reserved_usage = 16;
    int grf_usage = ir_usage + reserved_usage;
    if (grf_usage > hw_cfg_.regs()) {
        ir_warning()
                << "Estimated GRF usage is " << grf_usage
                << " which exceeds available space, retry with a smaller tile."
                << std::endl;

        return false;
    }

    ir_trace() << "Reorder kernel body:\n" << stmt_ << std::endl;
    return true;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
