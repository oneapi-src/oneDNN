/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#include <cctype>
#include <sstream>
#include <thread>

#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

layout_t::layout_t(const type_t &type, const expr_t &offset, int ndims,
        const std::vector<std::pair<int, dim_t>> &parts,
        const std::vector<dim_t> &dims, bool do_normalize)
    : type_(type), ndims_(ndims), offset_(offset) {
    if (!dims.empty() && ndims_ != int(dims.size())) {
        ir_error_not_expected() << "Format and dimensions do not match.";
    }
    for (auto &p : parts) {
        int dim_idx = p.first;
        dim_t block = p.second;
        ir_assert(dim_idx < ndims_);
        if (block == 0 && dims.empty())
            ir_error_not_expected()
                    << "Dimensions are missing. Can't deduce them from "
                       "the format.";
    }

    dim_t stride = 1;
    // Iterate from right to left (innermost to outermost).
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        int dim_idx = it->first;
        dim_t block = it->second;
        if (block == 0) {
            dim_t full_block = 1;
            for (auto &b : blocks_)
                if (b.dim_idx == dim_idx) full_block *= b.block;

            block = utils::div_up(dims[dim_idx], full_block);
        }

        blocks_.emplace_back(dim_idx, block, stride);
        stride = block * stride;
    }

    if (do_normalize) blocks_ = normalize_blocks(blocks_);
    sanity_check();
}

layout_t::layout_t(const memory_desc_wrapper &mdw, bool do_normalize)
    : type_(mdw.data_type()), offset_(mdw.offset0()) {
    ir_assert(mdw.is_blocking_desc()) << "Expected blocking memory descriptor.";

    ndims_ = mdw.ndims();
    blocks_ = compute_block_structure(
            mdw, /* inner_only */ false, /* do_normalize */ do_normalize);

    sanity_check();
}

memory_desc_t layout_t::to_dnnl(const dim_t *dims_hint) const {
    memory_desc_t md = {};
    md.ndims = ndims();
    std::copy(dims_hint, dims_hint + ndims(), md.dims);
    md.data_type = jit::to_dnnl(type_);
    md.offset0 = to_cpp<dim_t>(offset_);
    md.format_kind = format_kind::blocked;

    auto &blk = md.format_desc.blocking;
    bool seen[DNNL_MAX_NDIMS] = {};

    bool in_inner_block = false;
    dim_t prev_stride = 0;

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        auto &b = *it;
        if (!seen[b.dim_idx]) {
            // Outer block.
            ir_assert(!in_inner_block);
            MAYBE_UNUSED(in_inner_block);
            blk.strides[b.dim_idx] = b.stride;
            md.padded_dims[b.dim_idx] = b.block;
        } else {
            // Inner block.
            md.padded_dims[b.dim_idx] *= b.block;
            blk.inner_idxs[blk.inner_nblks] = b.dim_idx;
            blk.inner_blks[blk.inner_nblks] = b.block;
            blk.inner_nblks++;
            if (prev_stride > 0) {
                // Inner block must be dense.
                ir_assert(prev_stride == b.block * dim_t(b.stride));
            }
            prev_stride = b.stride;
            in_inner_block = true;
        }
        seen[b.dim_idx] = true;
    }

    return md;
}

layout_t layout_t::map(const tensor_t &tensor) const {
    if (ndims() != tensor.ndims())
        ir_error_not_expected() << "Dimensions do not match.";

    std::vector<dim_t> remaining_dims = tensor.dims();
    std::vector<block_t> mapped_blocks;

    for (auto &eb : enumerated_blocks()) {
        block_t &b = eb.second;
        bool b_is_outermost = is_outermost(eb);

        dim_t block = b.block;
        dim_t &rem_dim = remaining_dims[b.dim_idx];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.dim_idx, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            block = rem_dim;
        } else if (rem_dim % block != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (block % rem_dim == 0)
                return split_block(eb, rem_dim, block / rem_dim).map(tensor);

            // TODO: Remove exception usage.
            ir_except_not_implemented("Can't map tensor layout.");
        }
        rem_dim /= block;
        mapped_blocks.emplace_back(b.dim_idx, block, b.stride);
    }

    for (auto &d : remaining_dims) {
        // TODO: Remove exception usage.
        if (d != 1) ir_except_not_implemented("Can't map tensor layout.");
        MAYBE_UNUSED(d);
    }

    return layout_t(type(), ndims(), operator()(tensor.start()), mapped_blocks);
}

layout_t layout_t::reinterpret(
        const type_t &new_type, bool do_normalize) const {
    int old_size = type().size();
    int new_size = new_type.size();
    if (new_size == old_size) return *this;

    expr_t new_offset = 0;
    if (!has_zero_offset()) {
        ir_assert(is_const(offset_)) << "Expected constant offset.";
        int64_t off = to_cpp<int64_t>(offset_) * old_size;
        ir_assert(off % new_size == 0);
        new_offset = off / new_size;
    }

    if (old_size % new_size != 0 && new_size % old_size != 0) {
        ir_error_not_expected();
        return layout_t();
    }

    auto new_blocks = blocks_;
    if (new_blocks.empty()) {
        ir_error_not_expected() << "Can't reinterpret.";
        return layout_t();
    }

    auto &b0 = new_blocks.front();
    if (dim_t(b0.stride) != 1) {
        ir_error_not_expected();
        return layout_t();
    }

    if (new_size < old_size) {
        int factor = (old_size / new_size);
        b0.block *= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            b.stride *= factor;
        }
    } else {
        int factor = (new_size / old_size);
        if (b0.block % factor != 0) {
            ir_error_not_expected();
            return layout_t();
        }
        b0.block /= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            if (b.stride % factor != 0) {
                ir_error_not_expected();
                return layout_t();
            }
            b.stride /= factor;
        }
    }

    return layout_t(new_type, ndims(), new_offset, new_blocks, do_normalize);
}

layout_t layout_t::split_block(
        const std::pair<int, block_t> &eb, dim_t block0, dim_t block1) const {
    int block_idx = eb.first;
    auto &b = eb.second;
    ir_assert(b.block == block0 * block1) << "Incompatible block sizes.";
    MAYBE_UNUSED(b);

    auto new_blocks = blocks_;

    block_t &b0 = new_blocks[block_idx];
    block_t b1 = b0;

    b0.block = block0;
    b1.block = block1;
    b1.stride = b0.stride * block0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return layout_t(
            type(), ndims(), offset(), new_blocks, /*do_normalize=*/false);
}

layout_t layout_t::split_into_multi_blocks(
        const std::vector<dim_t> &multi_blocks) const {
    if (is_empty()) return *this;

    layout_t tmp(*this);
    std::vector<dim_t> rem_elems = multi_blocks;
    std::vector<dim_t> cur_elems(rem_elems.size(), 1);
    for (auto &eb : tmp.enumerated_blocks()) {
        auto &b = eb.second;
        for (int i = 0; i < int(rem_elems.size()); i++) {
            auto &e = rem_elems[i];
            if (e == 1) continue;
            if (b.block > e) {
                // Try to split this block.
                int next_block = utils::max_div(b.block, e);
                if (next_block == 1) return layout_t();
                return tmp.split_block(eb, next_block, b.block / next_block)
                        .split_into_multi_blocks(multi_blocks);
            }
            if (e % b.block != 0) return layout_t();
            e /= b.block;
            cur_elems[i] *= b.block;
            break;
        }
    }
    for (int i = 0; i < int(cur_elems.size()); i++) {
        if (cur_elems[i] != multi_blocks[i]) { return layout_t(); }
    }
    return tmp;
}

tensor_t layout_t::split_into_max_tile(
        dim_t max_tile_elems, bool is_dense_tile) const {
    stride_t dense_stride = 1;
    std::vector<dim_t> tile_dims(ndims(), 1);
    dim_t cur_elems = 1;
    for (auto &eb : enumerated_blocks()) {
        auto &b = eb.second;
        if (b.block == 1) continue;
        if (b.block * cur_elems <= max_tile_elems) {
            if (is_dense_tile) {
                if (b.stride.is_unknown()) break;
                if (dense_stride != b.stride) break;
                dense_stride = b.block * b.stride;
            }
            cur_elems *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            continue;
        }
        dim_t max_block = utils::max_div(b.block, max_tile_elems / cur_elems);
        if (max_block == 1) break;
        auto tmp_layout = split_block(eb, max_block, b.block / max_block);
        return tmp_layout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }
    return tensor_t(tile_dims);
}

void layout_t::align_layouts(layout_t &a, layout_t &b) {
    for (int i = 0; i < a.ndims(); i++) {
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        int a_max = int(a_blocks.size());
        int b_max = int(b_blocks.size());
        int a_idx = 0;
        int b_idx = 0;

        for (;;) {
            while (a_idx < a_max && a_blocks[a_idx].dim_idx != i)
                a_idx++;
            while (b_idx < b_max && b_blocks[b_idx].dim_idx != i)
                b_idx++;

            if (a_idx >= a_max || b_idx >= b_max) break;

            auto &ab = a_blocks[a_idx];
            auto &bb = b_blocks[b_idx];
            dim_t common_block = math::gcd(ab.block, bb.block);
            if (ab.block == common_block && bb.block == common_block) {
                a_idx++;
                b_idx++;
                continue;
            }

            if (ab.block != common_block) {
                a = a.split_block(
                        {a_idx, ab}, common_block, ab.block / common_block);
            }
            if (bb.block != common_block) {
                b = b.split_block(
                        {b_idx, bb}, common_block, bb.block / common_block);
            }
            break;
        }
    }
}

std::vector<std::pair<char, dim_t>> layout_t::parse_letter_blocks(
        const std::string &format) {
    std::vector<std::pair<char, dim_t>> ret;

    std::stringstream ss(format);
    while (!ss.eof()) {
        int next = ss.peek();
        if (ss.eof()) break;
        dim_t block = 0;
        while (std::isdigit(next)) {
            block = 10 * block + (next - '0');
            ss.ignore(1);
            next = ss.peek();
        }
        char letter = char(ss.peek());
        ir_assert(!ss.eof()) << "EOF is unexpected.";
        ss.ignore(1);
        ret.emplace_back(letter, block);
    }
    return ret;
}

std::vector<std::pair<int, dim_t>> layout_t::parse_format(
        const std::string &format, int ndims_hint) {
    bool seen_letters[DNNL_MAX_NDIMS] = {};
    int letter_ndims = 0;
    for (char c = 'a'; c < 'a' + DNNL_MAX_NDIMS; c++) {
        if (format.find(c) != std::string::npos) {
            seen_letters[c - 'a'] = true;
            MAYBE_UNUSED(seen_letters);
            letter_ndims++;
        }
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        ir_assert(seen_letters[i] == (i < letter_ndims));
    }

    auto letter_blocks = parse_letter_blocks(format);

    std::vector<std::pair<int, dim_t>> parts;
    for (auto &p : letter_blocks) {
        char letter = p.first;
        dim_t block = p.second;
        if (letter != 'x') {
            int dim_idx = std::tolower(letter) - 'a';
            parts.emplace_back(dim_idx, block);
        } else {
            ir_assert(ndims_hint >= letter_ndims);
            for (int i = letter_ndims; i < ndims_hint; i++) {
                parts.emplace_back(i, 0);
            }
        }
    }

    return parts;
}

void layout_t::sanity_check() const {
#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
    return;
#endif
    if (is_empty()) return;

    for (auto &b : blocks_) {
        ir_assert(b.block > 0) << "Incorrect block size.";
        MAYBE_UNUSED(b);
    }
    ir_assert(ndims_ <= max_ndims);
}

expr_t grid_splitter_t::pop_block(int size) {
    ir_assert(size > 1);
    ir_assert(can_pop_block(size));

    int new_stride = cur_stride_ * size;

    auto idx_expr = grid_.idx(cur_idx_);
    if (cur_stride_ != 1) idx_expr /= cur_stride_;
    if (new_stride != grid_.dim(cur_idx_)) idx_expr %= size;

    cur_stride_ = new_stride;
    if (cur_stride_ == grid_.dim(cur_idx_)) {
        // Move to the next dimension.
        cur_idx_--;
        skip_size_1_dims();
        cur_stride_ = 1;
    }
    return idx_expr;
}

stride_t tdim_t::compute_stride(const expr_t &e, int idx, const expr_t &var) {
    // e == var -> fixed stride.
    if (e.is_same(var)) return stride_t(1);

    auto e0 = e;
    auto e1 = substitute(e, var, var + 1);
    auto e_stride = simplify(e1 - e0);

    if (is_const(e_stride)) return stride_t(to_cpp<dim_t>(e_stride));

    // Stride is not a constant.
    return stride_t::unknown();
}

view_t view_t::create_sub_view(const tensor_t &sub_tensor) const {
    ir_assert(sub_tensor.ndims() == nvdims()) << "Dimensions don't match.";

    auto ret = *this;
    ret.vdims_ = sub_tensor.dims();
    for (int i = 0; i < nvdims(); i++) {
        auto &i_start = sub_tensor.start()[i];
        if (is_zero(i_start)) continue;
        auto &s = ret.vstart_[i];
        s += i_start;
        s = simplify(s);
    }
    return ret;
}

view_t view_t::substitute(const expr_t &from, const expr_t &to) const {
    view_t ret = *this;
    for (int i = 0; i < nvdims(); i++) {
        ret.vstart_[i] = jit::substitute(ret.vstart_[i], from, to);
        ret.vstart_[i] = simplify(ret.vstart_[i]);
    }
    return ret;
}

std::vector<expr_t> view_t::create_vvars(int nvdims) {
    static const int max_nvdims = 128;
    static thread_local std::vector<expr_t> _vvars([] {
        std::vector<expr_t> ret;
        ret.reserve(max_nvdims);
        for (int i = 0; i < max_nvdims; i++)
            ret.push_back(var_t::make(type_t::s32(), "_" + std::to_string(i)));
        return ret;
    }());

    ir_assert(nvdims <= max_nvdims) << "Too many dimensions: " << nvdims;
    return std::vector<expr_t>(_vvars.begin(), _vvars.begin() + nvdims);
}

layout_t view_t::create_pseudo_vlayout(
        const layout_t &tlayout, bool init_offset) const {
    ir_assert(!tlayout.is_empty());

    std::vector<dim_t> rem_vdims = vdims_;
    std::vector<block_t> blocks;

    for (auto &teb : tlayout.enumerated_blocks()) {
        block_t &tb = teb.second;
        bool tb_is_outermost = tlayout.is_outermost(teb);
        dim_t tblock = tb.block;

        auto &tinfo = tdims_[tb.dim_idx];
        if (tb_is_outermost) {
            // Use innermost dimension with maximum remaining size for first
            // block
            int max_idx = -1;
            int max_vidx = -1;
            int max_vdim = 1;
            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                int vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] > max_vdim) {
                    max_idx = i;
                    max_vidx = vidx;
                    max_vdim = rem_vdims[vidx];
                }
            }

            if (max_vdim > 1) {
                stride_t stride = tinfo.vstride(max_idx);
                blocks.emplace_back(
                        max_vidx, max_vdim, stride * stride_t(tb.stride));
                rem_vdims[max_vidx] = 1;
            }

            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                int vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] == 1) continue;

                stride_t stride = tinfo.vstride(i) * tb.stride;
                blocks.emplace_back(vidx, rem_vdims[vidx], stride);
                rem_vdims[vidx] = 1;
            }
            continue;
        }

        ir_assert(tinfo.is_identity()) << "Can't create pseudo-layout.";

        int vidx = tinfo.vidx(0);
        dim_t &rem_vdim = rem_vdims[vidx];
        if (rem_vdim == 1) continue;

        if (rem_vdim % tblock == 0) {
            rem_vdim /= tblock;
        } else if (rem_vdim % tblock != 0) {
            // Try to split the current block and start from scratch.
            if (tblock % rem_vdim == 0) {
                auto tmp_layout
                        = tlayout.split_block(teb, rem_vdim, tblock / rem_vdim);
                return create_pseudo_vlayout(tmp_layout, init_offset);
            }

            // TODO: Remove exception usage.
            ir_except_not_implemented("Can't create pseudo-layout.");
        }
        blocks.emplace_back(tb.dim_idx, tblock, tb.stride);
    }

    for (auto &d : rem_vdims) {
        ir_assert(d == 1) << "Can't create pseudo-layout.";
        MAYBE_UNUSED(d);
    }

    layout_t ret(tlayout.type(), nvdims(), 0, blocks);
    if (!init_offset) return ret;

    auto targs = cvt_vargs_to_targs();
    auto off = tlayout.offset(targs);
    return layout_t(tlayout.type(), nvdims(), off, blocks);
}

layout_t dim_assignment_t::map(const layout_t &layout) const {
    std::vector<block_t> new_blocks;
    for (auto &b : layout.blocks()) {
        int new_idx = assignments_[b.dim_idx];
        if (new_idx == -1) continue; // Drop this block.
        auto new_b = b;
        new_b.dim_idx = new_idx;
        new_blocks.push_back(new_b);
    }
    new_blocks = normalize_blocks(new_blocks,
            /*remove_size_1_blocks=*/false);
    auto ret = layout_t(layout.type(), new_ndims(), layout.offset(), new_blocks,
            /*do_normalize=*/false);
    ir_assert(layout.elems() == ret.elems())
            << "Assignment doesn't preserve number of elements.";
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
