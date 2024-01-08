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

#include "gpu/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace v2 {

static bool is_abx_tag(const std::string &s) {
    std::vector<bool> seen('z' - 'a' + 1);
    for (auto c : s) {
        auto c_lower = std::tolower(c);
        if (std::isalpha(c) && c == c_lower && c != 'x') {
            seen[c - 'a'] = true;
        }
    }
    for (size_t i = 0; i < seen.size(); i++) {
        if (seen[i]) continue;
        for (; i < seen.size(); i++) {
            if (seen[i]) return false;
        }
    }
    return true;
}

std::string block_t::brief_str() const {
    std::ostringstream oss;
    oss << "[dim = " << dim;
    oss << ", size = " << size;
    oss << ", stride = " << stride << "]";
    return oss.str();
}

std::string block_t::str() const {
    std::ostringstream oss;
    oss << "block:" << std::endl;
    oss << "  dim:    " << dim << std::endl;
    oss << "  size:   " << size << std::endl;
    oss << "  stride: " << stride;
    return oss.str();
}

void dim_mapper_t::set_dim(const prb_dim_t &dim, const expr_t &expr) {
    exprs_.set(dim, expr.is_empty() ? index_var(dim) : expr);
}

const expr_t &dim_mapper_t::expr(const prb_dim_t &dim) const {
    if (is_empty()) return index_var(dim);
    return exprs_[dim];
}

std::string dim_mapper_t::str() const {
    std::ostringstream oss;
    oss << "dim_mapper:" << std::endl;
    for (auto &dim : exprs_) {
        oss << "  " << dim.str() << " -> ";
        oss << exprs_[dim].str() << std::endl;
    }
    return oss.str();
}

layout_desc_t::layout_desc_t(const dim_map_t<prb_dim_t, char> &letter_map)
    : letter_map_(letter_map) {
    auto append = [&](const prb_dim_t &dim) {
        if (letter_map_.has(dim)) canonical_ += letter_map_[dim];
    };
    append(prb_dims::mb);
    append(prb_dims::g);
    append(prb_dims::oc);
    append(prb_dims::ic);
    append(prb_dims::id);
    append(prb_dims::ih);
    append(prb_dims::iw);
    append(prb_dims::od);
    append(prb_dims::oh);
    append(prb_dims::ow);
    append(prb_dims::kd);
    append(prb_dims::kh);
    append(prb_dims::kw);
}

char layout_desc_t::layout_letter(const prb_dim_t &dim) const {
    if (!letter_map_.has(dim)) return '?';
    return letter_map_.at(dim);
}

prb_dim_t layout_desc_t::prb_dim(int idx) const {
    ir_assert(idx >= 0 && idx < ndims());
    char c = canonical_[idx];
    for (auto &d : letter_map_) {
        if (layout_letter(d) == c) return d;
    }
    ir_error_not_expected();
    return prb_dims::undef;
}

int layout_desc_t::dim_index(const prb_dim_t &dim) const {
    for (int i = 0; i < ndims(); i++) {
        if (canonical_[i] == layout_letter(dim)) return i;
    }
    ir_error_not_expected();
    return -1;
}

std::string layout_desc_t::to_abx_tag(const std::string &tag) const {
    if (is_abx_tag(tag)) return tag;
    const char *tensor_map = canonical().c_str();
    std::string ret;
    for (auto c : tag) {
        if (!std::isalpha(c) || c == 'x') {
            ret += c;
            continue;
        }
        auto c_lower = std::tolower(c);
        bool found = false;
        for (int i = 0; i < (int)std::strlen(tensor_map); i++) {
            if (tensor_map[i] == c_lower) {
                char ab = 'a' + i;
                ret += (c == c_lower ? ab : std::toupper(ab));
                found = true;
                break;
            }
        }
        ir_assert(found);
    }
    return ret;
}

std::string layout_desc_t::str() const {
    std::ostringstream oss;
    oss << "canonical: " << canonical_ << std::endl;
    oss << ir_utils::add_tag("letter_map", letter_map_.str());
    return oss.str();
}

void layout_raw_tag_t::add_entry(char letter, int block, bool is_blocked) {
    entries_.emplace_back(letter, block, is_blocked);
}

int layout_raw_tag_t::entry_index(char letter) {
    for (int i = 0; i < (int)entries_.size(); i++) {
        if (entries_[i].letter == letter) return i;
    }
    ir_error_not_expected();
    return -1;
}

void layout_raw_tag_t::add_dim(char letter, int pos) {
    ir_assert(!has_x());
    std::vector<layout_raw_tag_entry_t> new_entries;
    for (int i = 0; i < (int)entries_.size(); i++) {
        auto &e = entries_[i];
        if (i == pos) new_entries.emplace_back(letter, 0, false);
        char new_letter = e.letter;
        if (new_letter >= letter) new_letter++;
        new_entries.emplace_back(new_letter, e.block, e.is_blocked);
    }
    entries_ = new_entries;
}

void layout_raw_tag_t::remove_dim(char letter) {
    ir_assert(!has_x());
    std::vector<layout_raw_tag_entry_t> new_entries;
    for (auto &e : entries_) {
        if (e.letter == letter) continue;
        char new_letter = e.letter;
        if (e.letter > letter) new_letter--;
        new_entries.emplace_back(new_letter, e.block, e.is_blocked);
    }
    entries_ = new_entries;
}

bool layout_raw_tag_t::is_blocked(char letter) const {
    for (auto &e : entries_) {
        if (e.letter == letter && e.is_blocked) return true;
    }
    return false;
}

int layout_raw_tag_t::ndims() const {
    ir_assert(!is_any());
    int max_index = 0;
    for (auto &e : entries_) {
        max_index = std::max(max_index, e.index());
    }
    return max_index + 1;
}

int layout_raw_tag_t::non_x_ndims() const {
    ir_assert(!is_any());
    std::vector<bool> seen('z' - 'a' + 1);
    for (auto &e : entries_) {
        if (!e.is_x()) seen[e.index()] = true;
    }
    int ret = 0;
    for (auto b : seen)
        if (b) ret++;
    return ret;
}

std::string layout_raw_tag_t::str() const {
    if (is_any()) return "any";
    std::ostringstream oss;
    for (auto &e : entries_)
        oss << e.str();
    return oss.str();
}

bool layout_raw_tag_t::matches(const layout_raw_tag_t &other,
        const layout_desc_t &desc, const prb_tile_t &sizes) const {
    if (is_any()) return true;
    int n0 = nentries();
    int n1 = other.nentries();
    auto skip0 = skip_mask(desc, sizes);
    auto skip1 = other.skip_mask(desc, sizes);
    int i0 = 0;
    int i1 = 0;
    for (;;) {
        while (i0 < n0 && skip0[i0])
            i0++;
        while (i1 < n1 && skip1[i1])
            i1++;
        if (i0 >= n0 || i1 >= n1) break;
        if (entries_[i0] != other.entries_[i1]) return false;
        i0++;
        i1++;
    }
    return i0 == n0 && i1 == n1;
}

void layout_raw_tag_t::init_entries(const std::string &s) {
    ir_assert(is_abx_tag(s)) << s;
    std::vector<bool> is_blocked('z' - 'a' + 1);
    auto letter_blocks = parse_letter_blocks(s);
    for (auto &p : letter_blocks) {
        if (p.second != 0) is_blocked[std::tolower(p.first)] = true;
    }
    for (auto &p : letter_blocks) {
        char letter = std::tolower(p.first);
        entries_.emplace_back();
        auto &e = entries_.back();
        e.letter = letter;
        e.block = p.second;
        e.is_blocked = is_blocked[letter - 'a'];
    }
}

bool layout_raw_tag_t::has_x() const {
    for (auto &e : entries_)
        if (e.is_x()) return true;
    return false;
}

void layout_raw_tag_t::normalize(int ndims) {
    if (!has_x() || ndims == 0) return;
    std::vector<layout_raw_tag_entry_t> new_entries;
    for (auto &e : entries_) {
        if (e.is_x()) {
            for (int i = non_x_ndims(); i < ndims; i++) {
                auto new_e = e;
                new_e.letter = 'a' + i;
                new_entries.push_back(new_e);
            }
        } else {
            new_entries.push_back(e);
        }
    }
    entries_ = new_entries;
}

std::vector<bool> layout_raw_tag_t::skip_mask(
        const layout_desc_t &desc, const prb_tile_t &sizes) const {
    std::vector<bool> ret(nentries());
    auto rem_sizes = sizes;
    for (int i = nentries() - 1; i >= 0; i--) {
        auto &e = entries_[i];
        int idx = e.letter - 'a';
        auto dim = desc.prb_dim(idx);
        ir_assert(sizes.has(dim));
        if (e.block != 0) {
            ir_assert(e.block != 1);
            rem_sizes[dim] = utils::rnd_up(rem_sizes[dim], e.block);
        }
        if (rem_sizes[dim] == 1) ret[i] = true;
        if (e.block != 0) rem_sizes[dim] /= e.block;
    }
    return ret;
}

std::vector<std::pair<char, int>> layout_raw_tag_t::parse_letter_blocks(
        const std::string &tag) {
    std::vector<std::pair<char, int>> ret;
    std::stringstream ss(tag);
    while (!ss.eof()) {
        int next = ss.peek();
        if (ss.eof()) break;
        int block = 0;
        while (std::isdigit(next)) {
            block = 10 * block + (next - '0');
            ss.ignore(1);
            next = ss.peek();
        }
        char letter = char(ss.peek());
        ir_assert(!ss.eof());
        ss.ignore(1);
        ret.emplace_back(letter, block);
    }
    return ret;
}

static void advance(prb_coord_t<int> &idx, const prb_tile_t &bound,
        const prb_tile_t &block) {
    int inc = 1;
    for (auto &d : idx) {
        int inc_idx = (idx[d] / block[d] + inc) % bound[d];
        inc = (idx[d] / block[d] + inc) / bound[d];
        idx[d] = inc_idx * block[d];
        if (inc == 0) break;
    }
}

static void advance(std::vector<int> &idxs, const std::vector<block_t> &blocks,
        const std::vector<int> &block_incs) {
    ir_assert(idxs.size() == blocks.size());
    ir_assert(idxs.size() == block_incs.size());
    for (size_t i = 0; i < idxs.size(); i++) {
        int size = blocks[i].int_size();
        if (idxs[i] + block_incs[i] < size) {
            idxs[i] += block_incs[i];
            break;
        }
        idxs[i] = 0;
    }
}

static inline void advance(
        std::vector<int> &idxs, const std::vector<block_t> &blocks, int inc) {
    for (size_t i = 0; i < idxs.size(); i++) {
        int size = blocks[i].int_size();
        int inc_idx = (idxs[i] + inc) % size;
        inc = (idxs[i] + inc) / size;
        idxs[i] = inc_idx;
        if (inc == 0) break;
    }
}

bool layout_tag_t::matches(
        const layout_tag_t &other, const prb_tile_t &sizes) const {
    return raw_tag().matches(other.raw_tag(), desc_, sizes);
}

std::string layout_tag_t::str() const {
    std::ostringstream oss;
    oss << raw_tag_ << ":" << type_;
    return oss.str();
}

int layout_t::elems() const {
    ir_assert(has_const_sizes());
    int ret = 1;
    for (auto &b : blocks_)
        ret *= b.int_size();
    return ret;
}

int layout_t::size() const {
    ir_assert(has_const_sizes());
    ir_assert(has_const_strides());
    if (is_empty()) return 0;
    int max_off = 0;
    int max_block_size = 0;
    for (auto &b : blocks_) {
        max_off += (b.int_size() - 1) * b.int_stride();
        max_block_size
                = std::max(max_block_size, b.int_size() * b.int_stride());
    }
    int max_off_bytes = (max_off + 1) * type().size();
    return std::max(max_off_bytes, max_block_size * type().size());
}

int layout_t::nblocks(const prb_dim_t &dim) const {
    int ret = 0;
    for (auto &b : blocks_)
        if (b.dim == dim) ret++;
    return ret;
}

int layout_t::int_dim_size(const prb_dim_t &dim) const {
    int ret = 1;
    for (auto &b : blocks_)
        if (b.dim == dim) ret *= b.int_size();
    return ret;
}

bool layout_t::has_const_sizes() const {
    for (auto &b : blocks_)
        if (!b.has_const_size()) return false;
    return true;
}

bool layout_t::has_const_strides() const {
    for (auto &b : blocks_)
        if (!b.has_const_stride()) return false;
    return true;
}

prb_tile_t layout_t::int_dim_sizes() const {
    prb_tile_t ret;
    for (auto &b : blocks_)
        ret[b.dim] = ret.get(b.dim, 1) * b.int_size();
    return ret;
}

dim_map_t<prb_dim_t, expr_t> layout_t::dim_sizes() const {
    dim_map_t<prb_dim_t, expr_t> ret;
    for (auto &b : blocks_)
        ret[b.dim] = ret.get(b.dim, 1) * b.size;
    return ret;
}

int layout_t::inner_block(const prb_dim_t &dim) const {
    int ret = 1;
    for (auto &b : blocks_) {
        if (b.dim == dim && b.has_const_size()) ret *= b.int_size();
    }
    return ret;
}

int layout_t::inner_stride() const {
    if (nblocks() == 0) return 1;
    return blocks_[0].int_stride();
}

expr_t layout_t::stride(const prb_dim_t &dim, int dim_block_idx) const {
    int idx = 0;
    for (auto &b : blocks_) {
        if (b.dim != dim) continue;
        if (idx == dim_block_idx) { return b.stride; }
        idx++;
    }
    return expr_t();
}

expr_t layout_t::offset_in_bytes(const std::vector<int> &block_off) const {
    expr_t ret = 0;
    for (int i = 0; i < nblocks(); i++) {
        auto &b = blocks_[i];
        if (block_off[i] != 0) ret += block_off[i] * b.stride;
    }
    return ret * type_.size();
}

int layout_t::offset_in_bytes(prb_coord_t<int> coord) const {
    ir_assert(has_const_sizes() && has_const_strides());
    ir_assert(has_zero_base());
    int ret = 0;
    for (int i = 0; i < nblocks(); i++) {
        auto &b = blocks_[i];
        int &rem_dim = coord[b.dim];
        ret += (rem_dim % b.int_size()) * b.int_stride();
        rem_dim /= b.int_size();
    }
    return ret * type_.size();
}

bool layout_t::is_blocked_by(const prb_dim_t &dim, int block) const {
    if (block == 1) return true;
    if (nblocks() == 0) return false;
    auto &b = blocks_[0];
    if (b.dim != dim) return false;
    if (!b.has_const_size()) return false;
    return (b.int_size() % block == 0);
}

bool layout_t::is_blocked_by(const layout_t &other) const {
    if (other.is_empty()) return true;

    ir_assert(other.type() == type());
    if (nblocks() < other.nblocks()) return false;

    for (int i = 0; i < other.nblocks(); i++) {
        bool is_last = (i == other.nblocks() - 1);
        auto &b = blocks()[i];
        auto &b_other = other.blocks()[i];
        if (b.dim != b_other.dim) return false;
        if (!b.has_same_stride(b_other)) return false;
        if (is_last && b.has_const_size() && b_other.has_const_size()) {
            if (b.int_size() % b_other.int_size() != 0) return false;
        } else if (!b.size.is_same(b_other.size)) {
            return false;
        }
    }
    return true;
}

void layout_t::add_block(
        const prb_dim_t &dim, const expr_t &size, const expr_t &_stride) {
    if (is_one(size)) return;
    expr_t stride = _stride;
    if (stride.is_empty()) {
        stride = 1;
        if (!blocks_.empty()) {
            auto &last = blocks_.back();
            stride = last.size * last.stride;
            if (stride_pad_ != 1 && stride.is<int_imm_t>()) {
                stride = utils::rnd_up(to_int(stride), stride_pad_);
            }
        }
    }
    blocks_.emplace_back(dim, size, stride);
}

void layout_t::block_by(const block_t &block) {
    ir_assert(has_zero_base());
    ir_assert(has_const_sizes());
    ir_assert(stride_pad_ == 1);
    auto rem_sizes = int_dim_sizes();
    if (!rem_sizes.try_factor(block.dim, block.int_size()))
        ir_error_not_expected();

    auto old_blocks = std::move(blocks_);
    blocks_.clear();
    add_block(block.dim, block.size);
    for (auto &b : old_blocks) {
        int b_size = b.int_size();
        bool ok = rem_sizes.try_factor(b.dim, b_size);
        if (!ok) {
            b_size = math::gcd(b_size, rem_sizes.at(b.dim));
            ok = rem_sizes.try_factor(b.dim, b_size);
        }
        ir_assert(ok);
        if (b_size == 1) continue;
        add_block(b.dim, b_size);
    }
    for (auto &d : rem_sizes)
        ir_assert(rem_sizes.at(d) == 1);
}

void layout_t::normalize() {
    block_t *prev = nullptr;
    expr_t stride = 1;
    bool changed = false;
    for (int i = 0; i < nblocks(); i++) {
        auto &cur = blocks_[i];
        if (prev && cur.dim == prev->dim && cur.stride.is_equal(stride)) {
            prev->size *= cur.size;
            cur.dim = prb_dim_t();
            changed = true;
        } else {
            prev = &cur;
        }
        stride = cur.size * cur.stride;
    }
    if (!changed) return;
    std::vector<block_t> new_blocks;
    new_blocks.reserve(blocks_.size());
    for (auto &b : blocks_) {
        if (!b.dim.is_undef()) new_blocks.push_back(b);
    }
    blocks_ = std::move(new_blocks);
}

layout_t layout_t::split_block(
        const block_t *block_ptr, int inner, int outer) const {
    std::vector<block_t> split_blocks;
    split_blocks.reserve(blocks_.size() + 1);
    for (auto &b : blocks_) {
        if (&b != block_ptr) {
            split_blocks.push_back(b);
            continue;
        }
        ir_assert(b.has_const_size());
        ir_assert(b.int_size() == inner * outer);
        split_blocks.emplace_back(b.dim, inner, b.stride);
        split_blocks.emplace_back(b.dim, outer, inner * b.stride);
    }
    return layout_t(desc(), type(), base(), split_blocks);
}

template <typename T>
struct div_helper_t {
    static T call(const T &a, int b) { return a / b; }
};

template <>
struct div_helper_t<expr_t> {
    static expr_t call(const expr_t &a, int b) { return linear_div(a, b); }
};

template <typename T>
layout_t layout_t::map(const dim_mapper_t &dim_mapper,
        const prb_coord_t<T> &coord, const prb_tile_t &tile) const {
    auto idxs = coord;
    auto rem_sizes = tile;
    idxs.fill_missing(0);
    rem_sizes.fill_missing(1);
    expr_t base = base_;
    expr_t stride = 1;
    std::vector<block_t> mapped_blocks;
    dim_map_t<prb_dim_t, bool> seen_outer;
    for (auto &b : blocks()) {
        auto &expr = dim_mapper.expr(b.dim);
        auto _linear = to_linear(expr);
        auto &linear = _linear.as<linear_t>();
        expr_t off = linear.c;
        for (int i = 0; i < linear.nargs(); i++) {
            auto dim = index_to_prb_dim(linear.v_vec[i]);
            int &cur_size = rem_sizes[dim];
            int mapped_size = cur_size;
            if (b.has_const_size() && cur_size != 1) {
                ir_assert(linear.nargs() == 1);
                int b_size = b.int_size();
                if (cur_size % b_size != 0) {
                    if (b_size % cur_size == 0) {
                        int inner = cur_size;
                        int outer = b_size / cur_size;
                        return split_block(&b, inner, outer)
                                .map(dim_mapper, coord, tile);
                    }
                    return layout_t();
                }
                mapped_size = b_size;
            }
            if (mapped_size != 1) {
                cur_size /= mapped_size;
                auto mapped_stride = linear.u_vec[i] * stride;
                mapped_blocks.emplace_back(dim, mapped_size, mapped_stride);
            }
            bool is_outer = true;
            if (b.has_const_size()) {
                ir_assert(!seen_outer.has(dim));
                int factor = linear_max_pow2_divisor(idxs[dim]);
                if (factor % b.int_size() == 0) {
                    idxs[dim] = div_helper_t<T>::call(idxs[dim], b.int_size());
                    is_outer = false;
                }
            }
            if (is_outer) {
                seen_outer.set(dim, true);
                off += idxs[dim] * linear.u_vec[i];
            }
        }
        base += off * stride;
        stride = b.size * b.stride;
    }
    return layout_t(desc(), type(), base, mapped_blocks);
}

template layout_t layout_t::map<int>(const dim_mapper_t &dim_mapper,
        const prb_coord_t<int> &coord, const prb_tile_t &tile) const;
template layout_t layout_t::map<expr_t>(const dim_mapper_t &dim_mapper,
        const prb_coord_t<expr_t> &coord, const prb_tile_t &tile) const;

prb_coord_t<int> layout_t::to_coord(const std::vector<int> &block_idx) const {
    ir_assert((int)block_idx.size() == nblocks());
    prb_coord_t<int> ret;
    prb_tile_t block_sizes(1);
    for (int i = 0; i < nblocks(); i++) {
        auto &d = blocks_[i].dim;
        auto &blk = block_sizes[d];
        ret[d] = ret.get(d, 0) + block_idx[i] * blk;
        blk *= blocks_[i].int_size();
    }
    return ret;
}

int layout_t::to_linear_index(
        const prb_tile_t &tile, const prb_coord_t<int> &coord) const {
    ir_assert(has_const_sizes());
    std::vector<int> tile_blocks;
    auto rem_tile = tile;
    rem_tile.fill_missing(1);
    for (auto &b : blocks_) {
        int &rem = rem_tile[b.dim];
        int factor = 1;
        if (rem != 1 && b.int_size() != 1) {
            factor = math::gcd(b.int_size(), rem);
            ir_assert(factor == std::min(b.int_size(), rem));
            rem /= factor;
        }
        tile_blocks.push_back(factor);
    }
    for (auto &d : rem_tile)
        ir_assert(rem_tile[d] == 1);
    int ntiles = ir_utils::safe_div(elems(), tile.elems());
    std::vector<int> idx(nblocks());
    for (int i = 0; i < ntiles; i++) {
        auto i_coord = to_coord(idx);
        if (i_coord == coord) return i;
        advance(idx, blocks_, tile_blocks);
    }
    return -1;
}

std::string layout_t::blocks_str() const {
    if (blocks_.empty()) return "(scalar):" + type().str();
    std::string ret;
    expr_t stride(1);
    dim_map_t<prb_dim_t, int> seen;
    for (auto &b : blocks_) {
        std::string b_str;
        char letter = desc_.layout_letter(b.dim);
        if (b.has_const_size()) {
            b_str = std::to_string(b.int_size());
            b_str.append(1, letter);
        } else {
            b_str.append(1, seen[b.dim] ? std::toupper(letter) : letter);
        }
        if (b.has_const_stride() && b.int_stride() != to_int(stride)) {
            b_str.append(1, '*');
        }
        ret = b_str + ret;
        if (b.has_const_size() && b.has_const_stride())
            stride = b.stride * b.size;
        seen[b.dim] = true;
    }
    return ret;
}

std::string layout_t::str() const {
    if (is_empty()) return "(empty)";
    std::ostringstream oss;
    oss << blocks_str();
    oss << ":" + type().str();
    if (!is_zero(base_)) {
        oss << std::endl;
        oss << ir_utils::add_tag("base", base_.str());
    }
    return oss.str();
}

std::string layout_t::str_with_size(const hw_t &hw) const {
    std::ostringstream oss;
    oss << str();
    int regs = (hw.is_undef() ? 0 : utils::div_up(size(), hw.grf_size()));
    oss << " (" << size() << " bytes, ";
    oss << regs << " regs)";
    return oss.str();
}

void for_each(const prb_tile_t &base_tile, prb_tile_t tile,
        const std::function<void(const prb_coord_t<int> &)> &func) {
    for (auto &d : tile) {
        ir_assert(base_tile.has(d));
        ir_assert(base_tile[d] % tile[d] == 0);
    }

    prb_coord_t<int> idx;
    prb_tile_t bound;
    int ntiles = 1;
    for (auto &d : base_tile) {
        if (!tile.has(d)) tile[d] = 1;
        idx[d] = 0;
        bound[d] = ir_utils::safe_div(base_tile[d], tile[d]);
        ntiles *= bound[d];
    }
    for (int i = 0; i < ntiles; i++) {
        func(idx);
        advance(idx, bound, tile);
    }
}
block_iterator_t::block_iterator_t(const layout_t &layout, bool set_to_end)
    : parent_(&layout), block_idx_(set_to_end ? parent_->nblocks() : 0) {
    ir_assert(layout.has_const_sizes());
    if (is_end()) return;
    block_ = parent_->blocks().front();
    block_.size = 1;
}

block_iterator_t &block_iterator_t::operator++() {
    if (!has_next()) {
        set_to_end();
        return *this;
    }
    int factor = next_factor();
    if (factor != -1) {
        elems_ /= block_.int_size();
        elems_ *= factor;
        block_.size = factor;
        return *this;
    }
    block_idx_++;
    block_ = parent_->blocks()[block_idx_];
    factor = next_factor(/*is_first=*/true);
    block_.size = factor;
    elems_ *= factor;
    return *this;
}

block_t block_iterator_t::remaining_block() const {
    ir_assert(!is_end());
    auto &b = parent_->blocks()[block_idx_];
    int size = b.int_size() / block_.int_size();
    auto stride = block_.stride * block_.size;
    return block_t(b.dim, size, stride);
}

bool block_iterator_t::is_dense(const prover_t &prover) const {
    if (is_end()) return false;
    expr_t stride = 1;
    for (int i = 0; i < block_idx_; i++) {
        auto &b = parent_->blocks()[i];
        if (!prover.prove(b.stride == stride)) return false;
        stride = b.int_size() * b.stride;
    }
    return prover.prove(block_.stride == stride);
}

int block_iterator_t::elems(const prb_dim_t &dim) const {
    if (dim.is_undef()) return elems_;
    int ret = 1;
    auto &blocks = parent_->blocks();
    for (int i = 0; i < block_idx_; i++) {
        if (blocks[i].dim == dim) ret *= blocks[i].int_size();
    }
    if (block_.dim == dim) ret *= block_.int_size();
    return ret;
}

layout_t block_iterator_t::sub_layout() const {
    layout_t ret(parent_->desc(), parent_->type());
    for (int i = 0; i < block_idx_; i++) {
        ret.add_block(parent_->blocks()[i].dim, parent_->blocks()[i].size);
    }
    if (!block_.is_empty()) ret.add_block(block_.dim, block_.size);
    return ret;
}

std::string block_iterator_t::str() const {
    std::ostringstream oss;
    oss << "block_idx: " << block_idx_ << std::endl;
    oss << "block:     " << block_.brief_str();
    return ir_utils::add_tag("block_iterator", oss.str());
}

void block_iterator_t::set_to_end() {
    block_idx_ = parent_->nblocks();
    block_ = block_t();
    elems_ = parent_->elems();
}

int block_iterator_t::next_factor(bool is_first) const {
    if (is_end()) return -1;

    auto &b = parent_->blocks()[block_idx_];
    int size = b.int_size();
    int start = (is_first ? 2 : block_.int_size() + 1);
    for (int i = start; i <= size; i++) {
        if (size % i == 0) { return i; }
    }
    return -1;
}

void add_remaining_blocks(layout_t &layout, const block_iterator_t &it) {
    if (it.is_end()) return;
    auto rem_block = it.remaining_block();
    layout.add_block(rem_block.dim, rem_block.size);
    auto &parent = it.parent();
    for (int i = it.block_index() + 1; i < parent.nblocks(); i++) {
        auto &b = parent.blocks()[i];
        layout.add_block(b.dim, b.size);
    }
}

layout_iterator_t::layout_iterator_t(const layout_t &layout, bool is_end)
    : parent_(&layout)
    , total_elems_(parent_->elems())
    , offset_(is_end ? total_elems_ : 0)
    , block_off_(parent_->nblocks()) {}

void layout_iterator_t::next(int elems) {
    if (!has_next(elems)) {
        set_to_end();
        return;
    }
    advance(block_off_, parent_->blocks(), elems);
    offset_ += elems;
}

int layout_iterator_t::offset(const prb_dim_t &dim) const {
    int ret = 1;
    int stride = 1;
    for (int i = 0; i < parent_->nblocks(); i++) {
        auto &b = parent_->blocks()[i];
        if (b.dim == dim) { ret += stride * block_off_[i]; }
        stride *= b.int_size();
    }
    return ret;
}

prb_coord_t<int> layout_iterator_t::coord() const {
    prb_coord_t<int> ret;
    prb_tile_t sizes;
    for (int i = 0; i < parent_->nblocks(); i++) {
        auto &b = parent_->blocks()[i];
        ret[b.dim] = ret.get(b.dim, 0) + block_off_[i] * sizes.get(b.dim, 1);
        sizes[b.dim] = sizes.get(b.dim, 1) * b.int_size();
    }
    return ret;
}

std::string layout_iterator_t::str() const {
    using namespace ir_utils;
    std::ostringstream oss;
    oss << "offset:    " << offset_ << std::endl;
    oss << "block_off: " << block_off_;
    return ir_utils::add_tag("layout_iterator", oss.str());
}

dim_mask_desc_t::dim_mask_desc_t(const prb_dim_t &dim, const expr_t &expr,
        const expr_t &bound, int block, bool do_zero_cmp)
    : dim(dim)
    , expr(expr)
    , bound(bound)
    , block(block)
    , do_zero_cmp(do_zero_cmp)
    , base(0) {
    ir_assert(math::is_pow2(block));
    init_abc_xy(expr);
}

template <typename T>
expr_t dim_mask_desc_t::to_expr(
        const prb_coord_t<T> &coord, bool with_const) const {
    expr_t ret = (with_const ? c : 0);
    if (coord.has(x_dim)) ret += a * coord[x_dim];
    if (!y_dim.is_undef() && coord.has(y_dim)) ret += b * coord[y_dim];
    return ret;
}

template expr_t dim_mask_desc_t::to_expr(
        const prb_coord_t<expr_t> &coord, bool with_const) const;
template expr_t dim_mask_desc_t::to_expr(
        const prb_coord_t<int> &coord, bool with_const) const;

dim_mask_desc_t dim_mask_desc_t::map(const prb_coord_t<expr_t> &coord) const {
    auto ret = *this;
    ret.base = to_expr(coord);
    if (!is_identity()) return ret;
    int x_div = linear_max_pow2_divisor(coord[x_dim]);
    ret.block = math::gcd(block, x_div);
    return ret;
}

bool dim_mask_desc_t::has(const prb_dim_t &dim) const {
    return utils::one_of(dim, x_dim, y_dim);
}

expr_t dim_mask_desc_t::dim_stride(const prb_dim_t &dim) const {
    if (dim == x_dim) return a;
    if (dim == y_dim) return b;
    return expr_t(0);
}

std::string dim_mask_desc_t::str() const {
    std::ostringstream oss;
    oss << expr << " < " << bound << " (zero_cmp: " << do_zero_cmp << ")"
        << std::endl;
    oss << "base:  " << base << std::endl;
    oss << "block: " << block;
    return oss.str();
}

void dim_mask_desc_t::init_abc_xy(const expr_t &expr) {
    auto _linear = to_linear(expr);
    auto &linear = _linear.as<linear_t>();
    c = linear.c;
    a = linear.u_vec[0];
    x = linear.v_vec[0];
    if (linear.nargs() > 1) {
        b = linear.u_vec[1];
        y = linear.v_vec[1];
    }
    x_dim = index_to_prb_dim(x);
    y_dim = index_to_prb_dim(y);
}

mask_desc_t::mask_desc_t(
        const dim_mapper_t &dim_mapper, const layout_t &layout) {
    auto dim_sizes = layout.dim_sizes();
    for (auto &d : dim_sizes) {
        auto &expr = dim_mapper.expr(d);
        int block = layout.inner_block(d);
        if (block == 1) {
            const int large_pow2 = (1 << 10);
            block = large_pow2;
        }
        bool do_zero_cmp
                = utils::one_of(d, prb_dims::id, prb_dims::ih, prb_dims::iw);
        dim_masks_.emplace_back(d, expr, dim_sizes[d], block, do_zero_cmp);
    }
}

const dim_mask_desc_t &mask_desc_t::operator[](int idx) const {
    ir_assert(idx >= 0 && idx < nmasks());
    return dim_masks_[idx];
}

mask_desc_t mask_desc_t::map(const prb_coord_t<expr_t> &coord) const {
    auto ret = *this;
    for (auto &dm : ret.dim_masks_)
        dm = dm.map(coord);
    return ret;
}

bool mask_desc_t::is_uniform(
        const block_iterator_t &it, const prover_t &prover) const {
    for (auto &dm : dim_masks_) {
        if (!dm.has((*it).dim)) continue;
        if (!dm.is_identity()) return false;
        int dim_size = it.elems((*it).dim);
        int dim_size_pow2 = utils::rnd_up_pow2(dim_size);
        if (dim_size > dm.block) return false;
        if (!prover.prove(dm.bound % dim_size_pow2 == 0)) return false;
    }
    return true;
}

std::string mask_desc_t::str() const {
    std::ostringstream oss;
    for (int i = 0; i < nmasks(); i++) {
        if (i != 0) oss << std::endl;
        auto tag = "#" + std::to_string(i);
        oss << ir_utils::add_tag(tag, dim_masks_[i].str());
    }
    return oss.str();
}

plane_t::plane_t(const layout_t &layout, const mask_desc_t &mask_desc) {
    type = layout.type();
    const block_t *w_block = nullptr;
    const block_t *h_block = nullptr;
    for (auto &b : layout.blocks()) {
        if (b.has_const_size() && b.int_size() == 1) continue;
        if (!w_block) {
            // Width dimension must be unit-strided.
            if (!is_one(b.stride)) return;
            w_block = &b;
            continue;
        }
        if (!h_block) {
            h_block = &b;
            continue;
        }
        break;
    }
    if (!w_block || !h_block) return;
    if (!w_block->has_const_size()) return;
    if (!h_block->has_const_size()) return;

    w_dim = w_block->dim;
    h_dim = h_block->dim;
    w = w_block->int_size();
    h = h_block->int_size();

    if (layout.nblocks(w_dim) != 1) return;
    if (layout.nblocks(h_dim) != 1) return;

    const dim_mask_desc_t *x_mask_desc = nullptr;
    const dim_mask_desc_t *y_mask_desc = nullptr;
    for (int i = 0; i < mask_desc.nmasks(); i++) {
        auto &dmd = mask_desc[i];
        if (dmd.has(w_dim)) {
            if (x_mask_desc) return;
            x_mask_desc = &dmd;
        }
        if (dmd.has(h_dim)) {
            if (y_mask_desc) return;
            y_mask_desc = &dmd;
        }
    }
    if (!is_one(x_mask_desc->dim_stride(w_dim))) return;

    y_stride = y_mask_desc->dim_stride(h_dim);
    if (!y_stride.is<int_imm_t>() && !y_stride.is<const_var_t>()) return;

    x_dim = x_mask_desc->dim;
    y_dim = y_mask_desc->dim;
    x = x_mask_desc->base;
    y = y_mask_desc->base;
    W = x_mask_desc->bound;
    H = y_mask_desc->bound;
    P = layout.stride(h_dim, 0);

    is_valid = true;
}

view_t::view_t(const dim_mapper_t &dim_mapper, const layout_t &base_layout,
        const prb_coord_t<expr_t> &coord, const prb_tile_t &tile)
    : dim_mapper_(dim_mapper)
    , base_layout_(base_layout)
    , coord_(coord)
    , tile_(tile) {
    mask_desc_t base_mask_desc(dim_mapper, base_layout);
    layout_ = base_layout.map(dim_mapper, coord, tile);
    mask_desc_ = base_mask_desc.map(coord);
    plane_ = plane_t(layout_, mask_desc_);
}

std::string view_t::str() const {
    std::ostringstream oss;
    oss << ir_utils::add_tag("coord", coord_.str()) << std::endl;
    oss << "tile: " << tile_ << std::endl;
    oss << ir_utils::add_tag("layout", layout_.str()) << std::endl;
    oss << ir_utils::add_tag("mask_desc", mask_desc_.str());
    return oss.str();
}

} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
