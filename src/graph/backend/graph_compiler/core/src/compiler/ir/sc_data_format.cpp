/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <algorithm>
#include "sc_data_format.hpp"
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/simple_licm.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

sc_data_format_kind_t::sc_data_format_kind_t(
        const std::vector<int> &storage_args) {
    COMPILE_ASSERT(storage_args.size() <= sc_data_format_kind_t::MAX_DIMS,
            "storage size should be less than MAX_DIMS");
    uint64_t res = set_ith_int(
            0xffffffffffffffff, sc_data_format_kind_t::MAX_DIMS, 0);
    for (size_t i = 0; i < storage_args.size(); ++i) {
        res = set_ith_int(res, i, storage_args[i]);
    }
    storage_ = res;
}

sc_data_format_kind_t sc_data_format_kind_t::get_plain_by_dims(size_t ndims) {
    COMPILE_ASSERT(ndims <= sc_data_format_kind_t::MAX_DIMS,
            "storage size should be less than MAX_DIMS");
    uint64_t res = set_ith_int(
            0xffffffffffffffff, sc_data_format_kind_t::MAX_DIMS, 0);
    for (size_t i = 0; i < ndims; ++i) {
        res = set_ith_int(res, i, i);
    }
    return sc_data_format_kind_t(res);
}

sc_data_format_kind_t sc_data_format_kind_t::get_2dblocking_by_dims(
        size_t ndims, bool is_weight, bool is_vnni_format) {
    COMPILE_ASSERT(ndims <= sc_data_format_kind_t::MAX_DIMS,
            "storage size should be less than MAX_DIMS");
    uint64_t res = set_ith_int(
            0xffffffffffffffff, sc_data_format_kind_t::MAX_DIMS, 0);
    for (size_t i = 0; i < ndims; ++i) {
        res = set_ith_int(res, i, i);
    }
    if (ndims == 1) {
        res = set_ith_int(res, ndims, ndims - 1);
    } else {
        // the first blocking
        res = set_ith_int(res, ndims, ndims - 2);
        // the second blocking
        res = set_ith_int(res, ndims + 1, ndims - 1);
        if (is_weight) {
            res = set_ith_int(res, ndims - 2, ndims - 1);
            res = set_ith_int(res, ndims - 1, ndims - 2);
        }
    }
    if (is_vnni_format) {
        assert(ndims >= 2);
        // the third blocking
        res = set_ith_int(res, ndims + 2, ndims - 2);
    }
    return sc_data_format_kind_t(res);
}

int sc_data_format_kind_t::ndims() const {
    for (int i = 0; i < MAX_DIMS; i++) {
        if (get(i) == UNDEF_DIM) { return i; }
    }
    return -1;
}

std::vector<int> sc_data_format_kind_t::collect_blocking_index(int axis) const {
    std::vector<int> index;
    int cur_idx = 0;
    int blocking_count[MAX_DIMS] = {0};
    for (int i = 0; i < MAX_DIMS; i++) {
        auto cur_axis = get(i);
        if (cur_axis == UNDEF_DIM) { return index; }
        blocking_count[cur_axis]++;
        if (blocking_count[cur_axis] > 1) {
            if (cur_axis == axis) { index.push_back(cur_idx); }
            cur_idx++;
        }
    }
    return index;
}

std::vector<std::vector<int>>
sc_data_format_kind_t::collect_p2b_mapping() const {
    std::vector<std::vector<int>> dist(norig_dims(), std::vector<int> {});
    for (int i = 0; i < MAX_DIMS; i++) {
        auto orig_axis = get(i);
        if (orig_axis == UNDEF_DIM) { return dist; }
        dist[orig_axis].emplace_back(i);
    }
    return dist;
}

void sc_data_format_kind_t::collect_dim_count(
        int out[sc_data_format_kind_t::MAX_DIMS]) const {
    for (int i = 0; i < MAX_DIMS; i++) {
        int orig_idx = get(i);
        if (orig_idx != UNDEF_DIM) {
            ++out[orig_idx];
        } else {
            break;
        }
    }
}

int sc_data_format_kind_t::norig_dims() const {
    int ret = -1;
    for (int i = 0; i < MAX_DIMS; i++) {
        int orig_idx = get(i);
        if (orig_idx != UNDEF_DIM) {
            ret = std::max(orig_idx, ret);
        } else {
            break;
        }
    }
    return ret + 1;
}

bool sc_data_format_t::is_convertible(const sc_data_format_t &other) const {
    if (format_code_ == format_kinds::any
            || other.format_code_ == format_kinds::any) {
        return true;
    }
    return format_code_.norig_dims() == other.format_code_.norig_dims();
}

bool sc_data_format_kind_t::is_channel_last() const {
    int i = 0;
    for (; i < MAX_DIMS - 1; i++) {
        if (get(i + 1) == UNDEF_DIM) {
            break;
        } else if ((i == 0 && get(i) != 0) || (i != 0 && get(i) != i + 1)) {
            return false;
        }
    }
    if (!i) return false;
    return get(i) == 1;
}

bool sc_data_format_kind_t::is_plain() const {
    int i = 0;
    for (; i < MAX_DIMS; i++) {
        if (get(i) == UNDEF_DIM) {
            break;
        } else if (get(i) != i) {
            return false;
        }
    }
    if (!i) return false;
    return true;
}

bool sc_data_format_kind_t::is_blocking() const {
    int out[sc_data_format_kind_t::MAX_DIMS] = {0};
    collect_dim_count(out);
    for (int i = 0; i < MAX_DIMS; i++) {
        if (out[i] > 1) {
            return true;
        } else if (out[i] == 0) {
            return false;
        }
    }
    return false;
}

sc_data_format_kind_t sc_data_format_kind_t::to_plain() const {
    std::vector<int> storage(this->norig_dims());
    for (int i = 0; i < this->norig_dims(); i++) {
        storage[i] = i;
    }
    return sc_data_format_kind_t(storage);
}

sc_data_format_kind_t sc_data_format_kind_t::to_channel_last() const {
    int ndims = this->norig_dims();
    std::vector<int> storage(ndims);
    for (int i = 1; i < ndims; i++) {
        storage[i] = i + 1;
    }
    storage[0] = 0;
    storage[ndims - 1] = 1;
    return sc_data_format_kind_t(storage);
}

bool sc_data_format_t::is_blocking() const {
    return format_code_.is_blocking();
}

bool sc_data_format_t::is_channel_last() const {
    return format_code_.is_channel_last();
}

bool sc_data_format_t::is_plain() const {
    return format_code_ != format_kinds::any && format_code_.is_plain();
}

bool sc_data_format_t::is_any() const {
    return format_code_ == format_kinds::any;
}

sc_data_format_t sc_data_format_t::to_plain() const {
    return sc_data_format_t(format_code_.to_plain());
}

sc_data_format_t sc_data_format_t::to_channel_last() const {
    return sc_data_format_t(format_code_.to_channel_last());
}

sc_format_category sc_data_format_t::get_format_category() const {
    if (format_code_ == format_kinds::any)
        return sc_format_category::any;
    else if (format_code_.is_blocking())
        return sc_format_category::blocked;
    else
        return sc_format_category::non_blocking;
}

void sc_data_format_t::to_string(std::ostream &os) const {
    os << *this << "\n";
}

runtime::dispatch_key sc_data_format_t::to_runtime() const {
    COMPILE_ASSERT(format_code_.ndims() < runtime::dispatch_key::meta::MAX_DIMS,
            "Cannot convert this sc_data_format_t to runtime dispatch_key");
    COMPILE_ASSERT(blocks_[0] < 256 && blocks_[1] < 256,
            "The blocks are too large for runtime dispatch_key");
    return runtime::dispatch_key(static_cast<uint64_t>(format_code_),
            blocks_[0], blocks_[1], impl_kind_t::normal,
            format_code_.is_plain());
}

std::ostream &operator<<(std::ostream &os, const sc_data_format_t &in) {
    if (in.is_any()) { return os << "any"; }
    int out[sc_data_format_kind_t::MAX_DIMS] = {0};
    int num_blocking = 0;
    for (int i = 0; i < sc_data_format_kind_t::MAX_DIMS; i++) {
        int orig_idx = in.format_code_.get(i);
        if (orig_idx != sc_data_format_kind_t::UNDEF_DIM) {
            out[orig_idx]++;
            char axis_name_base;
            // if the axis is blocking
            if (out[orig_idx] > 1) {
                // get the blocking number from in.blocks_
                COMPILE_ASSERT((size_t)num_blocking < in.blocks_.size(),
                        "Too many blocking dims");
                os << in.blocks_[num_blocking];
                num_blocking++;
                axis_name_base = 'a';
            } else {
                axis_name_base = 'A';
            }
            os << char(axis_name_base + orig_idx);
        } else {
            break;
        }
    }
    return os;
}

static bool is_fixed_blocks(const std::array<int, 4> &blocks, int number) {
    for (int i = 0; i < number; i++) {
        if (blocks[i] < 0) { return false; }
    }
    return true;
}

static void get_block_nums_for_axises(const sc_data_format_t &format,
        int blocking_num[sc_data_format_kind_t::MAX_DIMS]) {
    // index: index in the plain dims, value: number of currently met axies in
    // the format
    int orig_num_blocking[sc_data_format_kind_t::MAX_DIMS] = {0};
    size_t cur_blocking = 0;
    for (int i = 0; i < sc_data_format_kind_t::MAX_DIMS; i++) {
        int orig_idx = format.format_code_.get(i);
        if (orig_idx != sc_data_format_kind_t::UNDEF_DIM) {
            orig_num_blocking[orig_idx]++;
            if (orig_num_blocking[orig_idx] > 1) {
                // get the blocking number from in.blocks_
                COMPILE_ASSERT(cur_blocking < format.blocks_.size(),
                        "Too many blocking dims");
                blocking_num[i] = format.blocks_[cur_blocking];
                cur_blocking++;
            } else {
                // blocking_num[i]=0;
            }
        } else {
            break;
        }
    }
}

std::unordered_map<int, std::vector<int>>
sc_data_format_t::get_blocked_axis() const {
    std::unordered_map<int, std::vector<int>> blocked_axis;
    int out[sc_data_format_kind_t::MAX_DIMS] = {0};
    int block_pos = 0;
    for (int i = 0; i < sc_data_format_kind_t::MAX_DIMS; i++) {
        int orig_idx = format_code_.get(i);
        if (orig_idx != sc_data_format_kind_t::UNDEF_DIM) {
            ++out[orig_idx];
            if (out[orig_idx] > 1) {
                blocked_axis[orig_idx].push_back(blocks_[block_pos++]);
            }
        } else {
            break;
        }
    }
    return blocked_axis;
}

void get_blocking_shapes_impl(const sc_dims &plain_shapes,
        const sc_data_format_t &format, size_t base_out_dim,
        size_t num_format_dims, size_t num_out_dims,
        const std::function<void(int, int)> &callback) {
    COMPILE_ASSERT(plain_shapes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    // index: index in the format, value: the blocking number collected from
    // format.blocks_. e.g. for NCHW16c, blocking_num=[0,0,0,0,16,...]
    int blocking_num[sc_data_format_kind_t::MAX_DIMS] = {0};
    get_block_nums_for_axises(format, blocking_num);

    // calculate the real shapes
    // For example, KCRS16c8k4c, format code = [0,1,2,3,1,0,1], where plain dims
    // is KCRS=[a,b,c,d]
    // 1. for each axis in the target format, get the original axis. e.g. for C
    // axis, the original axis is 1.
    // 2. get the blocking/original shape of the axis. e.g. for C axis in
    // KCRS16c8k4c, it is the first axis of C, so the original shape on the axis
    // is b. For the first blocking axis of "16c", its blocking number is 16.
    // 3. finalize the shape of the current axis. It should be
    // (original_shape/next_blocking_number). e.g. for C axis in KCRS16c8k4c,
    // original shape=b, next blocking number=16, so the shape of C axis is
    // ceil(b/16). for "16c" axis in KCRS16c8k4c, original shape=16, next
    // blocking number=4, so the shape of "16c" axis is ceil(16/4)=4.
    for (auto out_idx = base_out_dim; out_idx < num_out_dims; out_idx++) {
        auto idx_in_format = out_idx - base_out_dim;
        int orig_axis = format.format_code_.get(idx_in_format);
        assert((size_t)orig_axis < plain_shapes.size());
        // find original shape from plain_shapes or blocking_num
        int new_shape;
        if (blocking_num[idx_in_format] != 0) {
            new_shape = blocking_num[idx_in_format];
        } else {
            new_shape = plain_shapes.at(orig_axis + base_out_dim);
        }
        // get next_blocking_number
        int next_blocking_number = 1;
        for (size_t i = idx_in_format + 1; i < num_format_dims; i++) {
            // find next axis in format with same axis name
            if (orig_axis == format.format_code_.get(i)) {
                next_blocking_number = blocking_num[i];
                break;
            }
        }
        callback(new_shape, next_blocking_number);
    }
}

sc_dims sc_data_format_t::get_blocking_shapes(
        const sc_dims &plain_shapes, const sc_data_format_t &format) {
    if (plain_shapes.empty()) { return sc_dims(); }
    // todo: should be is_plain()
    if (format.format_code_ == format_kinds::any) { return plain_shapes; }
    sc_dims ret;
    size_t base_out_dim = 0;
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_format_dims;
    ret.reserve(num_out_dims);
    COMPILE_ASSERT(plain_shapes.size() == num_plain_dims,
            "Wrong number of dimensions for format: "
                    << format
                    << ", plain shape = " << utils::print_vector(plain_shapes));
    auto callback = [&](int new_shape, int next_blocking_number) {
        if (is_dynamic_dim(new_shape) || is_dynamic_dim(next_blocking_number)) {
            ret.push_back(dimensions::dynamic_any);
        } else {
            ret.push_back(
                    utils::divide_and_ceil(new_shape, next_blocking_number));
        }
    };
    get_blocking_shapes_impl(plain_shapes, format, base_out_dim,
            num_format_dims, num_out_dims, callback);
    return ret;
}

static size_t throw_if_negative(int dim) {
    if (dim < 0) { throw std::runtime_error("Bad format"); }
    return dim;
}

std::vector<expr> get_blocking_shapes_expr(sc_graph_t &g,
        const sc_dims &plain_shapes, const sc_data_format_t &format) {
    if (plain_shapes.empty()) { return std::vector<expr>(); }
    // todo: should be is_plain()
    if (format.format_code_ == format_kinds::any) {
        return g.dims_to_expr(plain_shapes);
    }
    std::vector<expr> ret;
    size_t base_out_dim = 0;
    size_t num_plain_dims = throw_if_negative(format.format_code_.norig_dims());
    size_t num_format_dims = throw_if_negative(format.format_code_.ndims());
    size_t num_out_dims = num_format_dims;
    ret.reserve(num_out_dims);
    COMPILE_ASSERT(plain_shapes.size() == num_plain_dims,
            "Wrong number of dimensions for format: "
                    << format
                    << ", plain shape = " << utils::print_vector(plain_shapes));
    auto callback = [&](int new_shape, int next_blocking_number) {
        auto dim = next_blocking_number == 1
                ? g.dim_to_expr(new_shape)
                : divide_and_ceil(g.dim_to_expr(new_shape),
                        g.dim_to_expr(next_blocking_number));
        dim->attr().set(attr_key::const_attr, true);
        ret.push_back(dim);
    };
    get_blocking_shapes_impl(plain_shapes, format, base_out_dim,
            num_format_dims, num_out_dims, callback);
    return ret;
}

sc_dims sc_data_format_t::get_padded_plain_shapes(
        const sc_dims &real_shapes, const sc_data_format_t &format) {
    if (real_shapes.empty()) { return sc_dims(); }
    if (format.format_code_ == format_kinds::any) { return real_shapes; }
    sc_dims ret;
    size_t base_out_dim = 0;
    size_t num_plain_dims = format.format_code_.norig_dims();
    size_t num_format_dims = format.format_code_.ndims();
    size_t num_out_dims = num_plain_dims;
    ret.reserve(num_out_dims);
    COMPILE_ASSERT(real_shapes.size() == num_format_dims,
            "Wrong number of dimensions for format: "
                    << format
                    << ", real shape = " << utils::print_vector(real_shapes));
    if (!is_fixed_blocks(format.blocks_, 4)) {
        return sc_dims(num_out_dims, -1);
    }
    COMPILE_ASSERT(real_shapes.size() <= sc_data_format_kind_t::MAX_DIMS,
            "Too many dims in plain shapes");
    for (auto out_idx = base_out_dim; out_idx < num_out_dims; out_idx++) {
        auto orig_axis = out_idx - base_out_dim;
        int shape = 1;
        for (size_t i = 0; i < num_format_dims; i++) {
            if ((int)orig_axis == format.format_code_.get(i)) {
                if (is_dynamic_dim(real_shapes.at(base_out_dim + i))) {
                    shape = dimensions::dynamic_any;
                    break;
                }
                shape *= real_shapes.at(base_out_dim + i);
            }
        }
        // shape is the product of all axises with the same name
        ret.push_back(shape);
    }
    return ret;
}

sc_dims sc_data_format_t::get_reordered_shapes(const sc_dims &input_shapes,
        const sc_data_format_t &input_format,
        const sc_data_format_t &output_format) {
    COMPILE_ASSERT(input_format.is_convertible(output_format),
            "Can not convert input format "
                    << input_format << " to output format " << output_format);
    sc_dims plain_shapes = get_padded_plain_shapes(input_shapes, input_format);
    return get_blocking_shapes(plain_shapes, output_format);
}

int sc_data_format_t::get_blocks_size() const {
    int i = 0;
    for (; i < static_cast<int>(blocks_.size()); i++) {
        if (blocks_[i] == 0) { return i; }
    }
    return i;
}

bool sc_data_format_t::is_same_format_kind(
        const sc_data_format_t &input_format) const {
    return this->format_code_ == input_format.format_code_;
}

bool sc_data_format_cmper_t::operator()(
        const sc_data_format_t &fmt0, const sc_data_format_t &fmt1) const {
    if (fmt0.format_code_ != fmt1.format_code_) {
        return fmt0.format_code_ < fmt1.format_code_;
    }
    if (fmt0.blocks_ != fmt1.blocks_) {
        for (int i = 0; i < 4; i++) {
            if (fmt0.blocks_[i] != fmt1.blocks_[i]) {
                return fmt0.blocks_[i] < fmt1.blocks_[i];
            }
        }
    }
    // equal
    return false;
}

bool is_dynamic_blocking(
        const sc_dims &shapes, const sc_data_format_t &format) {
    auto &code = format.format_code_;
    for (size_t i = 0; i < shapes.size(); i++) {
        if (is_dynamic_dim(shapes[i])
                && !code.collect_blocking_index(i).empty()) {
            return true;
        }
    }
    return false;
}

// clang-format off
SC_CLASS(sc_data_format_t)
    SC_FIELD(format_code_)
    SC_FIELD(blocks_)
SC_CLASS_END()

SC_CLASS(sc_data_format_kind_t)
    SC_FIELD(storage_)
SC_CLASS_END()
// clang-format on

template struct reflection::type_registry<
        std::vector<std::vector<sc_data_format_t>>>;

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
std::size_t hash<dnnl::impl::graph::gc::sc_data_format_t>::operator()(
        const dnnl::impl::graph::gc::sc_data_format_t &k) const {
    namespace gc = dnnl::impl::graph::gc;
    size_t hash_ = 0;
    gc::hash_combine(hash_, (uint64_t)k.format_code_);
    gc::hash_combine(hash_, get<0>(k.blocks_));
    gc::hash_combine(hash_, get<1>(k.blocks_));
    gc::hash_combine(hash_, get<2>(k.blocks_));
    gc::hash_combine(hash_, get<3>(k.blocks_));
    return hash_;
}
} // namespace std
