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
#include <limits>
#include <vector>

#include "dynamic_dispatch_key.hpp"
#include "dynamic_lower_info.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_internal_info.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <compiler/ir/graph/fusion_data.hpp>
#include <compiler/ir/graph/quantization/quantize_op.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/graph/utils.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/simple_licm.hpp>
#include <util/general_object.hpp>
#include <util/hash_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_MODULE(sc_graph);

extern void __dummy_init();

template <typename valT>
valT &gt_map_t<valT>::get(graph_tensor *v) {
    auto itr = datamap_.find(v);
    if (itr != datamap_.end()) { return itr->second; }
    auto &ret = datamap_[v];
    return ret;
}

template <typename valT>
valT &gt_map_t<valT>::get(const graph_tensor_ptr &v) {
    return get(v.get());
}

template <typename valT>
bool gt_map_t<valT>::haskey(graph_tensor *v) const {
    return datamap_.find(v) != datamap_.end();
}

template <typename valT>
bool gt_map_t<valT>::haskey(const graph_tensor_ptr &v) const {
    return haskey(v.get());
}

template struct gt_map_t<fusion_data_t>;
template struct gt_map_t<slice_range_list>;
template struct gt_map_t<graph_tensor_ptr>;
template struct gt_map_t<std::vector<int>>;
template struct gt_map_t<expr>;
template struct gt_map_t<fuse_anchor_map_t *>;
template struct gt_map_t<bound_axis>;

sc_op_ptr op_traits::auto_copyable_t::copy(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_graph_t &mgr) {
    auto ths = dynamic_cast<sc_op *>(this);
    auto ret = mgr.make(ths->op_name_, ins, outs, ths->attrs_);
    ret->copy_dispatch_key_set_from_op(ths->shared_from_this());
    return ret;
}

std::vector<graph_tensor_ptr> copy_logical_tsr(
        const std::vector<graph_tensor_ptr> &v) {
    std::vector<graph_tensor_ptr> ret;
    ret.reserve(v.size());
    for (auto &t : v) {
        ret.emplace_back(std::make_shared<graph_tensor>(nullptr, t->details_));
    }
    return ret;
}

static std::vector<expr> *get_tensor_dims(const expr &tsr) {
    if (tsr.isa<tensor>()) {
        auto t = tsr.static_as<tensor>();
        return &t->dims_;
    } else {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        return &tsr.static_as<tensorptr>()->shape_;
    }
}

const std::vector<expr> &tensor_slice::get_base_dims() const {
    return *get_tensor_dims(tptr_);
}

sc_data_type_t tensor_slice::get_base_dtype() const {
    return get_real_tensor()->elem_dtype_;
}

tensor tensor_slice::get_real_tensor() const {
    auto &base = tptr_->base_;
    COMPILE_ASSERT(base.isa<indexing>(),
            "tensor_ptr base should be indexing, but got: " << base);
    auto tsr = base->ptr_;
    while (!tsr.isa<tensor>()) {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        auto base = tsr.static_as<tensorptr>()->base_;
        COMPILE_ASSERT(base.isa<indexing>(),
                "tensor_ptr base should be indexing, but got: " << base);
        tsr = base.checked_as<indexing>()->ptr_;
    }
    return tsr.static_as<tensor>();
}

slice_range tensor_slice::get_ranges() const {
    COMPILE_ASSERT(get_shape().size() == tptr_->base_->idx_.size(),
            "Unmatched shape and idx found");
    auto shape = get_shape();
    auto offset = tptr_->base_->idx_;
    slice_range ranges;
    for (int64_t i = 0; i < nslice_dims(); i++) {
        ranges.emplace_back(std::make_pair(offset[i], shape[i]));
    }
    return ranges;
}

bool tensor_slice::full_on_axis(const std::vector<int> &axis) const {
    auto &dims = get_base_dims();
    auto &idx = tptr_->base_->idx_;
    auto &shape = get_shape();
    for (auto &ax : axis) {
        if (!idx[ax].isa<constant>() || !shape[ax].isa<constant>()) {
            return false;
        }
        if (get_const_as_int(idx[ax].checked_as<constant>()) != 0
                || get_const_as_int(shape[ax].checked_as<constant>())
                        != get_const_as_int(dims[ax].checked_as<constant>())) {
            return false;
        }
    }
    return true;
}

bool tensor_slice::is_full() const {
    auto &dims = get_base_dims();
    std::vector<int> total_axis;
    total_axis.reserve(static_cast<int>(dims.size()));
    for (int i = 0; i < static_cast<int>(dims.size()); i++) {
        total_axis.emplace_back(i);
    }
    return full_on_axis(total_axis);
}

bool tensor_slice::is_const() const {
    return std::all_of(shape_.begin(), shape_.end(),
            [](const expr &e) { return do_cast_and_fold(e).isa<constant>(); });
}

tensor_slice::tensor_slice(const expr &tsr) {
    if (tsr.isa<tensor>()) {
        auto t = tsr.static_as<tensor>();
        tptr_ = builder::tensor_ptr(
                tsr, std::vector<expr>(t->dims_.size(), 0), {}, true)
                        .static_as<tensorptr>();
        shape_ = t->dims_;
    } else {
        COMPILE_ASSERT(tsr.isa<tensorptr>(),
                "tensor_slice only accepts a tensor or tensorptr, got: "
                        << tsr);
        tptr_ = tsr.static_as<tensorptr>();
        shape_ = tptr_->shape_;
    }
}

tensor_slice::tensor_slice(const expr &tsr, slice_range &&range) {
    auto dims = get_tensor_dims(tsr);
    if (dims->size() != range.size())
        COMPILE_ASSERT(dims->size() == 1
                        && get_const_as_int((*dims)[0].checked_as<constant>())
                                == 1,
                "Unmatched range found. Tensor: "
                        << (tsr.isa<tensor>() ? tsr.static_as<tensor>()->name_
                                              : "")
                        << " have dims: " << utils::print_vector(*dims)
                        << " but got slice range: "
                        << utils::print_pair_vector(range));
    tptr_ = builder::tensor_ptr(tsr,
            dims->size() != range.size() ? std::vector<expr> {0}
                                         : get_slice_idx(range),
            {}, true)
                    .static_as<tensorptr>();
    shape_ = get_slice_shape(range);
}

graph_tensor::graph_tensor(sc_op *owner) : producer_owner_(owner) {}
graph_tensor::graph_tensor(sc_op *owner, const logical_tensor_t &lt)
    : details_(lt), producer_owner_(owner) {}

graph_tensor::graph_tensor(sc_op *owner, const sc_data_format_t &format,
        const sc_dims &plain_shape, const sc_data_type_t &type,
        const sc_dims &stride)
    : details_(format, plain_shape, type, stride), producer_owner_(owner) {}

const sc_dims &logical_tensor_t::get_blocking_dims() const {
    return dims_;
}

std::vector<expr> logical_tensor_t::get_blocking_dims_expr(
        sc_graph_t &g) const {
    return get_blocking_shapes_expr(g, plain_dims_, format_);
}

static bool check_stride_validity(
        const bool is_dynamic, const sc_dims &dims, const sc_dims &strides) {
    return strides.size() == dims.size()
            && (is_dynamic
                    || std::is_sorted(strides.begin(), strides.end(),
                            std::greater<sc_dim>()));
}

void logical_tensor_t::internal_update() {
    dims_ = sc_data_format_t::get_blocking_shapes(plain_dims_, format_);
    if (strides_.empty()) {
        strides_ = compute_dense_stride(dims_);
    } else {
        COMPILE_ASSERT(check_stride_validity(is_dynamic(), dims_, strides_),
                "Specified strides value invalid or not consistent with "
                "real(blocking) dims.")
    }
}

bool logical_tensor_t::is_dynamic() const {
    return !std::all_of(plain_dims_.begin(), plain_dims_.end(),
            [](const sc_dim &dim) { return !is_dynamic_dim(dim); });
}

// sets the logical dims in plain format
void logical_tensor_t::set_plain_dims(const sc_dims &plain_dims) {
    COMPILE_ASSERT(is_dynamic() || is_dense(),
            "Forbid update format on a strided tensor.");
    strides_.clear();
    plain_dims_ = plain_dims;
    internal_update();
}

// TODO(xxx): this logic maybe not correct, just distinguish with set_plain_dims
void logical_tensor_t::set_blocking_dims(const sc_dims &blocking_dims) {
    // assert(format_.format_code_ == format_kinds::any);
    COMPILE_ASSERT(is_dense(), "Forbid set blocking dims on a strided tensor.");
    format_.format_code_ = format_kinds::any;
    plain_dims_ = blocking_dims;
    dims_ = blocking_dims;
    strides_ = compute_dense_stride(dims_);
}

void logical_tensor_t::set_format(const sc_data_format_t &newv) {
    COMPILE_ASSERT(is_dense(), "Forbid set format on a strided tensor.");
    strides_.clear();
    format_ = newv;
    internal_update();
}

void logical_tensor_t::set_strides(const sc_dims &strides) {
    COMPILE_ASSERT(check_stride_validity(is_dynamic(), dims_, strides),
            "Specified strides value invalid or not consistent with "
            "real(blocking) dims.")
    strides_ = strides;
}

void logical_tensor_t::set_format_and_stride(
        const sc_data_format_t &newv, const sc_dims &strides) {
    format_ = newv;
    strides_ = strides;
    internal_update();
}

void logical_tensor_t::add_format_candidate(const sc_data_format_t &newv) {
    format_candidates_.insert(newv);
}

void logical_tensor_t::remove_format_candidate(const sc_data_format_t &v) {
    auto it = format_candidates_.find(v);
    if (it != format_candidates_.end()) { format_candidates_.erase(it); }
}

void logical_tensor_t::set_format_candidates(
        const std::vector<sc_data_format_t> &newf) {
    format_candidates_.insert(newf.begin(), newf.end());
    if (format_candidates_.size() == 1) {
        format_ = *format_candidates_.begin();
    }
    internal_update();
}

size_t logical_tensor_t::get_blocking_byte_size() const {
    COMPILE_ASSERT(!is_dynamic(), "blocking byte size should be static shape.");
    size_t sz = utils::get_sizeof_type(dtype_);
    for (auto z : get_blocking_dims()) {
        sz *= z;
    }
    return sz;
}

bool logical_tensor_t::is_dense() {
    if (strides_.empty()) { return true; }
    if (is_dynamic()) { return true; }
    if (std::any_of(plain_dims_.begin(), plain_dims_.end(),
                [](const sc_dim &d) { return d == 0; })) {
        return true;
    }
    assert(strides_.size() == dims_.size());
    if (strides_.back() != 1) { return false; }
    for (int i = dims_.size() - 2; i >= 0; --i) {
        if (strides_[i] != strides_[i + 1] * dims_[i + 1]) { return false; }
    }
    return true;
}

std::vector<expr> logical_tensor_t::get_strides_expr(sc_graph_t &g) const {
    return is_dynamic() ? dims_to_dense_stride(get_blocking_dims_expr(g))
                        : g.dims_to_expr(get_strides());
}

sc_dims logical_tensor_t::compute_dense_stride(const sc_dims &dims) {
    sc_dims strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) {
        if (dims[i + 1] == 0) {
            strides[i] = strides[i + 1];
        } else {
            strides[i] = dims[i + 1] * strides[i + 1];
        }
    }
    return strides;
}

void logical_tensor_t::to_string(std::ostream &os) {
    os << '[' << dtype_ << ' ' << utils::print_vector(get_blocking_dims())
       << " @ " << format_ << ']';
}

size_t logical_tensor_t::hash() const {
    size_t seed = 0;
    hash_combine(seed, static_cast<uint64_t>(dtype_));
    hash_combine(seed, plain_dims_);
    hash_combine(seed, dims_);
    hash_combine(seed, strides_);
    hash_combine(seed, format_);
    return seed;
}

void graph_tensor::attach_use(sc_op_ptr op, int index) {
    uses_.emplace_back(std::make_pair(index, std::move(op)));
}

void graph_tensor::detach_use(const sc_op_ptr &op) {
    for (auto itr = uses_.begin(); itr != uses_.end();) {
        if (itr->second == op) {
            itr = uses_.erase(itr);
        } else {
            ++itr;
        }
    }
}

void graph_tensor::detach_use(const sc_op_ptr &op, int input_idx) {
    for (auto itr = uses_.begin(); itr != uses_.end();) {
        if (itr->first == input_idx && itr->second == op) {
            itr = uses_.erase(itr);
        } else {
            ++itr;
        }
    }
}

void graph_tensor::replace_with(const graph_tensor_ptr &v) {
    while (!uses_.empty()) {
        auto node = uses_.front();
        node.second->replace_input(node.first, v);
    }
}

graph_tensor_ptr graph_tensor::copy() {
    return std::make_shared<graph_tensor>(producer_owner_, details_);
}

void sc_op::replace_input(size_t index, const graph_tensor_ptr &new_input,
        const bool skip_shape_check) {
    if (!skip_shape_check) {
        assert(index < info_.inputs_.size());
        assert(new_input->details_.is_dynamic()
                || get_dims_product(
                           info_.inputs_[index]->details_.get_plain_dims())
                        == get_dims_product(
                                new_input->details_.get_plain_dims()));
    }
    info_.inputs_[index]->detach_use(shared_from_this(), index);
    info_.inputs_[index] = new_input;
    new_input->attach_use(shared_from_this(), index);
}

void sc_op::replace_uses_with_and_remove(const sc_op_ptr &replacer) {
    assert(info_.outputs_.size() == replacer->info_.outputs_.size());
    for (unsigned i = 0; i < info_.outputs_.size(); i++) {
        auto &ths_out = info_.outputs_[i];
        auto &replace_out = replacer->info_.outputs_[i];
        ths_out->replace_with(replace_out);
    }
    remove();
}

bool sc_op::has_graph_output() const {
    for (const auto &output : info_.outputs_) {
        for (const auto &use_node : output->uses_) {
            if (use_node.second->isa<output_op>()) { return true; }
        }
    }
    return false;
}

bool sc_op::is_single_output_single_use() {
    return info_.outputs_.size() == 1 && info_.outputs_[0]->uses_.size() == 1;
}

bool sc_op::is_dynamic() const {
    return !std::all_of(info_.inputs_.begin(), info_.inputs_.end(),
                   [](const graph_tensor_ptr &inp) {
                       return !inp->details_.is_dynamic();
                   })
            || !std::all_of(info_.outputs_.begin(), info_.outputs_.end(),
                    [](const graph_tensor_ptr &out) {
                        return !out->details_.is_dynamic();
                    });
}

const dispatch_set_ptr &sc_op::get_dispatch_key_set() const {
    assert(info_.dispatch_key_set_);
    return info_.dispatch_key_set_;
}

dispatch_set_ptr &sc_op::get_dispatch_key_set() {
    if (!info_.dispatch_key_set_) {
        info_.dispatch_key_set_ = std::make_shared<dispatch_key_set_t>();
    }
    return info_.dispatch_key_set_;
}

dispatch_set_ptr sc_op::get_internal_dispatch_key_set(const context_ptr &ctx) {
    throw std::runtime_error(
            "Internal dispatch key set should be implemented by concrete op.");
}

void sc_op::copy_dispatch_key_set_from_op(const sc_op_ptr &other) {
    if (other->info_.dispatch_key_set_) {
        info_.dispatch_key_set_ = other->info_.dispatch_key_set_->copy();
    }
}

void sc_op::remove() {
    for (auto &in : info_.inputs_) {
        in->detach_use(shared_from_this());
    }
    info_.inputs_.clear();
    info_.outputs_.clear();
    attrs_.as_map().clear();
    is_removed_ = true;
}

// template op and fusible op common constructor
sc_op::sc_op(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &producer_lt,
        const std::vector<graph_tensor_ptr> &consumer_lt,
        const any_map_t &attrs)
    : attrs_(attrs), op_name_(op_name) {
    info_.inputs_ = producer_lt;
    info_.outputs_ = consumer_lt;
    for (auto &op : info_.outputs_) {
        op->producer_owner_ = this;
    }
}

void sc_op::format_to_dense_format_stride_pair(
        const std::vector<std::vector<sc_data_format_t>> &in_formats,
        const std::vector<std::vector<sc_data_format_t>> &out_formats,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    supported_ins.resize(in_formats.size());
    for (size_t i = 0; i < in_formats.size(); ++i) {
        for (auto fmt : in_formats[i]) {
            logical_tensor_t dense_lt(fmt,
                    info_.inputs_[i]->details_.get_plain_dims(),
                    info_.inputs_[i]->details_.dtype_);
            supported_ins[i].emplace_back(
                    std::make_pair(fmt, dense_lt.get_strides()));
        }
    }
    supported_outs.resize(out_formats.size());
    for (size_t i = 0; i < out_formats.size(); ++i) {
        for (auto fmt : out_formats[i]) {
            logical_tensor_t dense_lt(fmt,
                    info_.outputs_[i]->details_.get_plain_dims(),
                    info_.outputs_[i]->details_.dtype_);
            supported_outs[i].emplace_back(fmt, dense_lt.get_strides());
        }
    }
}

sc_graph_t::sc_graph_t(sc_graph_t &&other)
    : ops_(std::move(other.ops_))
    , attrs_(std::move(other.attrs_))
    , dyn_info_(std::move(other.dyn_info_)) {
    for (auto &op : ops_) {
        op->set_owner_graph(this);
    }
}

sc_graph_t &sc_graph_t::operator=(sc_graph_t &&other) {
    ops_ = std::move(other.ops_);
    attrs_ = std::move(other.attrs_);
    dyn_info_ = std::move(other.dyn_info_);
    for (auto &op : ops_) {
        op->set_owner_graph(this);
    }
    return *this;
}

size_t sc_graph_t::hash_contents(
        const std::function<bool(const sc_op *, const std::string &)> &filter)
        const {
    size_t seed = 0;
    hash_combine(seed, attrs_.get_or_else("fpmath_mode", 0));
    op_visitor_t vis = op_visitor_t::bfs_topology_sort(this->ops_.size());
    vis.visit_graph(*this, [&](op_visitor_t *vis, const sc_op_ptr &op) {
        hash_combine(seed, op->hash_contents(filter));
    });
    return seed;
}

void sc_graph_t::reset_op_ids() {
    for (auto it = ops_.begin(); it != ops_.end();) {
        if ((*it)->is_removed_
                || ((*it)->get_inputs().empty()
                        && (*it)->get_outputs().empty())) {
            it = ops_.erase(it);
        } else {
            ++it;
        }
    }
    for (size_t i = 0; i < ops_.size(); ++i) {
        ops_[i]->logical_op_id_ = i;
    }
}

void sc_graph_t::resort_op_ids(
        const std::unordered_map<sc_op_ptr, int> &op_id_map) {
    std::sort(ops_.begin(), ops_.end(),
            [&op_id_map](const sc_op_ptr &A, const sc_op_ptr &B) {
                auto A_iter = op_id_map.find(A), B_iter = op_id_map.find(B);
                COMPILE_ASSERT(
                        A_iter != op_id_map.end() && B_iter != op_id_map.end(),
                        "op id map is not enough, could not do sorting")
                return A_iter->second < B_iter->second;
            });
    for (size_t i = 0; i < ops_.size(); ++i) {
        ops_[i]->logical_op_id_ = i;
    }
}

sc_dim sc_graph_t::get_next_dynamic_placeholder() {
    if (!dyn_info_) { dyn_info_ = std::make_shared<dynamic_lower_info_t>(); }
    COMPILE_ASSERT(std::numeric_limits<sc_dim>::min()
                    < dyn_info_->cur_dynamic_placeholder_,
            "Dynamic shapes are too many to mark!");
    return dyn_info_->cur_dynamic_placeholder_--;
}

expr sc_graph_t::dim_to_expr(const sc_dim &v) {
    const std::string dynamic_prefix = "dynamic_var_";
    if (is_dynamic_dim(v)) {
        if (!dyn_info_) {
            dyn_info_ = std::make_shared<dynamic_lower_info_t>();
        }
        assert(v != dimensions::dynamic_any);
        auto &m = dyn_info_->dim2expr_map_;
        auto it = m.find(v);
        if (it == m.end()) {
            auto dyn_var = make_expr<var_node>(
                    datatypes::index, dynamic_prefix + std::to_string(-v));
            dyn_var->attr().set(attr_key::const_attr, true);
            m[v] = dyn_var;
            return dyn_var;
        } else {
            return it->second;
        }
    }
    return dim2unsigned(v);
}

std::vector<expr> sc_graph_t::dims_to_expr(const sc_dims &dim) {
    std::vector<expr> dim_expr;
    dim_expr.reserve(dim.size());
    for (auto d : dim) {
        dim_expr.emplace_back(dim_to_expr(d));
    }
    return dim_expr;
}

std::vector<expr_c> sc_graph_t::dims_to_expr_c(const sc_dims &dim) {
    std::vector<expr_c> dim_expr;
    dim_expr.reserve(dim.size());
    for (auto d : dim) {
        dim_expr.emplace_back(dim_to_expr(d));
    }
    return dim_expr;
}

float sc_graph_t::get_gflop() const {
    float gflop = 0.f;
    for (auto &op : ops_) {
        if (op->is_removed_) continue;
        gflop += op->get_gflop();
    }
    return gflop;
}

bool sc_graph_t::is_dynamic() const {
    return !std::all_of(ops_.begin(), ops_.end(),
            [](const sc_op_ptr &op) { return !op->is_dynamic(); });
}

bool sc_graph_t::is_non_dense() const {
    if (is_dynamic()) {
        // currently dynamic graph is always dense
        return false;
    }
    for (const sc_op_ptr &op : ops_) {
        for (const graph_tensor_ptr &gt : op->get_inputs()) {
            if (!gt->details_.is_dense()) { return true; }
        }
    }
    return false;
}

void sc_graph_t::add(const sc_op_ptr &ret) {
    assert(ret->logical_op_id_ == 0);
    ret->logical_op_id_ = ops_.size();
    ret->set_owner_graph(this);
    for (auto &outs : ret->info_.outputs_) {
        assert(outs->producer_owner_ == nullptr
                || outs->producer_owner_ == ret.get());
        outs->producer_owner_ = ret.get();
    }
    for (unsigned i = 0; i < ret->info_.inputs_.size(); i++) {
        ret->info_.inputs_[i]->attach_use(ret, i);
    }
    ops_.emplace_back(ret);
}

std::shared_ptr<sc_op> sc_graph_t::make(const std::string &op_name,
        const std::vector<graph_tensor_ptr> &inputs,
        const std::vector<graph_tensor_ptr> &outputs, const any_map_t &attrs) {
    // make a reference to all needed ops to prevent them from being removed by
    // linker
    __dummy_init();
    std::shared_ptr<sc_op> ret, in_ret;
    // internally create input_op
    // todo: LLGA-sc front end should create input_op first, instead of creating
    // it internally
    for (auto &ins : inputs) {
        if (!ins->producer_owner_) {
            auto in_ret = std::make_shared<input_op>(
                    std::vector<graph_tensor_ptr> {ins});
            in_ret->logical_op_id_ = ops_.size();
            ops_.emplace_back(std::move(in_ret));
        }
    }

    std::string decay_op_name = graph::decay_quantized_op_name(op_name);
    // firstly search template, secondly search fusible
    // todo: add all tunable ops
    if (auto f = get_op_factory(decay_op_name)) {
        ret = f(inputs, outputs, attrs);
    } else {
        COMPILE_ASSERT(false, "Unsupported op: " << decay_op_name);
    }
    bool is_quantized = utils::string_startswith(op_name, "quantized");
    if (is_quantized) {
        ret->dyn_cast<op_traits::may_quantize_t>()->is_quantized_ = true;
    }
    add(ret);
    // As owner graph is initialzed after op's constructor and add, some ops
    // like conv need infer output plain shapes at this step.
    ret->infer_out_tensor_details();
    return ret;
}

std::shared_ptr<sc_op> sc_graph_t::make_output(
        const std::vector<graph_tensor_ptr> &inputs, const any_map_t &attrs) {
    auto ret = std::make_shared<output_op>(inputs);
    ret->owner_graph_ = this;
    ret->attrs_ = attrs;
    for (unsigned i = 0; i < inputs.size(); i++) {
        inputs[i]->attach_use(ret, i);
    }
    ret->logical_op_id_ = ops_.size();
    ops_.emplace_back(ret);
    return ret;
}

std::shared_ptr<sc_op> sc_graph_t::make_input(
        const std::vector<graph_tensor_ptr> &inputs, const any_map_t &attrs) {
    auto ret = std::make_shared<input_op>(inputs);
    ret->owner_graph_ = this;
    ret->attrs_ = attrs;
    ret->logical_op_id_ = ops_.size();
    if (ret->is_dynamic()) { ret->initialize_dynamic_placeholder(); }
    ops_.emplace_back(ret);
    return ret;
}

std::vector<sc_op_ptr> sc_graph_t::get_output_ops() {
    std::vector<sc_op_ptr> output_ops;
    for (auto &op : ops_) {
        if (op->isa<output_op>()) { output_ops.push_back(op); }
    }
    return output_ops;
}
std::vector<sc_op_ptr> sc_graph_t::get_input_ops() {
    std::vector<sc_op_ptr> input_ops;
    for (auto &op : ops_) {
        if (op->isa<input_op>()) { input_ops.push_back(op); }
    }
    return input_ops;
}

std::vector<sc_op_ptr> sc_graph_t::get_input_or_const_ops() const {
    std::vector<sc_op_ptr> input_ops;
    for (auto &op : ops_) {
        if (op->isa<input_op>() || op->isa<constant_op_t>()) {
            input_ops.push_back(op);
        }
    }
    return input_ops;
}

std::unordered_set<sc_dim> sc_graph_t::get_external_dynamic_vars() {
    std::unordered_set<sc_dim> ext_vars;
    auto extract_vars = [&ext_vars](const std::vector<sc_op_ptr> &ops) {
        for (auto &op : ops) {
            for (auto &ins : op->get_inputs()) {
                for (auto &d : ins->details_.get_plain_dims()) {
                    if (is_dynamic_dim(d)) { ext_vars.insert(d); }
                }
            }
            for (auto &outs : op->get_outputs()) {
                for (auto &d : outs->details_.get_plain_dims()) {
                    if (is_dynamic_dim(d)) { ext_vars.insert(d); }
                }
            }
        }
    };
    extract_vars(get_input_ops());
    extract_vars(get_output_ops());
    // dynamic reshape is also traited as external var.
    std::vector<sc_op_ptr> dyn_reshapes;
    for (auto &op : ops_) {
        if (op->op_name_ == "dynamic_reshape") { dyn_reshapes.push_back(op); }
    }
    extract_vars(dyn_reshapes);
    return ext_vars;
}

bool sc_graph_t::need_dynamic_internal_query() {
    return !std::all_of(ops_.begin(), ops_.end(), [](const sc_op_ptr &op) {
        return !op->need_dynamic_internal_query();
    });
}

bool sc_op::need_dynamic_internal_query() {
    bool ret = need_dynamic_internal_query_impl();
    if (ret && !info_.internal_info_) {
        info_.internal_info_ = std::make_shared<dyn_internal_info_t>();
    }
    return ret;
}

bool sc_op::compare_contents(const sc_op *other, // NOLINT
        const std::function<bool(const sc_op *, const std::string &)> &filter)
        const {
    if (op_name_ != other->op_name_) { return false; }
    int numattrs = 0, othernumattrs = 0;
    auto &othermap = other->attrs_.as_map();
    for (auto &kv : attrs_.as_map()) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        if (filter && !filter(this, kv.first)) { continue; }
        numattrs++;
        auto otherkv = othermap.find(kv.first);
        if (otherkv == othermap.end()) { return false; }
        if (kv.second.cmp(otherkv->second) != 0) { return false; }
    }
    for (auto &kv : othermap) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        if (filter && !filter(this, kv.first)) { continue; }
        othernumattrs++;
    }
    if (numattrs != othernumattrs) { return false; }

    return true;
}

size_t sc_op::standard_hash_contents(const sc_op *p,
        const std::function<bool(const sc_op *, const std::string &)> &filter) {
    size_t seed = 0;
    for (auto &in : p->info_.inputs_) {
        hash_combine(seed, in->details_.hash());
    }
    for (auto &out : p->info_.outputs_) {
        hash_combine(seed, out->details_.hash());
    }
    hash_combine(seed, p->op_name_);
    for (auto &kv : p->attrs_.as_map()) {
        if (utils::string_startswith(kv.first, "temp.")) { continue; }
        if (filter && !filter(p, kv.first)) { continue; }
        // To hash unordered_map, use `XOR`, which satisfies commutative law.
        // Otherwise, for ordered containers (like arrays), use `hash_combine`
        // to distinguish result from the differnt sequence order.
        if (!kv.second.empty()) { seed ^= kv.second.hash(); }
    }
    return seed;
}

size_t sc_op::hash_contents( // NOLINT
        const std::function<bool(const sc_op *, const std::string &)> &filter)
        const {
    return standard_hash_contents(this, filter);
}

static std::unordered_map<std::string, op_factory_func> &get_op_factory_map() {
    static std::unordered_map<std::string, op_factory_func> op_map;
    return op_map;
}

op_factory_func get_op_factory(const std::string &name) {
    auto &op_map = get_op_factory_map();
    auto itr = op_map.find(name);
    if (itr != op_map.end()) { return itr->second; }
    return nullptr;
}

void set_op_factory(const std::string &name, op_factory_func f) {
    auto &op_map = get_op_factory_map();
    COMPILE_ASSERT(op_map.find(name) == op_map.end(),
            "The op has already registered!");
    op_map[name] = f;
}

float sc_op::get_gflop() {
    return 0.0f;
}

std::vector<int> sc_op::get_impl_dispatch_candidates(const context_ptr &ctx) {
    return {};
}

reflection::shared_general_object_t sc_op::get_dynamic_runtime_info() {
    return reflection::shared_general_object_t();
}

namespace graph {
std::string decay_quantized_op_name(const std::string &op_name) {
    bool is_quantized = utils::string_startswith(op_name, "quantized");
    std::string qstring = "quantized_";
    std::string decay_op_name = is_quantized
            ? op_name.substr(qstring.size(), op_name.size() - qstring.size())
            : op_name;
    return decay_op_name;
}

void get_logical_tensors(
        ltensors *ins, const std::vector<graph_tensor_ptr> &flts) {
    ins->reserve(flts.size());
    for (auto &in : flts) {
        ins->emplace_back(in->details_);
    }
}

expr tensor_detail_to_ir_tensor(sc_graph_t &graph, const std::string &name,
        const logical_tensor_t &tsrd) {
    auto blocking_dims = tsrd.get_blocking_dims();
    auto strides = tsrd.get_strides();
    COMPILE_ASSERT(blocking_dims.size() == strides.size(),
            "Dims and strides does not match.");
    auto blocking_exprs = tsrd.get_blocking_dims_expr(graph);
    auto tsr = builder::make_stensor(name, blocking_exprs,
            tsrd.get_strides_expr(graph), tsrd.dtype_, address_space::automatic,
            nullptr);
    tsr->attr().set(
            attr_keys::plain_dims, graph.dims_to_expr(tsrd.get_plain_dims()));
    if (graph.is_dynamic()) { tsr->attr().set(attr_keys::always_trans, true); }
    return tsr;
}

std::vector<expr> tensor_detail_to_ir_tensor(sc_graph_t &graph,
        const std::string &name_prefix,
        const std::vector<logical_tensor_t> &tsrs) {
    std::vector<expr> ret;
    ret.reserve(tsrs.size());
    for (size_t i = 0; i < tsrs.size(); i++) {
        ret.emplace_back(tensor_detail_to_ir_tensor(
                graph, name_prefix + std::to_string(i), tsrs[i]));
    }
    return ret;
}

std::vector<expr> tensor_detail_to_ir_tensor(sc_graph_t &graph,
        const std::string &name_prefix,
        const std::vector<graph_tensor_ptr> &tsrs) {
    std::vector<expr> ret;
    ret.reserve(tsrs.size());
    for (size_t i = 0; i < tsrs.size(); i++) {
        ret.emplace_back(tensor_detail_to_ir_tensor(
                graph, name_prefix + std::to_string(i), tsrs[i]->details_));
    }
    return ret;
}

ltensors extract_detail_from_tensors(
        const std::vector<std::shared_ptr<graph_tensor>> &flts) {
    std::vector<logical_tensor_t> ret;
    ret.reserve(flts.size());
    for (auto &in : flts) {
        ret.emplace_back(in->details_);
    }
    return ret;
}

sc_graph_t make_single_op_graph(const std::string &opname,
        const std::vector<graph_tensor_ptr> &inputs,
        const std::vector<graph_tensor_ptr> &outputs, const any_map_t &attr) {
    sc_graph_t ret;
    ret.make_input(inputs);
    auto op = ret.make(opname, inputs, outputs, attr);
    ret.make_output(op->get_outputs());
    return ret;
}

} // namespace graph
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
std::size_t hash<dnnl::impl::graph::gc::logical_tensor_t>::operator()(
        const dnnl::impl::graph::gc::logical_tensor_t &k) const {
    size_t seed = 0;
    dnnl::impl::graph::gc::hash_combine(seed, k.dtype_);
    dnnl::impl::graph::gc::hash_combine(seed, k.format_);
    for (size_t i = 0; i < k.plain_dims_.size(); i++) {
        dnnl::impl::graph::gc::hash_combine(seed, k.plain_dims_[i]);
    }
    return seed;
}
} // namespace std
