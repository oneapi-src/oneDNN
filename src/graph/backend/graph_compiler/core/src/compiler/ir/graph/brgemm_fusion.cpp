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

#include "../builder.hpp"
#include "../intrinsics.hpp"
#include "../pass/ir_copy.hpp"
#include "../viewer.hpp"
#include "fusion_mgr.hpp"
#include <compiler/ir/graph/fusible_op_utils.hpp>
#include <runtime/microkernel/cpu/brgemm_alg_kind.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
void fusion_manager::break_brgemm_fusion() {
    brg_fusion_reg_.can_register_next_ = false;
}

bool fusion_manager::can_register_brgemm_fusion(const stmt &body) {
    return brg_fusion_reg_.can_register_brgemm_fusion(body);
}

class valid_brgemm_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    using ir_viewer_t::view;
    // currenly only support one valid brgemm in fusion manager
    expr get_valid_brgemm_node() const {
        if (valid_brgemm_count_ == 1) { return valid_brgemm_node_; }
        return expr();
    }
    void operator()(stmt_c v) { ir_viewer_t::dispatch(std::move(v)); }
    void view(intrin_call_c v) override {
        if (v->type_ == intrin_type::brgemm
                || v->type_ == intrin_type::list_brgemm) {
            if (v->intrin_attrs_->get_or_else(
                        intrin_attr::allow_brgemm_fusion, false)) {
                valid_brgemm_count_++;
                valid_brgemm_node_ = v.remove_const();
            }
        }
    }

private:
    int valid_brgemm_count_ = 0;
    expr valid_brgemm_node_;
};

class brgemm_inplace_replacer_t : public ir_inplace_visitor_t {
public:
    using ir_inplace_visitor_t::dispatch_impl;
    using ir_inplace_visitor_t::visit_impl;
    brgemm_inplace_replacer_t(
            std::unordered_map<expr, expr> &rmap, const expr &out_tsr)
        : rmap_(rmap), out_tsr_(out_tsr) {}
    stmt operator()(stmt v) {
        return ir_inplace_visitor_t::dispatch_impl(std::move(v));
    }
    expr visit_impl(intrin_call v) override {
        auto itr = rmap_.find(v);
        if (itr != rmap_.end()) {
            changed_ = true;
            return itr->second;
        }
        return v;
    }
    stmt visit_impl(stmts v) override {
        std::vector<stmt> seq;
        seq.reserve(v->seq_.size());
        bool has_out_tsr = false;
        for (auto &st : v->seq_) {
            auto newst = dispatch_impl(st);
            if (newst.isa<define>()
                    && newst.static_as<define>()->var_.ptr_same(out_tsr_)) {
                seq.insert(seq.begin(), newst);
            } else {
                seq.emplace_back(newst);
            }
        }
        v->seq_ = seq;
        return v;
    }

private:
    std::unordered_map<expr, expr> &rmap_;
    // output tsr for define node search and upshift.
    expr out_tsr_;
};

// Check if the shape is valid for brgemm.
// Valid cases:
//      [m, n] + [1, n]
//      [m, n] + [m, n]
//      [m, n] + [1, 1]
// Invalid cases:
//      [m, n] + [m, 1]
//      [m, n] + [m, 2n]
static bool is_brgemm_valid_shape(const std::vector<expr> &extra_in_shape) {
    if (extra_in_shape.size() < 2) { return false; }
    // brgemm not support [m, 1] broadcast.
    auto n = extra_in_shape[extra_in_shape.size() - 1];
    auto m = extra_in_shape[extra_in_shape.size() - 2];
    if (n.isa<constant>() && get_expr_as_int(n) == UINT64_C(1)
            && !(m.isa<constant>() && get_expr_as_int(m) == UINT64_C(1))) {
        return false;
    }
    for (size_t i = 0; i < extra_in_shape.size() - 2; i++) {
        auto &shape = extra_in_shape[i];
        if (!(shape.isa<constant>() && get_expr_as_int(shape) == INT64_C(1))) {
            return false;
        }
    }
    return true;
}

static sc_data_etype get_expr_etype(expr in) {
    while (in.isa<tensorptr>()) {
        in = in.static_as<tensorptr>()->base_->ptr_;
    }
    assert(in.isa<tensor>());
    return etypes::get_pointer_element(
            in.static_as<tensor>()->dtype_.type_code_);
}

static bool register_bias(
        brgemm_fusion_register *reg, sc_data_type_t dtype, const expr &bias) {
    if (reg->data_.at(brgemm::bias)->equals(get_ir_null())) {
        brgemm::bias_op_t op(dtype.as_etype());
        reg->setting_.emplace_back(op);
        reg->data_.at(brgemm::bias) = bias;
        return true;
    }
    return false;
}

static bool register_scales(brgemm_fusion_register *reg, const expr &scales) {
    if (reg->data_.at(brgemm::scales)->equals(get_ir_null())) {
        brgemm::scale_op_t op;
        reg->setting_.emplace_back(op);
        reg->data_.at(brgemm::scales) = scales;
        return true;
    }
    return false;
}

static bool register_zp(brgemm_fusion_register *reg, const expr &compen,
        const brgemm::alg_kind_t &alg) {
    brgemm::postop_data_kind dkind;
    if (alg == brgemm::a_zp) {
        dkind = brgemm::a_zp_compensations;
    } else if (alg == brgemm::b_zp) {
        dkind = brgemm::b_zp_compensations;
    } else {
        assert(alg == brgemm::c_zp);
        dkind = brgemm::c_zp_values;
    }
    if (reg->data_.at(dkind)->equals(get_ir_null())) {
        brgemm::zp_op_t op(alg);
        reg->setting_.emplace_back(op);
        reg->data_.at(dkind) = compen;
        return true;
    }
    return false;
}

static bool register_unary(brgemm_fusion_register *reg,
        brgemm::alg_kind_t alg_kind, const sc_op_ptr &op) {
    // todo: currenly only add relu for test.
    switch (alg_kind) {
        case brgemm::alg_kind_t::eltwise_relu: {
            brgemm::elt_op_t op2(alg_kind, 1.f, 0.f);
            reg->setting_.emplace_back(op2);
            return true;
        }
        default: return false;
    }
}

static bool register_binary(brgemm_fusion_register *reg,
        brgemm::alg_kind_t alg_kind, const sc_op_ptr &op, const expr &extra_in,
        const std::vector<expr> &extra_in_shape) {
    if (!is_brgemm_valid_shape(extra_in_shape)) { return false; }
    if (!reg->data_.at(brgemm::binary_post_ops_rhs)->equals(get_ir_null())) {
        return false;
    }
    sc_dims dims = get_expr_to_dims(extra_in_shape);
    assert(dims.size() >= 2);
    std::vector<int> int_dims(dims.end() - 2, dims.end());
    auto dtype = get_expr_etype(extra_in);
    // todo: currently only support add/sub/mul/div.
    switch (alg_kind) {
        case brgemm::binary_add:
        case brgemm::binary_sub:
        case brgemm::binary_mul:
        case brgemm::binary_div: {
            brgemm::bin_op_t op2(alg_kind, int_dims.data(), dtype);
            reg->setting_.emplace_back(op2);
            reg->data_.at(brgemm::binary_post_ops_rhs) = extra_in;
            return true;
        }
        default: return false;
    }
}

// always register success. output dtype setting is always the first setting op.
static bool register_out_dtype(
        brgemm_fusion_register *reg, const expr &output) {
    auto dtype = get_expr_etype(output);
    if (!utils::is_one_of(dtype, sc_data_etype::U8, sc_data_etype::S8,
                sc_data_etype::S32, sc_data_etype::BF16, sc_data_etype::F32)) {
        return false;
    }
    if (reg->setting_.empty()
            || reg->setting_[0].empty_op_.alg_ != brgemm::out_dtype) {
        brgemm::out_op_t op(dtype);
        reg->setting_.insert(reg->setting_.begin(), op);
    } else {
        reg->setting_[0].out_op_.dtype_ = dtype;
    }
    return true;
}

static void update_data_c_ptr(brgemm_fusion_register *reg, expr output) {
    if (!reg->data_.at(brgemm::binary_post_ops_rhs)->equals(get_ir_null())) {
        reg->data_.at(brgemm::data_C_ptr) = std::move(output);
    }
}

bool brgemm_fusion_register::register_op_infos(const sc_op_ptr &op,
        const expr &output, const expr &extra_in,
        const std::vector<expr> &extra_in_shape) {
    if (setting_.size() >= brgemm::postops_setting_t::max_postops_num) {
        SC_WARN << "Op " << op->op_name_
                << " can not be inserted into brgemm fusion as the number of "
                   "postops is full: "
                << setting_.size();
        return false;
    }
    auto brg_op = op->dyn_cast<op_traits::brgemm_fusion_acceptable_t>();
    // can not fuse in brgemm, return false
    if (!brg_op || !brg_op->fuse_in_brgemm_
            || brg_op->alg_kind_ == brgemm::alg_kind_undef) {
        return false;
    }
    bool status = false;
    if (brg_op->alg_kind_ == brgemm::bias_add) {
        status = register_bias(
                this, op->get_inputs()[1]->details_.dtype_, extra_in);
    } else if (brg_op->alg_kind_ == brgemm::out_scales) {
        status = register_scales(this, extra_in);
    } else if (brg_op->alg_kind_ == brgemm::out_dtype) {
        status = register_out_dtype(this, output);
    } else if (brg_op->alg_kind_ == brgemm::a_zp) {
        status = register_zp(this, extra_in, brgemm::a_zp);
    } else if (brg_op->alg_kind_ == brgemm::b_zp) {
        status = register_zp(this, extra_in, brgemm::b_zp);
    } else if (brg_op->alg_kind_ == brgemm::c_zp) {
        status = register_zp(this, extra_in, brgemm::c_zp);
    } else if (brg_op->alg_kind_ >= brgemm::eltwise_begin
            && brg_op->alg_kind_ <= brgemm::eltwise_end) {
        status = register_unary(this, brg_op->alg_kind_, op);
    } else if (brg_op->alg_kind_ >= brgemm::binary_begin
            && brg_op->alg_kind_ <= brgemm::binary_end) {
        status = register_binary(
                this, brg_op->alg_kind_, op, extra_in, extra_in_shape);
    } else {
        COMPILE_ASSERT(false, "Unsupported brgemm fusion op kind.");
    }
    if (status) {
        last_out_ = output;
        status = register_out_dtype(this, output);
        update_data_c_ptr(this, output);
    }
    COMPILE_ASSERT(
            setting_.size() <= brgemm::postops_setting_t::max_postops_num,
            "Current number of postops("
                    << setting_.size() << ") should be less than max number("
                    << brgemm::postops_setting_t::max_postops_num << ").");
    return status;
}

bool brgemm_fusion_register::can_register_brgemm_fusion(const stmt &body) {
    valid_brgemm_finder_t finder;
    finder(body);
    valid_brgemm_node_ = finder.get_valid_brgemm_node();
    return valid_brgemm_node_.defined();
}

void brgemm_fusion_register::reset() {
    can_register_next_ = true;
    last_out_ = expr();
    valid_brgemm_node_ = expr();
    setting_.clear();
    data_ = builtin::create_initialed_postops_data();
}

stmt brgemm_fusion_register::remake_brgemm_intrinsic_by_fusion(
        stmt body, expr c_buf) const {
    COMPILE_ASSERT(
            valid_brgemm_node_.defined(), "Should return valid remake brgemm.");
    auto node = valid_brgemm_node_.checked_as<intrin_call>();
    assert(node->type_ == intrin_type::brgemm
            || node->type_ == intrin_type::list_brgemm);
    auto extra_args = node->intrin_attrs_->get<brgemm_args::extra_args_t>(
            intrin_attr::brgemm_extras);
    assert(extra_args.postops_setting_.empty());
    if (setting_.empty()) { return body; }
    extra_args.postops_setting_ = setting_;
    assert(last_out_.isa<tensorptr>());
    extra_args.dtype_C_ = last_out_.static_as<tensorptr>()
                                  ->base_->ptr_.static_as<tensor>()
                                  ->elem_dtype_;
    int basic_arg_size = node->type_ == intrin_type::brgemm
            ? brgemm_args::NUM_BASIC_ARGS_STRIDE
            : brgemm_args::NUM_BASIC_ARGS_LIST;
    // layout of node->args (full args):
    //    | basic_args | postops_data list(11 elems) | c_buf | bdmask_idx
    auto new_args = std::vector<expr>(
            node->args_.begin(), node->args_.begin() + basic_arg_size);
    new_args.insert(new_args.end(), data_.begin(), data_.end());
    if (!c_buf.defined()) { c_buf = get_ir_null(); }
    new_args.emplace_back(c_buf);
    new_args.emplace_back(node->args_.back());
    new_args[brgemm_args::C] = last_out_;
    assert(new_args.size() == node->args_.size());
    auto new_node = copy_attr(*node,
            make_expr<intrin_call_node>(node->type_, new_args,
                    any_map_t {{intrin_attr::brgemm_extras, extra_args},
                            {intrin_attr::allow_brgemm_fusion,
                                    node->intrin_attrs_->get_or_else(
                                            intrin_attr::allow_brgemm_fusion,
                                            false)}}));
    std::unordered_map<expr, expr> rmap = {{node, new_node}};
    brgemm_inplace_replacer_t replacer(
            rmap, last_out_.static_as<tensorptr>()->base_->ptr_);
    return replacer(body);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
