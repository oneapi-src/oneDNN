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
#include <string>
#include <utility>
#include <vector>
#include "tensor_init.hpp"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/parallel_workload_attr.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/config.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_DECL_PASS_INFO(tensor_init,
        SC_PASS_DEPENDS_ON(tensor_shrinker, constant_folder),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

class tensor_init_impl_t : public ir_visitor_t {
public:
    context_ptr ctx_;
    tensor_init_impl_t(const context_ptr &ctx) : ctx_(ctx) {}
    using ir_visitor_t::dispatch;
    using ir_visitor_t::visit;
    std::vector<stmt_c> insert_;
    bool is_parallel_ = false;
    bool no_parallel_ = false;

    stmt_c visit(for_loop_c v) override {
        bool old_parallel = is_parallel_;
        is_parallel_ |= v->kind_ == for_type::PARALLEL;
        auto ret = ir_visitor_t::visit(v);
        is_parallel_ = old_parallel;
        return ret;
    }

    stmt_c visit(stmts_c v) override {
        std::vector<stmt_c> seq = std::move(insert_);
        bool changed = false;
        changed = !seq.empty();

        for (auto &s : v->seq_) {
            auto new_st = dispatch(s);
            seq.emplace_back(new_st);
            if (s.isa<define>()) {
                auto def = new_st.static_as<define>();
                if (def->linkage_ == linkage::local) {
                    insert_tensor_zero_init(seq, def->var_);
                }
            }
            changed |= !seq.back().ptr_same(s);
        }
        if (!changed) { return v; }
        return copy_attr(*v, builder::make_stmts_unattached(seq));
    }

    static void set_attr(const stmt &v, const tensor &tsr) {
        v->attr()[parallel_workload::attr_workload_number]
                = parallel_workload::write_weight
                * utils::get_sizeof_etype(tsr->elem_dtype_.type_code_);
    }

    void insert_tensor_zero_init(std::vector<stmt_c> &seq, const expr &e) {
        auto tsr = e.as<tensor>();
        if (tsr.defined() && tsr->init_value_
                && (!tsr->attr_
                        || !tsr->attr_->has_key(attr_keys::shared_const))) {
            union_val val;
            if (tsr->init_value_
                    == tensor_node::get_zero_tensor_initializer()) {
                val.u64 = 0;
            } else {
                COMPILE_ASSERT(tsr->init_value_->size_ == sizeof(val),
                        "Tensor initializer of local tensors must be size of "
                        "union_val");
                val = *reinterpret_cast<union_val *>(tsr->init_value_->data_);
            }
            // if already in parallel, or the func is marked no_parallel, we
            // cannot make parallel-for
            // todo(anyone): need check and reset non-parallel type for all
            // loops when threads == 1
            bool can_parallel = !is_parallel_ && !no_parallel_
                    && runtime_config_t::get().get_num_threads() > 1;
            assert(tsr->dims_.size() == 1);
            auto dim = tsr->dims_[0];
            if (dim.isa<constant>()) {
                auto len = get_const_as_int(dim.static_as<constant>());
                auto step = ctx_->get_max_vector_lanes(
                        tsr->elem_dtype_.type_code_);
                auto remainder = len % step;
                if (len >= step) {
                    expr itr = builder::make_var(datatypes::index, "itr");
                    stmt assign = builder::make_assign_unattached(
                            e[span_t({itr}, step)],
                            make_expr<constant_node>(val,
                                    sc_data_type_t {tsr->elem_dtype_.type_code_,
                                            step}));
                    set_attr(assign, tsr);
                    stmt body = builder::make_stmts_unattached({assign});
                    body = builder::make_for_loop_unattached(itr, UINT64_C(0),
                            static_cast<uint64_t>(len - remainder),
                            static_cast<uint64_t>(step), body, true,
                            can_parallel ? for_type::PARALLEL
                                         : for_type::NORMAL);
                    seq.emplace_back(body);
                }
                if (remainder != 0) {
                    expr itr = builder::make_var(datatypes::index, "itr_rem");
                    stmt assign = builder::make_assign_unattached(e[{itr}],
                            make_expr<constant_node>(val, tsr->elem_dtype_));
                    set_attr(assign, tsr);
                    stmt body = builder::make_stmts_unattached({assign});
                    body = builder::make_for_loop_unattached(itr,
                            static_cast<uint64_t>(len - remainder),
                            static_cast<uint64_t>(len),
                            static_cast<uint64_t>(1), body, true,
                            for_type::NORMAL);
                    seq.emplace_back(body);
                } else {
                    bool must_tensor2var = tsr->attr_
                            && tsr->attr_->get_or_else(
                                    attr_keys::must_tensor2var, false);
                    // no remainder loop, try unroll
                    if (len / step <= 8 || must_tensor2var) {
                        seq.back()
                                .remove_const()
                                ->attr()[stmt_attr_key::unroll_loop]
                                = 0;
                    }
                }
            } else {
                expr itr = builder::make_var(datatypes::index, "itr");
                stmt assign = builder::make_assign_unattached(e[{itr}],
                        make_expr<constant_node>(val, tsr->elem_dtype_));
                set_attr(assign, tsr);
                stmt body = builder::make_stmts_unattached({assign});
                body = builder::make_for_loop_unattached(itr,
                        static_cast<uint64_t>(0), dim, static_cast<uint64_t>(1),
                        body, true,
                        can_parallel ? for_type::PARALLEL : for_type::NORMAL);
                seq.emplace_back(body);
            }
        }
    }

    func_c dispatch(func_c f) override {
        no_parallel_ = f->attr_
                && f->attr_->get_or_else(function_attrs::no_parallel, false);
        for (auto &p : f->params_) {
            insert_tensor_zero_init(insert_, p);
        }
        return ir_visitor_t::dispatch(f);
    }
};

func_c tensor_init_t::operator()(func_c f) {
    tensor_init_impl_t simpl(ctx_);
    return simpl.dispatch(f);
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
