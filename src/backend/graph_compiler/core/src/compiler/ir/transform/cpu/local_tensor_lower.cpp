/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include "local_tensor_lower.hpp"
#include <vector>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/visitor.hpp>
#include <util/utils.hpp>

namespace sc {

SC_DECL_PASS_INFO(local_tensor_lowering_cpu,
        SC_PASS_DEPENDS_ON(constant_folder, buffer_scheduler, tensor_init,
                module_globals_resolver),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static func_t set_noalias_function(func_t f) {
    f->attr()[function_attrs::no_alias] = true;
    return f;
}

func_t get_cpu_temp_malloc_func(bool is_thread_local) {
    static func_t f_global = set_noalias_function(
            builder::_decl_func("sc_aligned_malloc", datatypes::pointer,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("size", datatypes::index)}));
    static func_t f_local = set_noalias_function(
            builder::_decl_func("sc_thread_aligned_malloc", datatypes::pointer,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("size", datatypes::index)}));
    return is_thread_local ? f_local : f_global;
}

func_t get_cpu_temp_free_func(bool is_thread_local) {
    static func_t f_global
            = builder::_decl_func("sc_aligned_free", datatypes::void_t,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("ptr", datatypes::pointer)});
    static func_t f_local
            = builder::_decl_func("sc_thread_aligned_free", datatypes::void_t,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("ptr", datatypes::pointer)});
    return is_thread_local ? f_local : f_global;
}

class tensor_lower_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    size_t threshold_;
    expr cur_rtl_ctx_;
    // the defined tensor stack. The first dimension is for nested stmts. The
    // second is for ordering the tensors defined in the same scope
    std::vector<std::vector<expr>> defined_tsr_;

    // not interested in expr
    expr_c dispatch(expr_c v) override { return v; }

    stmt_c visit(define_c v) override {
        if (!v->var_.isa<tensor>() || v->linkage_ != linkage::local
                || v->init_.defined()) {
            return v;
        }
        // only interested in local tensors
        auto tsr = v->var_.static_as<tensor>();

        COMPILE_ASSERT(
                tsr->dims_.size() == 1, "tensor_lower_impl needs 1D tensors");
        // check if it is staticaly-shaped and shape is small
        size_t sz = utils::get_sizeof_type(tsr->elem_dtype_);
        expr_c alloc_size;

        bool is_const = true;
        const auto &dim = tsr->dims_[0];
        if (!dim.isa<constant>()) {
            is_const = false;
        } else {
            sz *= get_const_as_int(dim.static_as<constant>());
        }
        if (is_const && sz <= threshold_) {
            // if the tensor is small enough
            return v;
        }
        alloc_size = is_const
                ? expr(sz)
                : auto_caster_t()(tsr->dims_[0]
                        * utils::get_sizeof_type(tsr->elem_dtype_));

        bool thread_loca = tsr->attr_
                && tsr->attr_->get_or_else("is_thread_buffer", false);
        // a large local tensor/dynamic tensor
        expr initv = builder::make_call(get_cpu_temp_malloc_func(thread_loca),
                {cur_rtl_ctx_, alloc_size});
        defined_tsr_.back().emplace_back(tsr);
        return copy_attr(*v,
                builder::make_var_tensor_def_unattached(
                        tsr, v->linkage_, initv));
    }

    stmt_c visit(stmts_c v) override {
        defined_tsr_.emplace_back();
        auto ret = ir_visitor_t::visit(v);
        auto &current_scope = defined_tsr_.back();
        if (!current_scope.empty()) {
            assert(!ret.ptr_same(v));
            auto &seq = ret.checked_as<stmts>()->seq_;
            bool is_ret = !seq.empty() && seq.back().isa<returns>();
            for (auto itr = current_scope.rbegin(); itr != current_scope.rend();
                    ++itr) {
                bool thread_loca = (*itr)->attr_
                        && (*itr)->attr_->get_or_else(
                                "is_thread_buffer", false);
                stmt the_call = builder::make_evaluate_unattached(
                        builder::make_call(get_cpu_temp_free_func(thread_loca),
                                {cur_rtl_ctx_, *itr}));
                if ((*itr)->attr_
                        && (*itr)->attr_->get_or_else(
                                "temp.may_inplace", false)) {
                    assert((*itr).isa<tensor>());
                    auto tsr = (*itr).static_as<tensor>();
                    the_call = builder::make_if_else_unattached(
                            tsr->dims_[0] > UINT64_C(0),
                            builder::make_stmts_unattached({the_call}), stmt());
                }

                // if the last stmt is ret, should insert before it.
                // Otherwise, append to the last position
                auto pos = is_ret ? (seq.end() - 1) : seq.end();
                seq.insert(pos, the_call);
            }
        }
        defined_tsr_.pop_back();
        return ret;
    }
};

func_c local_tensor_lowering_cpu_t::operator()(func_c m) {
    if (m->attr_ && m->attr_->get_or_else(function_attrs::low_level, false)) {
        return m;
    }
    tensor_lower_impl_t impl;
    COMPILE_ASSERT(m->params_.size() >= 2
                    && m->params_.front()->dtype_ == datatypes::pointer,
            "local_tensor_lowering_cpu_t expecting the first function arugment "
            "as a pointer, got: "
                    << m);
    impl.cur_rtl_ctx_ = m->params_.front();
    impl.threshold_ = size_threshold_;
    return impl.dispatch(m);
}

} // namespace sc
