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
#include "local_tensor_lower.hpp"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/visitor.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
SC_MODULE(pass.local_tensor_lower);

SC_DECL_PASS_INFO(local_tensor_lowering_cpu,
        SC_PASS_DEPENDS_ON(constant_folder, buffer_scheduler, tensor_init,
                module_globals_resolver, simple_loop_invariant_code_motion),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

static func_t set_noalias_function(func_t f) {
    f->attr()[function_attrs::no_alias] = true;
    return f;
}

func_t get_acquire_const_cache_func() {
    static func_t f_global = set_noalias_function(
            builder::_decl_func("sc_acquire_const_cache", datatypes::pointer,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("cacheptr", datatypes::index),
                            _arg_("size", datatypes::index),
                            _arg_("outinited", datatypes::s32, {1})}));
    return f_global;
}

func_t get_release_const_cache_func() {
    static func_t f_global
            = builder::_decl_func("sc_release_const_cache", datatypes::void_t,
                    {_arg_("stream", datatypes::pointer),
                            _arg_("cacheptr", datatypes::index),
                            _arg_("ptr", datatypes::pointer)});
    return f_global;
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

namespace local_tsr_lower {
// record the base tensor and the offset of a tensor after buffer-scheduling
struct tensor_base_offset_t {
    expr base_;
    int64_t start_offset_;
    int64_t end_offset_; // inclusive offset
};

bool tensor_less_than_by_name(const expr &v1, const expr &v2) {
    return v1.checked_as<tensor>()->name_ < v2.checked_as<tensor>()->name_;
}

// the "trace" for our pass to scan the positions in the base tensor to decide
// which tensors are alias
struct tensor_trace_t {
    expr tsr_;
    int64_t offset_;
    bool is_end_;
    bool operator<(const tensor_trace_t &v) const {
        if (offset_ < v.offset_) { return true; }
        if (offset_ > v.offset_) { return false; }
        // offset == v.offset_
        if (is_end_ != v.is_end_) {
            // let "start" trace < "end" trace to let them overlap
            // if is_end, then v.is_end_ is false, and we should let v< *this
            return !is_end_;
        }
        // offset == v.offset_ && is_end_ == v.is_end_
        // the same position trace, sort by tensor name
        return tensor_less_than_by_name(tsr_, v.tsr_);
    }
};

} // namespace local_tsr_lower

static const char *shared_const_handle_name = "__shared_const_handle";

using namespace local_tsr_lower;

class tensor_lower_impl_t : public ir_visitor_t {
public:
    using ir_visitor_t::dispatch;
    size_t threshold_;
    expr cur_rtl_ctx_;
    // the defined tensor stack. The first dimension is for nested stmts. The
    // second is for ordering the tensors defined in the same scope
    std::vector<std::vector<expr>> defined_tsr_;
    // tensor -> [base, start_offset, end_offset]
    std::unordered_map<expr, tensor_base_offset_t> scheduled_tensor_position_;
    // the ordered tensors in scheduled_tensor_position_
    std::vector<expr> scheduled_tensors_;
    // tenosr -> hoisted base tensor in outer loop
    std::unordered_map<expr, expr> hoisted_tensor_map_;
    expr shared_const_handles, tsr_is_init;

    // not interested in expr
    expr_c dispatch(expr_c v) override { return v; }

    int64_t get_tensor_size(const tensor &tsr) {
        COMPILE_ASSERT(tsr->dims_.size() == 1 && tsr->dims_[0].isa<constant>(),
                "Expecting 1D constant sized tensor");
        return get_const_as_int(tsr->dims_[0].static_as<constant>())
                * (int64_t)utils::get_sizeof_type(tsr->elem_dtype_);
    }

    stmt_c visit(define_c v) override {
        if (!v->var_.isa<tensor>()) { return v; }
        if (v->linkage_ == linkage::local
                && v->var_.static_as<tensor>()->name_
                        == shared_const_handle_name) {
            shared_const_handles = v->var_;
        }
        if (v->linkage_ != linkage::local) { return v; }
        // only interested in local tensors
        auto tsr = v->var_.static_as<tensor>();
        if (tsr->name_ == "__is_init") { tsr_is_init = tsr; }
        if (v->init_.defined()) {
            if (v->init_.isa<tensorptr>()) {
                auto tptr = v->init_.static_as<tensorptr>();
                auto attr = tptr->base_->ptr_->attr_.get();
                if (attr && attr->get_or_else("hoisted", false)) {
                    // the tensor is a tensor based on a hoisted tensor
                    COMPILE_ASSERT(tptr->base_->ptr_.isa<tensor>()
                                    && !tptr->base_->ptr_.ptr_same(tsr),
                            "Expecting a tensor on the base of hoisted tensor");
                    hoisted_tensor_map_[tsr] = tptr->base_->ptr_;
                } else if (attr
                        && attr->get_or_else(
                                attr_keys::can_be_scheduled, false)) {
                    auto base = tptr->base_->ptr_;
                    COMPILE_ASSERT(tptr->base_->idx_.size() == 1
                                    && tptr->base_->idx_[0].isa<constant>(),
                            "Expecting 1D constant sized tensor");
                    auto offset = get_const_as_int(
                            tptr->base_->idx_[0].static_as<constant>());
                    COMPILE_ASSERT(tsr->dims_.size() == 1
                                    && tsr->dims_[0].isa<constant>(),
                            "Expecting 1D constant sized tensor");
                    auto tsr_size = get_tensor_size(tsr);
                    COMPILE_ASSERT(tsr_size > 0, "Bad size of tensor");
                    // recursively find the base tensor
                    for (;;) {
                        auto itr = scheduled_tensor_position_.find(base);
                        if (itr != scheduled_tensor_position_.end()) {
                            base = itr->second.base_;
                            offset += itr->second.start_offset_;
                        } else {
                            break;
                        }
                    }
                    scheduled_tensor_position_.insert(std::make_pair(tsr,
                            tensor_base_offset_t {
                                    base, offset, offset + tsr_size - 1}));
                    scheduled_tensors_.emplace_back(tsr);
                }
            }
            return v;
        }
        COMPILE_ASSERT(
                tsr->dims_.size() == 1, "tensor_lower_impl needs 1D tensors");
        // check if it is staticaly-shaped and shape is small
        size_t sz = utils::get_sizeof_type(tsr->elem_dtype_);
        expr_c alloc_size;

        bool is_const = true;
        const auto &dim = tsr->dims_[0];
        auto shared_cached_buffer = any_map_t::fetch_or_else<
                std::shared_ptr<cached_const_graph_tensor>>(
                v->var_->attr_.get(), attr_keys::shared_const, nullptr);
        if (!dim.isa<constant>()) {
            is_const = false;
        } else {
            sz *= get_const_as_int(dim.static_as<constant>());
        }
        if (is_const && sz <= threshold_ && !shared_cached_buffer
                && /*and the tensor is not marked runtime_stack_alloc*/
                !(tsr->attr_
                        && tsr->attr_->get_or_else(
                                attr_keys::runtime_stack_alloc, false))) {
            // if the tensor is small enough
            return v;
        }
        alloc_size = is_const
                ? expr(sz)
                : auto_caster_t()(tsr->dims_[0]
                        * utils::get_sizeof_type(tsr->elem_dtype_));
        expr initv;
        if (shared_cached_buffer) {
            // handle special local tensors: shared cached const
            auto handle_idx = any_map_t::fetch_or_null<size_t>(
                    v->var_->attr_.get(), attr_keys::shared_const_base_idx);
            COMPILE_ASSERT(handle_idx, "Expecting attr shared_const_base_idx");
            COMPILE_ASSERT(shared_const_handles.defined(),
                    "Expecting shared_const_handles defined");
            if (shared_cached_buffer->buf_base_->is_lazy_) {
                COMPILE_ASSERT(
                        tsr_is_init.defined(), "Expecting __is_init defined");
                initv = builder::make_call(get_acquire_const_cache_func(),
                        {cur_rtl_ctx_,
                                shared_const_handles[uint64_t(*handle_idx)],
                                alloc_size, tsr_is_init});
            } else {
                initv = builder::make_reinterpret(
                        shared_const_handles[uint64_t(*handle_idx)],
                        datatypes::pointer);
            }
        } else {
            bool thread_loca = tsr->attr_
                    && tsr->attr_->get_or_else("is_thread_buffer", false);
            // a large local tensor/dynamic tensor
            initv = builder::make_call(get_cpu_temp_malloc_func(thread_loca),
                    {cur_rtl_ctx_, alloc_size});
        }
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
                stmt the_call;
                if (auto cached_buffer = any_map_t::fetch_or_else<
                            std::shared_ptr<cached_const_graph_tensor>>(
                            (*itr)->attr_.get(), attr_keys::shared_const,
                            nullptr)) {
                    auto handle_idx = any_map_t::fetch_or_null<size_t>(
                            (*itr)->attr_.get(),
                            attr_keys::shared_const_base_idx);
                    COMPILE_ASSERT(
                            handle_idx, "Expecting attr shared_const_base_idx");
                    COMPILE_ASSERT(shared_const_handles.defined(),
                            "Expecting shared_const_handles defined");
                    if (cached_buffer->buf_base_->is_lazy_) {
                        the_call = builder::make_evaluate_unattached(
                                builder::make_call(
                                        get_release_const_cache_func(),
                                        {cur_rtl_ctx_,
                                                shared_const_handles[uint64_t(
                                                        *handle_idx)],
                                                *itr}));
                    } else {
                        // compile-time constants, do not need to call release
                        continue;
                    }
                } else {
                    bool thread_loca = (*itr)->attr_
                            && (*itr)->attr_->get_or_else(
                                    "is_thread_buffer", false);
                    the_call = builder::make_evaluate_unattached(
                            builder::make_call(
                                    get_cpu_temp_free_func(thread_loca),
                                    {cur_rtl_ctx_, *itr}));
                }
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

static std::vector<std::shared_ptr<alias_info::tensor_alias_identity_t>>
mark_alias_for_scheduled_tensors(
        const std::unordered_map<expr, tensor_base_offset_t>
                &scheduled_tensor_position,
        const std::unordered_map<expr, expr> &hoisted_tensor_map,
        const std::vector<expr> &scheduled_tensors) {
    if (scheduled_tensor_position.empty()) return {};
    std::vector<std::shared_ptr<alias_info::tensor_alias_identity_t>> ret;
    std::unordered_map<expr, std::vector<tensor_trace_t>> base_tsr_to_traces;
    for (auto &kv : scheduled_tensor_position) {
        auto &tsr = kv.first;
        auto &base_info = kv.second;
        base_tsr_to_traces[base_info.base_].emplace_back(
                tensor_trace_t {tsr, base_info.start_offset_, false});
        base_tsr_to_traces[base_info.base_].emplace_back(
                tensor_trace_t {tsr, base_info.end_offset_, true});
        ret.emplace_back(alias_info::get_or_create_alias_info(*tsr));
    }
    // sort the base tensors by name to have stable result
    using pair_expr_vec_trace = std::pair<expr, std::vector<tensor_trace_t> *>;
    std::vector<pair_expr_vec_trace> sorted_base_tsr_to_traces;
    sorted_base_tsr_to_traces.reserve(base_tsr_to_traces.size());
    for (auto &kv : base_tsr_to_traces) {
        sorted_base_tsr_to_traces.emplace_back(kv.first, &kv.second);
    }
    std::sort(sorted_base_tsr_to_traces.begin(),
            sorted_base_tsr_to_traces.end(),
            [](const pair_expr_vec_trace &v1, const pair_expr_vec_trace &v2) {
                return tensor_less_than_by_name(v1.first, v2.first);
            });
    int64_t clique_id = 1;
    for (auto &kv : sorted_base_tsr_to_traces) {
        auto &base = kv.first;
        auto &traces = *kv.second;
        if (traces.empty()) { continue; }
        // the traces are sorted by offset, then by is_end, then by tensor name
        std::sort(traces.begin(), traces.end());
        // the algorithm:
        // think there is a cursor pointing to an offset of the base tensor. We
        // move the cursor from 0 offset to max offset of the tensor. The traces
        // marks the start and end of scheduled tensors and are sorted by the
        // offset. So when the cursor points to an offset, we can know which
        // tensors contain the position.

        // the set of tensors that contain the current cursor
        alias_info::alias_set_t cur_alias;
        std::shared_ptr<alias_info::alias_set_t> cur_clique;
        for (auto &trace : traces) {
            auto alias_id = alias_info::get_or_create_alias_info(*trace.tsr_);
            if (!trace.is_end_) {
                if (!cur_clique) {
                    cur_clique = cur_alias.copy();
                    cur_clique->id_ = clique_id;
                    clique_id++;
                }
                cur_alias.set_.insert(alias_id);
                alias_id->add_to_clique(cur_clique);
            } else {
                cur_alias.set_.erase(alias_id);
                // a tensor leaves the clique, need to make a new clique when
                // another tensor joins
                cur_clique = nullptr;
            }
        }
    }

    // if scheduled tensor is based on a hoisted tensor, it should have same
    // alias with the hoisted base. Recurisvely find the hoisted base
    std::vector<std::pair<alias_info::tensor_alias_identity_t *,
            alias_info::alias_set_t *>>
            to_add_hoisted_alias;
    // the vector to_add_hoisted_alias is to remember which scheduled tensor
    // need to copy which alias set of a base hoisted tensor
    for (auto &tsr : scheduled_tensors) {
        auto itr = scheduled_tensor_position.find(tsr);
        auto &base_info = itr->second;
        auto aid_tsr = alias_info::get_or_create_alias_info(*tsr);
        expr base = base_info.base_;
        for (;;) {
            auto itr = hoisted_tensor_map.find(base);
            if (itr != hoisted_tensor_map.end()) {
                auto hoisted = itr->second;
                auto sche_itr = scheduled_tensor_position.find(hoisted);
                if (sche_itr != scheduled_tensor_position.end()) {
                    if (auto aid_base = alias_info::get_alias_info(*hoisted)) {
                        // the tensor "tsr" is based on hoisted tensor "base"
                        // and "tsr" should have same aliases of "base"
                        for (auto &aset : aid_base->alias_cliques_) {
                            if (!aset->set_.has(aid_tsr)) {
                                to_add_hoisted_alias.emplace_back(
                                        aid_tsr.get(), aset.get());
                            }
                        }
                    }
                    auto &hoisted_base_info = sche_itr->second;
                    base = hoisted_base_info.base_;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    for (auto &kv : to_add_hoisted_alias) {
        auto &aid_tsr = kv.first;
        auto &aset = kv.second;
        auto cli_copy = aset->copy();
        aid_tsr->add_to_clique(cli_copy);
        cli_copy->id_ = clique_id;
        clique_id++;
    }
    // output to INFO log
    if (auto sc_stream_temp
            = runtime::get_info_logging_stream(__sc_module_name)) {
        for (auto &kv : scheduled_tensor_position) {
            (*sc_stream_temp.stream_) << kv.first << ':';
            if (auto aid = alias_info::get_alias_info(*kv.first)) {
                for (auto &aset : aid->alias_cliques_) {
                    (*sc_stream_temp.stream_) << aset->id_ << ',';
                }
            }
            (*sc_stream_temp.stream_) << '\n';
        }
    }
    return ret;
}

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
    auto ret = impl.dispatch(m);
    if (!impl.scheduled_tensor_position_.empty()) {
        auto alias_ids = mark_alias_for_scheduled_tensors(
                impl.scheduled_tensor_position_, impl.hoisted_tensor_map_,
                impl.scheduled_tensors_);
        std::const_pointer_cast<func_base>(ret)->attr()["alias_sets"]
                = std::move(alias_ids);
    }
    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
