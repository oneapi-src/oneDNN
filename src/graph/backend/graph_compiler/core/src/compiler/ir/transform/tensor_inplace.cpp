/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "buffer_schedule.hpp"
#include "buffer_schedule_utils.hpp"
#include "pointer_alias_info.hpp"
#include "tensor_inplace.hpp"
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/pass_dep_util.hpp>
#include <compiler/ir/transform/tensor_inplace.hpp>
#include <compiler/ir/viewer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <util/any_map.hpp>

SC_MODULE(pass.tensor_inplace);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

std::ostream &operator<<(std::ostream &os, const tensor_inplace_info_t &value) {
    os << '{' << value.used_arg_idx_ << '}';
    return os;
}

SC_DECL_PASS_INFO(tensor_inplace,
        SC_PASS_DEPENDS_ON(
                constant_folder, index_flattener, validator, auto_caster),
        SC_PASS_REQUIRE_STATE(), SC_PASS_REQUIRE_NOT_STATE(),
        SC_PASS_SET_STATE(), SC_PASS_UNSET_STATE());

using namespace special_ticks;

class func_finder_t : public ir_viewer_t {
public:
    using ir_viewer_t::dispatch;
    std::unordered_set<std::shared_ptr<func_base>> funcs_;

    void view(call_c v) override {
        ir_viewer_t::view(v);
        auto funct = std::dynamic_pointer_cast<func_base>(v->func_);
        if (funct) { funcs_.insert(funct); }
    }
};

using inplace_hint_t
        = std::vector<std::pair<int, std::vector<tensor_inplace_info_t>>>;

static void filter_and_sync_inplace_hint(
        const func_base *callee, const func_t &f) {
    if (!callee->attr_) { return; }
    auto hint_in_def = callee->attr_->get_or_null<inplace_hint_t>(
            function_attrs::inplace_hint);
    if (!hint_in_def) { return; }
    auto &params = callee->params_;
    // the alias id is the in-place result selected by
    // buffer_scheduler_t, try to use it to filter the
    // inplace_hint
    for (auto itrkv = hint_in_def->begin(); itrkv != hint_in_def->end();) {
        auto out_arg_idx = itrkv->first;
        auto &hints = itrkv->second;
        if (auto out_alias_id
                = alias_info::get_alias_info(*params.at(out_arg_idx))) {
            for (auto itr_in = hints.begin(); itr_in != hints.end();) {
                auto in_arg_idx = itr_in->used_arg_idx_;
                if (auto in_alias_id
                        = alias_info::get_alias_info(*params.at(in_arg_idx))) {
                    if (out_alias_id->is_alias_of(in_alias_id)) {
                        // if tensors are alias, the inplace happens. continue
                        ++itr_in;
                        continue;
                    }
                }
                itr_in = hints.erase(itr_in);
            }
            ++itrkv;
        } else {
            itrkv = hint_in_def->erase(itrkv);
        }
    }
    f->attr()[function_attrs::inplace_hint] = *hint_in_def;
    f->decl_->attr()[function_attrs::inplace_hint] = *hint_in_def;
    if (!hint_in_def->empty()) {
        SC_MODULE_INFO << "arg inplace: " << f->name_ << " : "
                       << utils::general_print(*hint_in_def);
    }
}

static void get_inplace_args_from_called_funcs(const func_c &f, // caller
        const std::unordered_map<expr_c, tensor_tick_info_t> &ticks,
        // the output arg -> input arg map
        std::vector<std::pair<expr_c, expr_c>> &inplace_pairs) {
    const std::vector<expr> &f_args = f->params_;
    // the in args that already inplaced. (We may not need this, if the upper
    // level framework promises that it will not inplace multiple out arg to one
    // in arg. Keep it for security.)
    std::unordered_set<expr_c> inplaced_in_args;
    for (const auto &st : f->body_.checked_as<stmts>()->seq_) {
        if (!st.isa<evaluate_c>()) { continue; }
        const auto &callee = st.static_as<evaluate_c>()->value_;
        if (!callee.isa<call_c>()) { continue; }
        const auto &func = std::dynamic_pointer_cast<func_base>(
                callee.static_as<call_c>()->func_);
        if (!func || !func->attr_) { continue; }
        const auto &args = callee.static_as<call_c>()->args_;

        auto hint = func->attr_->get_or_null<inplace_hint_t>(
                function_attrs::inplace_hint);
        if (hint) {
            SC_MODULE_INFO << "Get inplace hint size: " << hint->size();
            for (const auto &pair : *hint) {
                // The buffer scheduler does not inplace graph's input and
                // output args, so we need to know decide which input is
                // actually selected to inplace. Use the first for now.
                auto out_arg = args[pair.first];
                if (out_arg.isa<tensorptr>()) {
                    out_arg = get_base_tensor_of(out_arg).dyn_as<expr>();
                }
                if (ticks.find(out_arg) != ticks.end()
                        && ticks.at(out_arg).is_arg_ && out_arg->attr_
                        && out_arg->attr_->has_key("write_buffer")) {
                    // this func out arg is also graph's out arg
                    for (const auto &in_candidate : pair.second) {
                        auto in_arg = args[in_candidate.used_arg_idx_];
                        if (in_arg.isa<tensorptr>()) {
                            in_arg = get_base_tensor_of(in_arg).dyn_as<expr>();
                        }
                        if (ticks.find(in_arg) != ticks.end()
                                && ticks.at(in_arg).is_arg_ && in_arg->attr_
                                && in_arg->attr_->has_key("read_buffer")
                                && !(in_arg->attr_->has_key("write_buffer"))) {
                            // this func in arg is also graph's in arg
                            if (inplaced_in_args.count(in_arg)) { continue; }
                            inplaced_in_args.insert(in_arg);
                            SC_MODULE_INFO << "Inplace hint: #" << pair.first
                                           << " arg: " << out_arg << " with #"
                                           << in_candidate.used_arg_idx_
                                           << " arg: " << in_arg;
                            inplace_pairs.emplace_back(
                                    std::make_pair(out_arg, in_arg));
                            break;
                        }
                    }
                }
            }
        } else {
            SC_MODULE_INFO << "No inplace hint in func, try another way";
            for (const auto &arg : args) {
                if (ticks.find(arg) != ticks.end() && ticks.at(arg).is_arg_
                        && arg->attr_ && arg->attr_->has_key("write_buffer")) {
                    // this arg is an output arg of caller f.
                    COMPILE_ASSERT(std::any_of(f_args.begin(), f_args.end(),
                                           [&arg](const expr &e) {
                                               return e.ptr_same(arg);
                                           }),
                            "Arg is not output arg of caller");
                    for (const auto &f_arg : f_args) {
                        if (std::none_of(args.begin(), args.end(),
                                    [&f_arg](const expr &e) {
                                        return e.ptr_same(f_arg);
                                    })) {
                            // f_arg is an arg of f but not an arg of func
                            if (inplaced_in_args.count(f_arg)) { continue; }
                            inplaced_in_args.insert(f_arg);
                            inplace_pairs.emplace_back(
                                    std::make_pair(arg, f_arg));
                            SC_MODULE_INFO << "Inplace candidate: arg: " << arg
                                           << " with arg: " << f_arg;
                        }
                    }
                } else {
                    // the input args starts, so stop finding.
                    break;
                }
            }
        }
    }
}

void schedule_func_args(const func_c &f,
        // pairs of {out_arg, in_arg} that out_arg can inplace in_arg
        std::vector<std::pair<expr_c, expr_c>> &inplace_map) {
    std::unordered_map<expr_c, tensor_tick_info_t> ticks;
    std::vector<expr_c> defined;
    annotate_ticks(f, ticks, defined);

    // get inplace candidates from each op's perspective
    std::vector<std::pair<expr_c, expr_c>> inplace_pairs;
    get_inplace_args_from_called_funcs(f, ticks, inplace_pairs);
    SC_MODULE_INFO << "Get inplace pairs from called functions, size: "
                   << inplace_pairs.size();

    if (inplace_pairs.empty()) { return; }

    // get inplace hint from buffer lives in main entry f
    // last read tick => tensor, decending order
    std::multimap<int64_t, expr_c, std::greater<int64_t>> last_read_tensor;
    for (auto &itr : ticks) {
        if (itr.second.is_arg_) {
            int64_t &lastread = itr.second.last_read_;
            // if lastread == TICK_NOT_EXIST, then this is an output arg. We
            // want it to be in the former part of the map, so we set it to a
            // great value
            if (lastread == TICK_NOT_EXIST) {
                lastread = std::numeric_limits<int64_t>::max();
            }
            if (lastread != COMPLICATED_ACCESS) {
                last_read_tensor.insert(std::make_pair(lastread, itr.first));
            }
        }
    }

    for (const auto &tsr_tick : ticks) {
        const auto &out_arg = tsr_tick.first;
        // only care about output args
        if (!(out_arg->attr_ && out_arg->attr_->has_key("write_buffer")
                    && !(out_arg->attr_->has_key("read_buffer"))
                    && tsr_tick.second.is_arg_)) {
            continue;
        }
        SC_MODULE_INFO << "Find inplace for out arg: " << out_arg;
        const auto &out_tsr_tick = tsr_tick.second;
        if (out_tsr_tick.last_read_ == COMPLICATED_ACCESS) {
            SC_MODULE_INFO << "Complex access on " << out_arg;
            continue;
        }
        tensor_c out_tsr = out_arg.static_as<tensor_c>();
        assert(out_tsr->dims_.size() == 1);
        if (!out_tsr->dims_[0].isa<constant>()) {
            SC_MODULE_INFO << "Tensor " << out_tsr << " has non-constant shape";
            continue;
        }
        int64_t out_tsr_size
                = get_const_as_int(out_tsr->dims_[0].static_as<constant_c>());

        // find input tensors whose last-read tick is equal or less than current
        // first_access tick
        auto titr = last_read_tensor.lower_bound(out_tsr_tick.first_access_);
        while (titr != last_read_tensor.end()) {
            if (!(titr->second->attr_
                        && titr->second->attr_->has_key("read_buffer")
                        && !(titr->second->attr_->has_key("write_buffer")))) {
                ++titr;
                continue;
            }
            const auto &in_arg = titr->second;
            auto in_tsr = in_arg.checked_as<tensor_c>();
            if (in_tsr.ptr_same(out_tsr)) {
                // this can occur when a tensor is written, but is never read
                ++titr;
                continue;
            }
            const auto &in_tsr_tick = ticks.at(in_tsr);
            // only care about input args
            if (!in_tsr_tick.is_arg_) { continue; }
            SC_MODULE_INFO << "Candidate input arg: " << in_arg;
            if (in_tsr_tick.create_ <= out_tsr_tick.first_access_
                    && in_tsr_tick.delete_ >= out_tsr_tick.delete_
                    && utils::get_sizeof_type(in_tsr->elem_dtype_)
                            == utils::get_sizeof_type(out_tsr->elem_dtype_)) {
                // check that the candidate has no writes during the time range
                // when out_tsr is in use: [out_tsr_tick.first_access_,
                // out_tsr_tick.last_read_]
                if (out_tsr_tick.last_read_ != TICK_NOT_EXIST) {
                    auto lower = in_tsr_tick.writes_.lower_bound(
                            out_tsr_tick.first_access_);
                    auto upper = in_tsr_tick.writes_.upper_bound(
                            out_tsr_tick.last_read_);
                    // lower: the first element >= first_access_
                    // upper: the first element > last_read_
                    if (lower != upper) {
                        // there are writes between first_access and last_read
                        SC_MODULE_INFO << "Write after read: Failed "
                                       << out_tsr->name_ << "->"
                                       << in_tsr->name_ << "@" << *lower;
                        ++titr;
                        continue;
                    }
                }

                // check if the in_tsr is large enough
                assert(in_tsr->dims_.size() == 1);
                int64_t in_tsr_size = get_const_as_int(
                        in_tsr->dims_[0].static_as<constant_c>());
                if (out_tsr_size != in_tsr_size) {
                    ++titr;
                    continue;
                }

                if (std::any_of(inplace_pairs.begin(), inplace_pairs.end(),
                            [&out_arg, &in_arg](std::pair<expr_c, expr_c> &e) {
                                return out_arg.ptr_same(e.first)
                                        && in_arg.ptr_same(e.second);
                            })) {
                    inplace_map.emplace_back(std::make_pair(out_arg, in_arg));
                    SC_MODULE_INFO << "Inplace result: " << out_arg << " use "
                                   << in_arg;
                }
                // It is possible that an out arg can inplace multiple in args.
                // Keep them all in the inplace_map vector.
            }
            ++titr;
        }
    }
}

const_ir_module_ptr tensor_inplace_t::operator()(const_ir_module_ptr f) {
    auto ret = std::make_shared<ir_module_t>(*f);
    auto main_entry = f->get_entry_func();
    for (auto &entry_f : ret->get_contents()) {
        // skip functions that 1) are not main_entry function and 2) are not top
        // level functions
        bool *top_level = nullptr;
        if (entry_f->attr_) {
            top_level = entry_f->attr_->get_or_null<bool>(
                    function_attrs::top_level);
        }
        bool should_run = true;
        // if the func is explicitly marked top-level/not-top-level, follow that
        // instruction
        if (top_level) {
            should_run = *top_level;
        } else {
            // else, if the func is main entry
            should_run = (entry_f == main_entry);
        }
        if (!should_run) { continue; }
        // find all calls to func decl, and sync the inplace_hint with decl and
        // definition
        func_finder_t finder;
        finder.dispatch(entry_f);
        for (auto &funct : finder.funcs_) {
            if (!funct->body_.defined()) {
                // if the func is decl
                auto func_def = f->get_func(funct->name_);
                if (func_def && func_def->attr_) {
                    if (auto hint
                            = func_def->attr_->get_or_null<inplace_hint_t>(
                                    function_attrs::inplace_hint)) {
                        funct->attr()[function_attrs::inplace_hint] = *hint;
                    }
                }
            }
        }

        if (!entry_f->attr_
                || !(entry_f->attr_->has_key(function_attrs::inplace_hint))) {
            // pairs of {out arg, in arg} that out arg can inplace in arg
            std::vector<std::pair<expr_c, expr_c>> inplace_map;
            schedule_func_args(entry_f, inplace_map);
            std::vector<std::pair<size_t, size_t>> inplace_pairs;
            if (!inplace_map.empty()) {
                for (const auto &out_in : inplace_map) {
                    int out_idx = -1;
                    int in_idx = -1;
                    for (int i = 0; i < int(entry_f->params_.size()); ++i) {
                        if (out_in.first.ptr_same(entry_f->params_[i])) {
                            out_idx = i;
                        }
                        if (out_in.second.ptr_same(entry_f->params_[i])) {
                            in_idx = i;
                        }
                    }
                    if (out_idx >= 0 && in_idx >= 0) {
                        inplace_pairs.emplace_back(
                                std::make_pair(in_idx, out_idx));
                    }
                }
                // entry_f->attr()[function_attrs::inplace_hint] =
                // inplace_pairs;
            }
        }

        buffer_scheduler_t scheduler {ctx_, true, true};
        auto new_func = scheduler(entry_f);
        // if no changes, continue
        if (new_func == entry_f) { continue; }
        finder.funcs_.clear();
        finder.dispatch(entry_f);
        // sync back alias group info to callee functions
        for (auto &funct : finder.funcs_) {
            auto func_def = f->get_func(funct->name_);
            if (func_def) {
                filter_and_sync_inplace_hint(funct.get(), func_def);
                if (func_def != funct) {
                    for (size_t arg_id = 0; arg_id < funct->params_.size();
                            arg_id++) {
                        auto &arg_in_decl = funct->params_[arg_id];
                        auto &arg_in_def = func_def->params_.at(arg_id);
                        if (auto alias_id
                                = alias_info::get_alias_info(*arg_in_decl)) {
                            // sync the alias info
                            arg_in_def->attr()[attr_keys::pointer_alias]
                                    = arg_in_decl->attr_->get_any(
                                            attr_keys::pointer_alias);
                        }
                    }
                }
            }
        }
        // drop the new_func instead of replacing it with entry_f. Because
        // parallel_merge pass needs to run before buffer scheduling
    }

    return ret;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
