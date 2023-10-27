/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_UTILS_HPP

#include "tensor_inplace_info.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace special_ticks {
// the tensor is never accessed
static constexpr int64_t TICK_NOT_EXIST = -2;
// the tensor has complicated access pattern: have you assigned a tensor to a
// pointer?
static constexpr int64_t COMPLICATED_ACCESS = -1;
// the tensor was declared in for loop, and its lifetime is complicated. But can
// be merged with other buffers with hints
static constexpr int64_t HINT_IN_LOOP = -3;
} // namespace special_ticks

// the struct to track the call site which may have inplace reuse
struct call_site_info_t {
    func_c func_;
    // the tensors passed to the function on the caller side. It has length of
    // func->args_. If an arg is not tensor related, the element in this array
    // will be empty.
    std::vector<expr_c> tensors_passed_;
};

// the tensor is not thread local
static constexpr uint64_t NOT_THREAD_LOCAL = 0;
struct tensor_tick_info_t {
    // first read/write tick, will not be reset by complex scopes, useful for
    // sorting the tensors
    int64_t real_first_access_ = special_ticks::TICK_NOT_EXIST;
    int64_t first_access_
            = special_ticks::TICK_NOT_EXIST; // first read/write tick
    int64_t last_read_ = special_ticks::TICK_NOT_EXIST; // last read tick
    std::set<int64_t> writes_; // all write ticks
    int64_t create_ = special_ticks::TICK_NOT_EXIST; // tensor creation tick
    int64_t delete_ = special_ticks::TICK_NOT_EXIST; // the tick that the tensor
            // scope is done
    bool is_arg_ = false; // if is the tensor defined in function args
    // if the tensor is already scheduled, infered from the define stmt's
    // init_. If the tensor is arg tensor, already_scheduled_base_ is the tensor
    // itself
    expr_c already_scheduled_base_;
    uint64_t scope_ = NOT_THREAD_LOCAL; // parallel scope id
    bool has_hint_ = false; // if the tensor has hint tick info
    // the tensors that the current tensor can inplace reuse buffer with. Not
    // all of the tensors in this set is valid. Only if last_access_tick of the
    // tensor in the set == the first_access_tick of the current tensor can
    // the tensor be reused.
    std::vector<std::pair<expr_c, inplace_kind>> inplace_reuse_;
    // the call site of the inplace function call. may be null. It is used to
    // back-propagate the inplace result to the function to be called, to inform
    // it that some of the args has pointer alias
    std::shared_ptr<call_site_info_t> inplace_call_site_;
    // Only valid when this tensor is an argument to list_brgemm calls. The set
    // is used to record the A and B data tensors that the address list tensor
    // of list_brgemm points to.
    std::unique_ptr<std::unordered_set<expr_c>> list_brgemm_tensors_;

    bool is_already_scheduled() const {
        return already_scheduled_base_.defined();
    }
    int64_t get_last_access() const {
        int64_t last_access = last_read_;
        if (!writes_.empty()) {
            last_access = std::max(last_access, *writes_.rbegin());
        }
        if (last_access == special_ticks::TICK_NOT_EXIST) {
            last_access = first_access_;
        }
        return last_access;
    }
};

void annotate_ticks(const func_c &f,
        std::unordered_map<expr_c, tensor_tick_info_t> &ticks,
        std::vector<expr_c> &defined);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
