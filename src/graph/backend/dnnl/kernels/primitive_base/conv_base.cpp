/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "graph/backend/dnnl/kernels/primitive_base/conv_base.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void conv_base_t::prepare_args_set(const execution_args_set_t *res,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const scratchpad_t &scratchpad) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        mem_idx.first.set_data_handle(inputs[mem_idx.second].get_data_handle());
    }
    for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
        mem_idx.first.set_data_handle(
                outputs[mem_idx.second].get_data_handle());
    }

    grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
            scratchpad.get_buffer());

    for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
        mem_offkey.first.set_data_handle(var_grantor.get(mem_offkey.second));
    }
}

status_t conv_base_t::execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    temporary_scratchpad_t scratchpad(
            memory_planner_.total_internal_temporary_size(), p_engine_,
            *g_alloc_);
    assertm(scratchpad.size()
                    >= memory_planner_.total_internal_temporary_size(),
            "no enough scratchpad memory");
    prepare_args_set(res, inputs, outputs, scratchpad);

    constant_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_cache_t::cached_t> c_promise;
        constant_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(), p_engine_,
                    g_alloc_);
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                subgraph_->execs_[i]->execute(
                        p_stream, res->get_exec_args()[i]);
            }

            c_promise.set_value(c_buffer);
        }
    }

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
    }

    return status::success;
}

#ifdef DNNL_WITH_SYCL
status_t conv_base_t::sycl_execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    auto deps = sycl_deps;
    ::sycl::event returned_event;
    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    temporary_scratchpad_t scratchpad(
            memory_planner_.total_internal_temporary_size(), p_engine_,
            *g_alloc_);
    assertm(scratchpad.size()
                    >= memory_planner_.total_internal_temporary_size(),
            "no enough scratchpad memory");
    prepare_args_set(res, inputs, outputs, scratchpad);

    constant_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_cache_t::cached_t> c_promise;
        constant_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(), p_engine_,
                    g_alloc_);
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                returned_event = subgraph_->execs_[i]->execute_sycl(
                        p_stream, res->get_exec_args()[i], deps);
                deps = {returned_event};
            }

            c_promise.set_value(c_buffer);
        }
    }

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        returned_event = subgraph_->execs_[i]->execute_sycl(
                p_stream, res->get_exec_args()[i], deps);
        deps = {returned_event};
    }

    scratchpad.set_deps(returned_event);
    if (sycl_event) *sycl_event = returned_event;

    return status::success;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
status_t conv_base_t::ocl_execute_impl(const stream_t *g_stream,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs,
        const std::vector<cl_event> &ocl_deps, cl_event *ocl_event) {

    auto deps = ocl_deps;
    cl_event returned_event {};
    dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    temporary_scratchpad_t scratchpad(
            memory_planner_.total_internal_temporary_size(), p_engine_,
            *g_alloc_);
    assertm(scratchpad.size()
                    >= memory_planner_.total_internal_temporary_size(),
            "no enough scratchpad memory");
    prepare_args_set(res, inputs, outputs, scratchpad);

    constant_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_cache_t::cached_t> c_promise;
        constant_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(), p_engine_,
                    g_alloc_);
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                returned_event = subgraph_->execs_[i]->execute_ocl(
                        p_stream, res->get_exec_args()[i], deps);
                deps = {returned_event};
            }

            c_promise.set_value(c_buffer);
        }
    }

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        returned_event = subgraph_->execs_[i]->execute_ocl(
                p_stream, res->get_exec_args()[i], deps);
        deps = {returned_event};
    }

    scratchpad.set_deps(returned_event);
    if (ocl_event) *ocl_event = returned_event;

    return status::success;
}
#endif
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
