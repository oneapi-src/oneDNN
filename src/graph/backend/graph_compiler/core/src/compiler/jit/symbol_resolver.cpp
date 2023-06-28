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

#include "symbol_resolver.hpp"
#include <runtime/generic_val.hpp>
#include <runtime/microkernel/cpu/microkernel.hpp>
#ifdef SC_ENABLE_L0_BACKEND
#include <runtime/l0_runtime.hpp>
#endif
#include <math.h>
#include <runtime/barrier.hpp>
#include <runtime/config.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <runtime/dynamic_dispatch/op_func_decl.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/managed_thread_pool_exports.hpp>
#include <runtime/memorypool.hpp>
#include <runtime/parallel.hpp>
#include <runtime/runtime.hpp>
#include <runtime/thread_locals.hpp>
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

const std::unordered_map<std::string, void *> &get_runtime_function_map() {
    static std::unordered_map<std::string, void *> table = {
            {"dnnl_brgemm_init", (void *)dnnl_brgemm_init},
            {"dnnl_brgemm_update", (void *)dnnl_brgemm_update},
            {"dnnl_brgemm_init_update", (void *)dnnl_brgemm_init_update},
            {"dnnl_brgemm_init_list_update",
                    (void *)dnnl_brgemm_init_list_update},
            {"dnnl_brgemm_list_update", (void *)dnnl_brgemm_list_update},
            {"dnnl_brgemm_list_call", (void *)dnnl_brgemm_list_call},
            {"dnnl_brgemm_list_call_range",
                    (void *)dnnl_brgemm_list_call_range},
            {"dnnl_brgemm_list_call_postops",
                    (void *)dnnl_brgemm_list_call_postops},
            {"dnnl_brgemm_list_func", (void *)dnnl_brgemm_list_func},
            {"dnnl_brgemm_func", (void *)dnnl_brgemm_func},
            {"dnnl_brgemm_call", (void *)dnnl_brgemm_call},
            {"dnnl_brgemm_call_range", (void *)dnnl_brgemm_call_range},
            {"dnnl_brgemm_call_postops", (void *)dnnl_brgemm_call_postops},
            {"dnnl_brgemm_postops_data_init",
                    (void *)dnnl_brgemm_postops_data_init},
            {"print_float", (void *)print_float},
            {"print_index", (void *)print_index},
            {"print_int", (void *)print_int},
            {"print_str", (void *)print_str},
            {"sc_global_aligned_alloc", (void *)sc_global_aligned_alloc},
            {"sc_global_aligned_free", (void *)sc_global_aligned_free},
            {"sc_thread_aligned_malloc", (void *)sc_thread_aligned_malloc},
            {"sc_thread_aligned_free", (void *)sc_thread_aligned_free},
            {"sc_acquire_const_cache", (void *)sc_acquire_const_cache},
            {"sc_release_const_cache", (void *)sc_release_const_cache},
            {"sc_aligned_malloc", (void *)sc_aligned_malloc},
            {"sc_aligned_free", (void *)sc_aligned_free},
            {"sc_make_trace", (void *)sc_make_trace},
            {"sc_make_trace_kernel", (void *)sc_make_trace_kernel},
            {"sc_get_tls_amx_buffer", (void *)sc_get_tls_amx_buffer},
            {"sc_parallel_call_cpu_with_env",
                    (void *)runtime_config_t::get()
                            .thread_pool_table_->parallel_call},
            {"sc_is_in_parallel",
                    (void *)runtime_config_t::get()
                            .thread_pool_table_->is_in_parallel},
            {"sc_get_thread_id",
                    (void *)runtime_config_t::get()
                            .thread_pool_table_->get_thread_id},
            {"sc_parallel_call_managed",
                    (void *)runtime_config_t::get()
                            .thread_pool_table_->parallel_call_managed},
            {"sc_set_idle_func_managed", (void *)sc_set_idle_func_managed},
            {"sc_arrive_at_barrier", (void *)sc_arrive_at_barrier},
            {"sc_init_barrier", (void *)sc_init_barrier},
            // dynamic query function
            {"query_format_matmul_core_op",
                    (void *)query_format_matmul_core_op},
            {"query_format_managed_matmul_core_op",
                    (void *)query_format_managed_matmul_core_op},
            {"query_format_conv_fwd_core_op",
                    (void *)query_format_conv_fwd_core_op},
            {"query_format_unary_fusible_op",
                    (void *)query_format_unary_fusible_op},
            {"query_format_binary_fusible_op",
                    (void *)query_format_binary_fusible_op},
            {"query_format_reorder_op", (void *)query_format_reorder_op},
            {"query_format_padding_op", (void *)query_format_padding_op},
            {"query_format_reduce_op", (void *)query_format_reduce_op},
            {"query_format_tensor_view_op",
                    (void *)query_format_tensor_view_op},
            {"query_format_select_op", (void *)query_format_select_op},
            {"query_combined_fused_op", (void *)query_combined_fused_op},
            {"get_matmul_dyn_cfg_single", (void *)get_matmul_dyn_cfg_single},
    };
    return table;
}

void *default_external_symbol_resolve(const std::string &name) {
    auto &table = get_runtime_function_map();
    auto itr = table.find(name);
    if (itr == table.end()) { return nullptr; }
    return itr->second;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
