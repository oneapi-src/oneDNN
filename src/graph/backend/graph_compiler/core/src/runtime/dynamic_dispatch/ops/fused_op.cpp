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
#include <stdint.h>
#include <string.h>
#include "impl_type.hpp"
#include "util.hpp"
#include <runtime/dynamic_dispatch/op_dispatch_tables.hpp>
#include <util/null_check.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
extern "C" void *sc_global_aligned_alloc(size_t sz, size_t align);
extern "C" void sc_global_aligned_free(void *ptr, size_t align);
extern "C" void query_combined_fused_op(void *table, uint64_t **combined_keys,
        int *combined_algs, int *each_op_num_key, int op_num, void *kernel) {
    // first combine alg into dispatch keys
    int total_key_num = 0;
    for (int i = 0; i < op_num; i++) {
        total_key_num += each_op_num_key[i];
    }
    size_t sz = sizeof(uint64_t) * total_key_num;
    runtime::dispatch_key *final_query_keys
            = static_cast<runtime::dispatch_key *>(
                    sc_global_aligned_alloc(sz, 64));
    SC_ABORT_IF_NULL(final_query_keys);
    runtime::dispatch_key **combined_dispatch_keys
            = reinterpret_cast<runtime::dispatch_key **>(combined_keys);
    // link all impl alg of reorder, if all of impl of reorder is no_padding,
    // then use no_padding; else reset all of impl to normal.
    int linked_reorder_impl = impl_kind_t::no_padding;
    bool stop = false;
    for (int i = 0; !stop && i < op_num; i++) {
        for (int k = 0; k < each_op_num_key[i]; k++) {
            // currently use number == 2 to judge if it is a reorder.
            if (!combined_algs
                    || (each_op_num_key[i] == 2
                            && combined_algs[i] == impl_kind_t::normal)) {
                linked_reorder_impl = impl_kind_t::normal;
                stop = true;
                break;
            }
        }
    }
    int offset = 0;
    for (int i = 0; i < op_num; i++) {
        for (int k = 0; k < each_op_num_key[i]; k++) {
            final_query_keys[offset + k] = *combined_dispatch_keys[offset + k];
            if (each_op_num_key[i] == 2) {
                final_query_keys[offset + k].reset_blocks_and_impl();
                final_query_keys[offset + k].set_impl_alg(linked_reorder_impl);
            } else {
                final_query_keys[offset + k].set_impl_alg(combined_algs[i]);
            }
        }
        offset += each_op_num_key[i];
    }
    // query kernel, need determine the impl alg first.
    runtime::op_dispatch_tables_t *op_table
            = reinterpret_cast<runtime::op_dispatch_tables_t *>(table);
    auto &kernel_table = op_table->kernel_table_;
    if (kernel_table) {
        void *func = runtime::run_query_and_wait(
                op_table->kernel_dispatch_func_, kernel_table.get(),
                reinterpret_cast<uint64_t *>(final_query_keys), total_key_num);
        *reinterpret_cast<void **>(kernel) = func;
    }
    // reset blocks and impl
    for (int i = 0; i < total_key_num; i++) {
        combined_dispatch_keys[i]->reset_blocks_and_impl();
    }
    sc_global_aligned_free(final_query_keys, 64);
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
