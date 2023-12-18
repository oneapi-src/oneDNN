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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_CONFIG_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_RUNTIME_DYNAMIC_DISPATCH_OPS_CONFIG_HPP
#include <util/def.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
struct target_machine_t;
}
/**
 * @brief Get the dynamic config single block from the plain dynamic dimension
 * for matmul
 *
 * @param in the dynamic dimension
 * @param is_batch default false, candidates are [16, 32, 64], if true,
 * candidates are [2, 4, 8, 16, 32, 64].
 * @return the selected block config
 */
extern "C" SC_API int get_matmul_dyn_cfg_single(int in, bool is_batch = false);

// The function calculate the config of managed matmul, it is used in both
// compiler and runtime.
void get_managed_matmul_config(const runtime::target_machine_t &tm,
        int &M_split_num, int &N_split_num, int &M_sub_block, int &N_sub_block,
        int &K_sub_block, int &im_loop_order, const int M, const int N,
        const int K, const int iim_block, const int iin_block,
        const int iik_block, const int sizeofdtypeA, const int sizeofdtypeC,
        bool is_int8, bool is_f32, bool is_dynamic, int64_t dispatch_avx = 0);

// The function calculate the block of dynamic conv
int get_dyn_conv_default_block(const bool is_1x1, const int dtype_size,
        const bool has_pad, const bool is_f32);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
