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

#include <algorithm>

#include "compiler_graph.hpp"
#include "compiler_partition_impl.hpp"
#include "patterns/concat_pattern.hpp"
#include "patterns/conv_pattern.hpp"
#include "patterns/mha_pattern.hpp"
#include "patterns/misc_pattern.hpp"
#include "patterns/mlp_pattern.hpp"
#include "target_machine.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

size_t compiler_backend_t::get_mem_size(const logical_tensor_t &lt) const {
    assert(lt.layout_type == layout_type::strided);
    // for 0-d tensor (scalar), mem_size is by default 1
    size_t mem_size = 1;
    for (int32_t i = 0; i < lt.ndims; ++i) {
        mem_size *= lt.dims[i];
    }
    return mem_size * logical_tensor_wrapper_t(lt).data_type_size();
}

bool compiler_backend_t::register_passes() {
    REQUIRE_AVX512_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(fp32_mha_pattern, pass_registry_);
    REQUIRE_AMX_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(fp32_mlp_pattern, pass_registry_);
    REQUIRE_AMX_END
    COMPILER_BACKEND_REGISTER_PASSES_CALL(
            fp32_conv_training_pattern, pass_registry_);
    REQUIRE_AMX_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(
            fp32_conv_inference_pattern, pass_registry_);
    REQUIRE_AMX_END
    REQUIRE_BF16_AMXBF16_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(bf16_mha_pattern, pass_registry_);
    REQUIRE_AMXBF16_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(bf16_mlp_pattern, pass_registry_);
    COMPILER_BACKEND_REGISTER_PASSES_CALL(
            bf16_conv_training_pattern, pass_registry_);
    COMPILER_BACKEND_REGISTER_PASSES_CALL(
            bf16_conv_inference_pattern, pass_registry_);
    REQUIRE_AMXBF16_END
    REQUIRE_BF16_AMXBF16_END
    REQUIRE_VNNI_AMXINT8_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(int8_mha_pattern, pass_registry_);
    COMPILER_BACKEND_REGISTER_PASSES_CALL(int8_mlp_pattern, pass_registry_);
    REQUIRE_AMX_BEGIN
    COMPILER_BACKEND_REGISTER_PASSES_CALL(
            int8_conv_inference_pattern, pass_registry_);
    REQUIRE_AMX_END
    REQUIRE_VNNI_AMXINT8_END
    COMPILER_BACKEND_REGISTER_PASSES_CALL(misc_pattern, pass_registry_);
    COMPILER_BACKEND_REGISTER_PASSES_CALL(concat_patterns, pass_registry_);
    REQUIRE_AVX512_END
    pass_registry_.sort_passes();
    return true;
}

status_t compiler_backend_t::get_partitions(
        graph_t &agraph, partition_policy_t policy) {
    // this environment variable is similar to DISABLE_DNNL_BACKEND
    // only for internal testing purpose
    const bool disable_compiler_bkd
            = graph::utils::getenv_int_internal("DISABLE_COMPILER_BACKEND", 0)
            > 0;
    if (disable_compiler_bkd) return status::success;
    graph::pass::pass_manager_t pm(get_pass_registry());
    pm.run_passes(agraph, "", policy);
    return status::success;
}

} // namespace compiler_impl

void register_compiler_backend() {
    backend_registry_t::get_singleton().register_backend(
            &compiler_impl::compiler_backend_t::get_singleton());
}

} // namespace graph
} // namespace impl
} // namespace dnnl
