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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_PREFETCH_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TRAIT_MAY_PREFETCH_HPP

#include <vector>
#include <compiler/ir/graph/tensor_slice.hpp>
#include <compiler/ir/graph/traits.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace op_traits {
struct may_prefetch_t : public virtual op_base_trait_t {
    /**
     * @brief Query the the indices of inputs which needs prefetching
     *
     * @param ctx the context
     * @param is_global query for global or local prefetch. @see
     * generate_prefetcher_and_set_idle
     * @param ins the vector of input tensor slices
     * @return std::vector<int> a vector of input tensor indices
     */
    virtual std::vector<int> query_prefetch(const context_ptr &ctx,
            bool is_global, const std::vector<tensor_slice> &ins)
            = 0;

    /**
     * @brief generate prefetcher code for the inputs selected by query_prefetch
     * into the current ir_builder. This function is for tensor slice.
     *
     * @param ctx the context
     * @param func_args the arguments for the prefetcher function. The first arg
     * is the pointer to the trigger value(as a tensor). The second arg is the
     * expected value. The prefetcher should exit when trigger value !=
     * expected.
     * @param ins the inputs, in the new prefetch IR function created by
     * generate_prefetcher_func. The tensors and vars are in func_args
     * @param indices the input indices selected by query_prefetch
     */
    virtual void generate_prefetcher_body_for_slice(const context_ptr &ctx,
            const std::vector<expr> &func_args,
            const std::vector<tensor_slice> &ins,
            const std::vector<int> &indices);

    /**
     * @brief generate prefetcher code for the inputs selected by query_prefetch
     * into the current ir_builder. This function is for whole tensor.
     *
     * @param ctx the context
     * @param func_args the arguments for the prefetcher function. The first
     * arg is the pointer to the trigger value(as a tensor). The second arg is
     * the expected value. The prefetcher should exit when trigger value ==
     * expected. The third arg is the tid.
     * @param ins the inputs, in the new prefetch IR function created by
     * generate_prefetcher_func. The tensors and vars are in func_args
     * @param indices the input indices selected by query_prefetch
     */
    virtual void generate_prefetcher_body_for_tensor(const context_ptr &ctx,
            const std::vector<expr> &func_args, const std::vector<expr> &ins,
            const std::vector<int> &indices)
            = 0;

    /**
     * @brief generate prefetcher function for the inputs selected by
     * query_prefetch. The default inplementation calles
     * generate_prefetcher_body. Op implementations usually don't need to
     * override this function.
     *
     * @param ctx the context
     * @param is_global whether the user wants a global or local prefetcher. A
     * global prefetcher will be registered in the main entry function via
     * managed thread pool API and the local prefetcher is passed to
     * sc_arrive_at_barrier. Local prefetchers are usually used in fused op. The
     * prefetcher interfaces are slightly different.
     * @param ins the slice for all inputs of the Op, in the Op's computation
     * code. Should be the same as the parameter passed to query_prefetch
     * @param indices the input indices selected by query_prefetch
     * @param out_set_idle_code outputs the statements for calling
     * set_thread_idle_func()
     * @return the generated prefetcher function
     */
    virtual func_t generate_prefetcher_and_set_idle(const context_ptr &ctx,
            bool is_global, const std::vector<tensor_slice> &ins,
            const std::vector<int> &indices,
            std::vector<stmt> &out_set_idle_code);
};

} // namespace op_traits
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
