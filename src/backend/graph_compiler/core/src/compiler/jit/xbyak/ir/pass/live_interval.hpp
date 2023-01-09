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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_PASS_LIVE_INTERVAL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_JIT_XBYAK_IR_PASS_LIVE_INTERVAL_HPP

#include <compiler/ir/function_pass.hpp>

namespace sc {
namespace sc_xbyak {

/* *
 * Calculate the liveness of each ir expr, the live range is determined by fist
 * and last use of an expr at stmt ir index.
 * */
class live_interval_t : public function_pass_t {
public:
    live_interval_t() = default;
    func_c operator()(func_c v) override;

private:
};

} // namespace sc_xbyak
} // namespace sc

#endif
