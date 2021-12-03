/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_INTRINSICS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_INTRINSICS_HPP

#include <ostream>
#include <string>
#include "sc_expr.hpp"
namespace sc {
class ir_visitor_t;
struct intrinsic_handler_t {
    std::string name_;
    virtual void on_initialize(intrin_call_node &node) = 0;
    intrinsic_handler_t(const std::string &name);
    virtual ~intrinsic_handler_t() = default;
};

// the indices of arguments of the brgemm intrinsices
namespace brgemm_args {
constexpr int A = 0;
constexpr int B = 1;
constexpr int C = 2;
constexpr int NUM = 3;
constexpr int M = 4;
constexpr int N = 5;
constexpr int K = 6;
constexpr int LDA = 7;
constexpr int LDB = 8;
constexpr int LDC = 9;
constexpr int STRIDE_A = 10;
constexpr int STRIDE_B = 11;
constexpr int LEN = 12;
constexpr int NUM_ARGS_CPU = STRIDE_B + 1;
constexpr int NUM_ARGS_LIST = LEN + 1;

struct cpu_t {
    // use init_update or update
    bool init_;
};
struct extra_args_t {
    bool is_cpu_;
    sc_data_type_t dtype_A_ = datatypes::undef; // element dtype of mat A
    sc_data_type_t dtype_B_ = datatypes::undef; // element dtype of mat B
    sc_data_type_t dtype_C_ = datatypes::undef; // element dtype of mat C
    union {
        cpu_t cpu_;
    };
    extra_args_t(const cpu_t &g, sc_data_type_t dtypeA,
            sc_data_type_t dtypeB = datatypes::undef,
            sc_data_type_t dtypeC = datatypes::undef)
        : is_cpu_(true)
        , dtype_A_(dtypeA)
        , dtype_B_(dtypeB == datatypes::undef ? dtypeA : dtypeB)
        , dtype_C_(dtypeC == datatypes::undef ? dtypeA : dtypeC)
        , cpu_(g) {}
};

extern sc_data_type_t arg_types[NUM_ARGS_CPU];
extern sc_data_type_t list_arg_types[NUM_ARGS_LIST];
} // namespace brgemm_args

intrinsic_handler_t &get_intrinsic_handler(intrin_type intrin);

} // namespace sc

#endif
