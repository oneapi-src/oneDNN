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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_INFO_MACROS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_INFO_MACROS_HPP

#ifndef NDEBUG
#define SC_DECL_PASS_DEPDENCYINFO() \
    void get_dependency_info(tir_pass_dependency_t &out) const override;
#else
#define SC_DECL_PASS_DEPDENCYINFO()
#endif

#define SC_DECL_PASS_INFO_FUNC() \
    const char *get_name() const override; \
    SC_DECL_PASS_DEPDENCYINFO();

#endif
