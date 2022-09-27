/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_types.h"

TEST(ErrorAPI, ErrorResult2Str) {
    std::vector<std::pair<dnnl_graph_status_t, std::string>> mapper {
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_success, "success"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_out_of_memory, "out of memory"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_invalid_arguments, "invalid arguments"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_unimplemented, "unimplemented"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_iterator_ends, "iterator ends"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_runtime_error, "runtime error"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_not_required, "not required"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_invalid_graph, "invalid graph"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_invalid_graph_op, "invalid op"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_invalid_shape, "invalid shape"},
            std::pair<dnnl_graph_status_t, std::string> {
                    dnnl_graph_invalid_data_type, "invalid data type"},
            std::pair<dnnl_graph_status_t, std::string> {
                    (dnnl_graph_status_t)11, "unknown error"}};
    for (auto &item : mapper) {
        auto err = dnnl::graph::error(item.first, "test");
        ASSERT_EQ(err.result2str(item.first), item.second);
        ASSERT_EQ(std::string(err.what()), "test: " + item.second);
    }
}
