/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"

#include "interface/logger.hpp"

/*
 * 1. Override the default verbosity level
 * 2. Log messages with different verbosity mode
 * 3. Expect only the unfiltered levels are actually logged
 */
TEST(logger_test, output_stdout) {
    if (llga::impl::Logger::disabled_) {
        GTEST_SKIP() << "Verbose mode disabled during compilation";
    }

    if (std::getenv("DNNL_GRAPH_VERBOSE_OUTPUT")) {
        GTEST_SKIP()
                << "Verbose output must not be redirected during this test";
    }

    if (std::getenv("DNNL_GRAPH_VERBOSE")) {
        GTEST_SKIP() << "Verbosity level must not be modified during this test";
    }

    testing::internal::CaptureStdout();

    llga::impl::Logger::set_default_log_level(dnnl_graph_log_level_error);

    DNNL_GRAPH_LOG(dnnl_graph_log_level_error) << "msg1";
    DNNL_GRAPH_LOG(dnnl_graph_log_level_debug) << "msg2";
    DNNL_GRAPH_LOG(dnnl_graph_log_level_info) << "msg3";
    DNNL_GRAPH_LOG_ERROR() << "msg4";
    DNNL_GRAPH_LOG_DEBUG() << "msg5";
    DNNL_GRAPH_LOG_INFO() << "msg6";

    std::string actual = testing::internal::GetCapturedStdout();

    ASSERT_NE(actual.find("msg1"), std::string::npos) << actual;
    ASSERT_EQ(actual.find("msg2"), std::string::npos) << actual;
    ASSERT_EQ(actual.find("msg3"), std::string::npos) << actual;
    ASSERT_NE(actual.find("msg4"), std::string::npos) << actual;
    ASSERT_EQ(actual.find("msg5"), std::string::npos) << actual;
    ASSERT_EQ(actual.find("msg6"), std::string::npos) << actual;
}
