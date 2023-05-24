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

#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/sc_data_format.hpp>

using namespace dnnl::impl::graph::gc;

static std::vector<int> get_format_as_vec(sc_data_format_kind_t k) {
    std::vector<int> ret;
    ret.reserve(k.ndims());
    for (int i = 0; i < k.ndims(); i++) {
        ret.push_back(k.get(i));
    }
    return ret;
}

TEST(GCCore_CPU_data_format_cpp, TestFormatKind) {
    auto fmt = format_kinds::ABCD;
    EXPECT_EQ(fmt.ndims(), 4);
    EXPECT_EQ(fmt.norig_dims(), 4);
    EXPECT_EQ(get_format_as_vec(fmt), (std::vector<int> {0, 1, 2, 3}));

    fmt = format_kinds::ABCDba;
    EXPECT_EQ(fmt.ndims(), 6);
    EXPECT_EQ(fmt.norig_dims(), 4);
    EXPECT_EQ(get_format_as_vec(fmt), (std::vector<int> {0, 1, 2, 3, 1, 0}));

    EXPECT_EQ(sc_data_format_t::get_plain_by_dims(4),
            sc_data_format_t(format_kinds::ABCD));
}

static std::string fmt2str(const sc_data_format_t &v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

TEST(GCCore_CPU_data_format_cpp, TestFormatPrint) {
    auto fmt = sc_data_format_t::NCHWc(16);
    EXPECT_EQ(fmt2str(fmt), "ABCD16b");
}

TEST(GCCore_CPU_data_format_cpp, TestFormatGetShape) {
    // reorder, no blocking
    sc_dims plain, blocking;
    plain = {1, 2, 3, 4};
    blocking = {2, 1, 4, 3};
    EXPECT_EQ(sc_data_format_t::get_blocking_shapes(plain,
                      sc_data_format_t(sc_data_format_kind_t(1, 0, 3, 2))),
            blocking);
    EXPECT_EQ(sc_data_format_t::get_padded_plain_shapes(blocking,
                      sc_data_format_t(sc_data_format_kind_t(1, 0, 3, 2))),
            plain);
    // simple blocking
    EXPECT_EQ(sc_data_format_t::get_blocking_shapes(
                      {16, 31, 64, 128}, sc_data_format_t::NCHWc(16)),
            (sc_dims {16, 2, 64, 128, 16}));
    EXPECT_EQ(sc_data_format_t::get_padded_plain_shapes(
                      {16, 2, 64, 128, 16}, sc_data_format_t::NCHWc(16)),
            (sc_dims {16, 32, 64, 128}));

    // nested blocking
    EXPECT_EQ(sc_data_format_t::get_blocking_shapes(
                      {63, 256}, sc_data_format_t::NKkn2k(32, 16)),
            (sc_dims {16, 2, 16, 16, 2}));
    EXPECT_EQ(sc_data_format_t::get_padded_plain_shapes(
                      {16, 2, 16, 16, 2}, sc_data_format_t::NKkn2k(32, 16)),
            (sc_dims {64, 256}));
    EXPECT_EQ(sc_data_format_t::get_blocking_shapes(
                      {63, 256}, sc_data_format_t::NKkn2k(-1, 16)),
            (sc_dims {16, -1, -1, 16, 2}));
    EXPECT_EQ(sc_data_format_t::get_padded_plain_shapes(
                      {16, 2, 16, 16, 2}, sc_data_format_t::NKkn2k(-1, 16)),
            (sc_dims(2, -1)));
}
