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

#include <compiler/ir/content_hash.hpp>
#include <compiler/ir/easy_build.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

template <typename T>
constant_c mk_const(T v) {
    return expr(v).static_as<constant_c>();
}

TEST(GCCore_CPU_content_hash_cpp, TestContentHashConst) {
    builder::ir_builder_t builder;
    content_hash_t<constant_c> hasher;
    content_hash_t<std::vector<constant_c>> vhasher;
    content_equals_t<std::vector<constant_c>> veq;
    EXPECT_EQ(hasher(mk_const(100UL)), hasher(mk_const(100UL)));
    EXPECT_EQ(hasher(mk_const(100.0f)), hasher(mk_const(100.0f)));
    EXPECT_EQ(hasher(mk_const(int(100))), hasher(mk_const(int(100))));

    EXPECT_NE(hasher(mk_const(int(100))), hasher(mk_const(100UL)));
    EXPECT_NE(hasher(mk_const(0.0f)), hasher(mk_const(0UL)));

    std::vector<constant_c> vec {mk_const(1), mk_const(1.f), mk_const(100UL)};
    std::vector<constant_c> vec2 {mk_const(1), mk_const(1.f), mk_const(100UL)};
    std::vector<constant_c> vec3 {mk_const(1), mk_const(1.f)};
    std::vector<constant_c> vec4 {mk_const(0), mk_const(1.f), mk_const(100UL)};

    EXPECT_EQ(vhasher(vec), vhasher(vec2));
    EXPECT_TRUE(veq(vec, vec2));

    EXPECT_NE(vhasher(vec), vhasher(vec3));
    EXPECT_FALSE(veq(vec, vec3));

    EXPECT_NE(vhasher(vec), vhasher(vec4));
    EXPECT_FALSE(veq(vec, vec4));
}
