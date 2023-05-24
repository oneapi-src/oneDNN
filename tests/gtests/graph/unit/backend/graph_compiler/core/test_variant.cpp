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

#include <string>
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <util/variant.hpp>

using namespace dnnl::impl::graph::gc;

struct vtest_AAA {
    static int num_dtor;
    ~vtest_AAA() { num_dtor++; }
};
int vtest_AAA::num_dtor = 0;

TEST(GCCore_CPU_variant_cpp, TestVarient) {
    using the_var = variant<std::string, int, float, vtest_AAA>;
    the_var a;
    vtest_AAA mystruct;
    a = vtest_AAA {};
    std::string str = "123";
    a = std::move(str);
    // old value dtor called
    ASSERT_EQ(vtest_AAA::num_dtor, 2);
    // moved
    ASSERT_TRUE(str.empty());
    ASSERT_EQ(a.get<std::string>(), "123");

    // test copy assign
    str = "321";
    a = str;
    ASSERT_FALSE(str.empty());
    ASSERT_EQ(a.get<std::string>(), "321");
    a = str;

    // test POD
    ASSERT_FALSE(a.isa<int>());
    a = 123.0f;
    ASSERT_TRUE(a.isa<float>());
    ASSERT_EQ(a.get<float>(), 123.0f);

    // test type check
    EXPECT_SC_ERROR({ a.get<int>(); }, "Bad variant cast");

    // test move ctor
    {
        the_var b {std::move(str)};
        ASSERT_TRUE(str.empty());
        ASSERT_EQ(b.get<std::string>(), "321");
    }

    str = "321";
    // test copy ctor
    {
        the_var b {str};
        ASSERT_FALSE(str.empty());
        ASSERT_EQ(b.get<std::string>(), "321");
        // test dtor
        b = mystruct;
    }
    ASSERT_EQ(vtest_AAA::num_dtor, 3);

    {
        // variant to variant copy
        the_var b {std::string("aaa")};
        a = b;
        ASSERT_EQ(a.get<std::string>(), "aaa");
        ASSERT_EQ(b.get<std::string>(), "aaa");

        ASSERT_TRUE(a.defined());
        a.clear();
        // check calling clear() for multiple times
        a.clear();
        ASSERT_FALSE(a.defined());

        // variant to variant move
        a = std::move(b);
        ASSERT_FALSE(b.defined());
        ASSERT_EQ(a.get<std::string>(), "aaa");
    }

    // the following line should not compile
    // a.get<long>();
}

struct vtest_Base {
    int val = 0;
};

struct vtest_BBB : public vtest_Base {
    vtest_BBB() { val = 1; }
};

struct vtest_CCC : public vtest_Base {
    vtest_CCC() { val = 2; }
};

struct vtest_DDD : public vtest_BBB {
    vtest_DDD() { val = 3; }
};

TEST(GCCore_CPU_variant_cpp, TestVarientAs) {
    using the_var = variant<vtest_Base, vtest_BBB, vtest_CCC, vtest_DDD, int>;
    {
        the_var a {vtest_DDD {}};
        auto &base = a.as<vtest_Base>();
        ASSERT_EQ(base.val, 3);
        auto &ddd = a.as<vtest_DDD>();
        ASSERT_EQ(ddd.val, 3);
        auto &bbb = a.as<vtest_BBB>();
        ASSERT_EQ(bbb.val, 3);
        auto cptr = a.as_or_null<vtest_CCC>();
        ASSERT_EQ(cptr, nullptr);
        auto iptr = a.as_or_null<int>();
        ASSERT_EQ(iptr, nullptr);
    }

    {
        the_var a {vtest_BBB {}};
        auto &base = a.as<vtest_Base>();
        ASSERT_EQ(base.val, 1);
        auto dptr = a.as_or_null<vtest_DDD>();
        ASSERT_EQ(dptr, nullptr);
        auto &bbb = a.as<vtest_BBB>();
        ASSERT_EQ(bbb.val, 1);
        auto cptr = a.as_or_null<vtest_CCC>();
        ASSERT_EQ(cptr, nullptr);
        auto iptr = a.as_or_null<int>();
        ASSERT_EQ(iptr, nullptr);
    }

    {
        the_var a {10};
        auto dptr = a.as_or_null<vtest_DDD>();
        ASSERT_EQ(dptr, nullptr);
        auto iptr = a.as_or_null<int>();
        ASSERT_EQ(*iptr, 10);
    }

    using the_var2 = variant<std::shared_ptr<vtest_CCC>, int>;
    the_var2 bbb = {std::make_shared<vtest_CCC>()};
    EXPECT_SC_ERROR({ bbb.cast<float>(); }, "Bad variant cast");
    auto ptr = bbb.cast<std::shared_ptr<vtest_Base>>();
    ASSERT_EQ(ptr->val, 2);

    bbb = 1;
    EXPECT_SC_ERROR(
            { bbb.cast<std::shared_ptr<vtest_Base>>(); }, "Bad variant cast");
    auto fv = bbb.cast<float>();
    ASSERT_EQ(fv, 1.0f);
}
