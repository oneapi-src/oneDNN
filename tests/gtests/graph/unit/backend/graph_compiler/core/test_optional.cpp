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

#include <memory>
#include <string>
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <unordered_map>
#include <util/optional.hpp>
#include <util/optional_find.hpp>

using namespace dnnl::impl::graph::gc;

namespace optional_test {
struct AAA {
    virtual ~AAA() = default;
};

struct BBB : public AAA {
    static int cnt;
    static int dtor_cnt;
    static int move_cnt;
    int val;
    BBB(const BBB &v) : val(v.val) { cnt++; }
    BBB(BBB &&v) : val(v.val) {
        move_cnt++;
        cnt++;
    }
    BBB(int val) : val(val) { cnt++; }
    ~BBB() { dtor_cnt++; }
};

int BBB::cnt = 0;
int BBB::dtor_cnt = 0;
int BBB::move_cnt = 0;
} // namespace optional_test

using namespace optional_test;

TEST(GCCore_CPU_optional_cpp, TestOptional) {
    // check some_opt and map
    {
        auto v = some_opt(std::unique_ptr<BBB>(new BBB {10}))
                         .map([](const std::unique_ptr<BBB> &v) {
                             return v->val;
                         })
                         .get_or_else(-1);
        ASSERT_EQ(v, 10);
        ASSERT_EQ(BBB::cnt, 1);
        ASSERT_EQ(BBB::dtor_cnt, 1);
    }

    // check map fall through and nullptr
    {
        auto v = some_opt(std::unique_ptr<BBB>())
                         .map([](const std::unique_ptr<BBB> &v) {
                             return v->val;
                         })
                         .get_or_else(-1);
        ASSERT_EQ(v, -1);
    }
    static_assert(sizeof(optional<std::unique_ptr<BBB>>)
                    == sizeof(std::unique_ptr<BBB>),
            "Sizeof optional check failed");

    // check std::move(optional) and std::move(unique_ptr)
    {
        optional<std::unique_ptr<AAA>> v {std::unique_ptr<BBB>(new BBB {11})};
        // check move construct
        optional<std::unique_ptr<AAA>> v2 = std::move(v);
        ASSERT_EQ(BBB::cnt, 2);
        ASSERT_EQ(BBB::dtor_cnt, 1);
        // check moved optional
        ASSERT_FALSE(v.has_value());
        // check get() throw
        EXPECT_SC_ERROR({ v.get(); }, "Bad optional");

        auto ptr = std::move(v2.get());
        ASSERT_EQ(BBB::cnt, 2);
        ASSERT_EQ(BBB::dtor_cnt, 1);

        optional<std::unique_ptr<AAA>> v3 = std::move(ptr);
        ASSERT_EQ(BBB::cnt, 2);
        ASSERT_EQ(BBB::dtor_cnt, 1);

        // check move assign
        v3 = std::move(v2);
        ASSERT_EQ(BBB::dtor_cnt, 2);
        ASSERT_FALSE(v3.has_value());

        v3 = optional<std::unique_ptr<AAA>> {
                std::unique_ptr<BBB>(new BBB {11})};
    }
    ASSERT_EQ(BBB::dtor_cnt, 3);

    //////////////////// check copyable
    {
        auto opt = some_opt(BBB {123});
        ASSERT_EQ(BBB::move_cnt, 1);
        ASSERT_EQ(BBB::cnt, 5);
        ASSERT_EQ(BBB::dtor_cnt, 4);
        optional<BBB> opt2 {opt};
        ASSERT_TRUE(opt.has_value());
        ASSERT_EQ(BBB::cnt, 6);
        ASSERT_EQ(BBB::move_cnt, 1);
        ASSERT_EQ(BBB::dtor_cnt, 4);

        opt2 = std::move(opt);
        ASSERT_FALSE(opt.has_value());
        ASSERT_EQ(BBB::cnt, 7);
        ASSERT_EQ(BBB::move_cnt, 2);
        // old opt2 and opt destroyed
        ASSERT_EQ(BBB::dtor_cnt, 6);
        ASSERT_EQ(opt2.get().val, 123);

        optional<BBB> opt3;
        ASSERT_FALSE(opt3.has_value());
        opt3 = std::move(opt2);
        ASSERT_EQ(BBB::cnt, 8);
        ASSERT_EQ(BBB::move_cnt, 3);
        // opt2 destroyed
        ASSERT_EQ(BBB::dtor_cnt, 7);
        ASSERT_EQ(opt3.get().val, 123);
    }
    ASSERT_EQ(BBB::dtor_cnt, 8);

    //////////////////// check flat_map
    {
        auto check_func = [](int input) {
            return some_opt(input).flat_map([](int v) {
                if (v == 2) {
                    return some_opt(v + 1);
                } else {
                    return optional<int> {};
                }
            });
        };
        ASSERT_FALSE(check_func(1).has_value());
        auto opt = check_func(2);
        ASSERT_TRUE(opt.has_value());
        ASSERT_TRUE(opt.get() == 3);
    }

    // check or_else
    {
        int outv1 = 1;
        auto check_func = [&outv1](int *input) {
            return some_opt(input).or_else(
                    [&outv1]() { return some_opt(&outv1); });
        };
        auto opt = check_func(nullptr);
        ASSERT_TRUE(opt.has_value());
        ASSERT_TRUE(opt.get() == &outv1);

        opt = check_func(&outv1);
        ASSERT_TRUE(opt.has_value());
        ASSERT_TRUE(opt.get() == &outv1);
    }

    // check filter
    {
        auto check_func = [](optional<int> v) {
            return std::move(v).filter([](int v) { return v == 2; });
        };
        ASSERT_FALSE(check_func(1).has_value());
        ASSERT_FALSE(check_func(none_opt {}).has_value());
        auto opt = check_func(2);
        ASSERT_TRUE(opt.has_value());
        ASSERT_TRUE(opt.get() == 2);
    }

    // check functional get_or_else
    {
        int v = optional<int> {}.get_or_else([]() { return 123; });
        ASSERT_TRUE(v == 123);
    }

    {
        expr e = expr(1) + (2 * builder::make_var(datatypes::s32, "aaa"));

        auto the_var_type
                = e.cast<add>()
                          .flat_map([](add v) { return v->r_.cast<mul>(); })
                          .map([](mul v) { return v->r_.as<var>(); })
                          .map([](var v) { return v->dtype_; })
                          .get_or_else(datatypes::undef);
        ASSERT_EQ(the_var_type, datatypes::s32);

        ASSERT_FALSE(some_opt(expr {}).has_value());
    }

    // map find
    {
        std::unordered_map<int, int> v = {{1, 2}, {3, 4}};
        auto opt1 = utils::find_map_value(v, 1);
        ASSERT_TRUE(*opt1.get() == 2);
        opt1 = utils::find_map_value(v, 2);
        ASSERT_FALSE(opt1.has_value());

        auto opt2 = utils::find_map_pair(v, 1);
        ASSERT_TRUE(opt2.get()->second == 2);
        opt2 = utils::find_map_pair(v, 2);
        ASSERT_FALSE(opt2.has_value());
    }
}
