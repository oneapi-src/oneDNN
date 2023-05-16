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

#include <functional>
#include <iostream>
#include <vector>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/passlet/volatility_analysis.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <gtest/gtest.h>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

namespace volatility_analysis {

using namespace passlet;
struct viewer_t : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    using ssa_viewer_t::view;

    volatility_analysis_t v_ana_;
    struct result_t {
        int r2 = 0;
        volatility_result_t r1;
    };
    viewer_t(bool is_loop,
            std::unordered_map<stmt_c, volatility_result_t> *mapper)
        : v_ana_ {is_loop, nullptr} {
        volatility_analysis_t::typed_addresser_t addresser;
        if (mapper) {
            addresser = map_addresser<
                    std::unordered_map<stmt_c, volatility_result_t>>(*mapper);
        } else {
            addresser = temp_data_addresser<result_t, volatility_result_t,
                    &result_t::r1>();
        }
        v_ana_.stmt_result_func_ = addresser;
    }
    void view(define_c v) override {
        v_ana_.view(v, passlet::PRE_VISIT);
        ssa_viewer_t::view(v);
        v_ana_.view(v, passlet::POST_VISIT);
    };

    func_c dispatch(func_c f) override {
        v_ana_.view(f, passlet::PRE_VISIT);
        ssa_viewer_t::dispatch(f);
        v_ana_.view(f, passlet::POST_VISIT);
        return f;
    }
};

struct reset_temp_data : public ssa_viewer_t {
    using ssa_viewer_t::dispatch;
    stmt_c dispatch(stmt_c f) override {
        f->temp_data().clear();
        ssa_viewer_t::dispatch(f);
        return f;
    }
};

} // namespace volatility_analysis

using namespace passlet;
static constexpr auto s32 = datatypes::s32;

static volatility_result_t::state_t get_status(const stmt &v) {
    return v->get_temp_data()
            .get<volatility_analysis::viewer_t::result_t>()
            .r1.is_volatile_;
}

static std::string get_str(const stmt &v) {
    std::stringstream ss;
    v->to_string(ss, 0);
    return ss.str();
}

static void check(const func_c &ssa_ccc, bool is_loop,
        volatility_result_t::state_t expected_loop_var) {
    volatility_analysis::viewer_t(is_loop, nullptr).dispatch(ssa_ccc);
    auto &body = ssa_ccc->body_.static_as<stmts>()->seq_;
    ASSERT_EQ(get_str(body.at(1)), "var __tmp0: s32 = 3");
    ASSERT_EQ(get_status(body.at(1)), volatility_result_t::NO);

    ASSERT_EQ(get_str(body.at(2)), "var __tmp1: s32 = (a + __tmp0)");
    ASSERT_EQ(get_status(body.at(2)), volatility_result_t::NO);

    ASSERT_EQ(get_str(body.at(4)), "var __tmp3: s32 = (a + __tmp2)");
    ASSERT_EQ(get_status(body.at(4)), volatility_result_t::NO);

    // indexing should be volatile
    ASSERT_EQ(get_str(body.at(9)), "var __tmp7: s32 = A[__tmp6]");
    ASSERT_EQ(get_status(body.at(9)), volatility_result_t::YES);

    // depends on indexing, should be volatile
    ASSERT_EQ(get_str(body.at(10)), "var __tmp8: s32 = (__tmp5 + __tmp7)");
    ASSERT_EQ(get_status(body.at(10)), volatility_result_t::YES);

    ASSERT_EQ(get_str(body.at(13)), "var __tmp10: s32 = (a + __tmp9)");
    ASSERT_EQ(get_status(body.at(13)), volatility_result_t::NO);

    // depends on global var, should be volatile
    ASSERT_EQ(get_str(body.at(15)), "var __tmp12: s32 = (__tmp10 + __tmp11)");
    ASSERT_EQ(get_status(body.at(15)), volatility_result_t::YES);

    auto &the_for_body = body.at(22)
                                 .checked_as<for_loop>()
                                 ->body_.checked_as<stmts>()
                                 ->seq_;

    ASSERT_EQ(get_str(the_for_body.at(0)),
            "var loop_v_0: s32 = phi(loop_v, loop_v_1 loop)");
    ASSERT_EQ(get_status(the_for_body.at(0)), expected_loop_var);

    ASSERT_EQ(get_str(the_for_body.at(2)),
            "var loop_v_1: s32 = (loop_v_0 + __tmp17)");
    ASSERT_EQ(get_status(the_for_body.at(2)), expected_loop_var);

    // loop var depends on global var
    ASSERT_EQ(get_str(the_for_body.at(3)),
            "var loop_g_2: s32 = phi(loop_g, loop_g_3 loop)");
    ASSERT_EQ(get_status(the_for_body.at(3)), volatility_result_t::YES);

    ASSERT_EQ(get_str(the_for_body.at(5)),
            "var loop_g_3: s32 = (loop_g_2 + __tmp20)");
    ASSERT_EQ(get_status(the_for_body.at(5)), volatility_result_t::YES);

    ASSERT_EQ(get_str(the_for_body.at(8)),
            "var __tmp24: s32 = (loop_v_1 * __tmp23)");
    ASSERT_EQ(get_status(the_for_body.at(8)), expected_loop_var);

    // depends on for-loop-var
    ASSERT_EQ(get_str(the_for_body.at(12)),
            "var __tmp27: s32 = (__tmp25 + __tmp26)");
    ASSERT_EQ(get_status(the_for_body.at(12)), expected_loop_var);
}

TEST(GCCore_CPU_volatility_analysis, TestVolatilityAnalysis) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32)) {
        _bind_(A, a);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        A[a + 3] = a + 2;
        A[a] = (a + 2) + A[0];
        A[a] = (a + 2) + g;
        _var_ex_(loop_v, s32, linkage::local, 0);
        _var_ex_(loop_g, s32, linkage::local, 0);
        _for_(i, 0, 100) {
            loop_v = loop_v + 1;
            loop_g = loop_g + g;
            A[a] = loop_v * 3;
            A[a] = builder::make_cast(s32, i) + 2;
        }
        _return_(loop_g);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    // loop vars should not be volatile
    check(ssa_ccc, false, volatility_result_t::NO);
    volatility_analysis::reset_temp_data().dispatch(ssa_ccc);
    // loop vars should be volatile
    check(ssa_ccc, true, volatility_result_t::YES);

    volatility_analysis::reset_temp_data().dispatch(ssa_ccc);
    std::unordered_map<stmt_c, volatility_result_t> result_map;

    volatility_analysis::viewer_t(false, &result_map).dispatch(ssa_ccc);
    ASSERT_FALSE(result_map.empty());
    bool met = false;
    for (auto &kv : result_map) {
        if (kv.first.static_as<define_c>()->var_.checked_as<var>()->name_
                == "__tmp0") {
            ASSERT_EQ(kv.second.is_volatile_, volatility_result_t::NO);
            met = true;
        }
    }
    ASSERT_TRUE(met);
}
