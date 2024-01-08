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

#include <iostream>
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/pass/ir_copy.hpp>
#include <compiler/ir/transform/loop_split.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_loop_split_cpp, TestIfSpilttedLoop) {
    builder::ir_builder_t builder;

    _function_(datatypes::void_t, aa, _arg_("jj", datatypes::index),
            _arg_("buf", datatypes::index, {100, 200})) {
        _bind_(jj, buf);
        _for_(i, 0, 20) {
            _for_(j1, 0, 50) {
                _if_(j1 < 20) {
                    // _for_(j1, 0, 20)
                    buf[{i, j1}] = i * j1;
                }
            }
            _for_(j2, 0, 50) {
                _if_(j2 <= 20) {
                    // _for_(j2, 0, 21)
                    buf[{i, j2}] = i * j2;
                }
            }
            _for_(j3, 1, 50, 4) {
                _if_(j3 > 20) {
                    // _for_(j3, 21, 50, 4)
                    buf[{i, j3}] = i * j3;
                }
            }
            _for_(j4, 1, 50, 4) {
                _if_(j4 >= 20) {
                    // _for_(j4, 21, 50, 4)
                    buf[{i, j4}] = i * j4;
                }
            }
            _for_(j5, 0, jj) {
                _if_(j5 < 20) {
                    // no change
                    buf[{i, j5}] = i * j5;
                }
            }
            _for_(j6, jj, 50) {
                _if_(j6 > 20) {
                    // no change
                    buf[{i, j6}] = i * j6;
                }
            }
            _for_(j7, 0, 50) {
                _if_(i < 10) {
                    // no change
                    buf[{i, j7}] = i * j7;
                }
            }
        }
    }

    _function_(datatypes::void_t, bbb, _arg_("jj", datatypes::index),
            _arg_("buf", datatypes::index, {100, 200})) {
        _bind_(jj, buf);
        _for_(i, 0, 20, 1) {
            _for_(j1, 0, 20) { buf[{i, j1}] = i * j1; }
            _for_(j2, 0, 21) { buf[{i, j2}] = i * j2; }
            _for_(j3, 21, 50, 4) { buf[{i, j3}] = i * j3; }
            _for_(j4, 21, 50, 4) { buf[{i, j4}] = i * j4; }
            _for_(j5, 0, jj) {
                _if_(j5 < 20) { buf[{i, j5}] = i * j5; }
            }
            _for_(j6, jj, 50) {
                _if_(j6 > 20) { buf[{i, j6}] = i * j6; }
            }
            _for_(j7, 0, 50) {
                _if_(i < 10) { buf[{i, j7}] = i * j7; }
            }
        }
    }

    loop_splitter_t loop_splitter;
    auto aaa = loop_splitter(aa);

    ir_comparer cmper;
    EXPECT_TRUE(aaa->equals(bbb, cmper));
}
