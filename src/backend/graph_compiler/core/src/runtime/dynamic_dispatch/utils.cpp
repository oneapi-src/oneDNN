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

#include <limits>
#include "utils.hpp"
#include <util/simple_math.hpp>

namespace sc {
namespace runtime {

int get_dyn_cfg_single(int in, bool has_48) {
    int blk = 16;
    bool has_no_tail = false;
    int padded_in = std::numeric_limits<int>::max();
    for (int i = 1; i <= 4; i++) {
        int cur_blk = 16 * i;
        if (!has_48 && cur_blk == 48) { continue; }
        int cur_num_blk = ::sc::utils::divide_and_ceil(in, cur_blk);
        int cur_padded_in = cur_num_blk * cur_blk;
        if (in % cur_padded_in == 0) {
            has_no_tail = true;
            blk = cur_blk;
        } else if (!has_no_tail && in / (float)cur_padded_in >= 0.8) {
            blk = cur_blk;
            padded_in = cur_padded_in;
        } else if (!has_no_tail) {
            if (cur_padded_in <= padded_in) {
                blk = cur_blk;
                padded_in = cur_padded_in;
            }
        }
    }
    return blk;
}
} // namespace runtime
} // namespace sc
