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

#ifndef GPU_JIT_IR_GRF_PERMUTATION_HPP
#define GPU_JIT_IR_GRF_PERMUTATION_HPP

#include <array>

#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to permute registers. Used to permute registers after applying
// dpas -> dpasw transformation.
class grf_permutation_t {
public:
    grf_permutation_t() { permutation_.fill(-1); }

    int map(int off) const {
        ir_assert(off >= 0 && off < max_regs);
        if (permutation_[off] == -1) return off;
        return permutation_[off];
    }

    bool is_empty() const { return is_empty_; }

    void set_permute(int old_off, int new_off) {
        ir_assert(old_off >= 0 && old_off < max_regs);
        if (old_off == new_off || new_off == -1) return;
        is_empty_ = false;
        ir_assert(utils::one_of(permutation_[old_off], -1, new_off))
                << "Already assigned to a different offset.";
        permutation_[old_off] = new_off;
    }

    bool operator==(const grf_permutation_t &other) const {
        for (int i = 0; i < max_regs; i++) {
            if (permutation_[i] != other.permutation_[i]) return false;
        }
        return true;
    }

    bool operator!=(const grf_permutation_t &other) const {
        return !operator==(other);
    }

private:
    static const int max_regs = 256;

    std::array<int, max_regs> permutation_;
    bool is_empty_ = true;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
