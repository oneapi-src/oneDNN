/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_WALK_ORDER_HPP
#define GPU_INTEL_JIT_IR_WALK_ORDER_HPP

#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

// Represents blocked kernel grid walk order, together with assignments to
// X/Y/Z grid IDs (0, 1, 2 indices).
class walk_order_t {
public:
    struct block_t {
        block_t() = default;
        block_t(const pvar_t &dim, int size, int grid_id)
            : dim(dim), size(size), grid_id(grid_id) {}
        pvar_t dim;
        int size = 0;
        int grid_id = -1;
    };

    struct dim_info_t {
        dim_info_t() = default;
        dim_info_t(const pvar_t &dim, int size) : dim(dim), size(size) {
            grid_var = var_t::make(type_t::s32(), dim.str() + "_grid_var");
        }

        pvar_t dim;
        int size = 0;
        expr_t grid_var;
    };

    walk_order_t() = default;
    walk_order_t(const std::string &s) {
        auto parts = gpu_utils::split(s, ",");
        ir_assert(parts.size() <= 3);
        for (int i = 0; i < (int)parts.size(); i++) {
            for (auto &kv : ir_utils::to_string_int_pairs(parts[i])) {
                add(pvar_t(kv.first), kv.second, i);
            }
        }
    }

    void add(const pvar_t &dim, dim_t block_size, int grid_id) {
        if (!blocks_.empty()) {
            auto &last = blocks_.back();
            if (last.dim == dim && last.grid_id == grid_id) {
                last.size *= block_size;
                return;
            }
        }
        blocks_.emplace_back(dim, block_size, grid_id);
    }

    const std::vector<block_t> &blocks() const { return blocks_; }
    const std::vector<dim_info_t> &dim_infos() const { return dim_infos_; }

    bool has(const pvar_t &dim) const {
        for (auto &info : dim_infos_) {
            if (info.dim == dim) return true;
        }
        return false;
    }

    bool is_blocked(int id) const {
        for (auto &info : dim_infos_) {
            if (grid_id(info.dim) != id) continue;
            int count = 0;
            for (auto &b : blocks_) {
                if (b.dim != info.dim) continue;
                count += 1;
            }
            if (count > 1) return true;
        }
        return false;
    }

    std::vector<pvar_t> grid_dims(int id) const {
        std::vector<pvar_t> ret;
        for (auto &info : dim_infos_) {
            if (grid_id(info.dim) == id) ret.push_back(info.dim);
        }
        return ret;
    }

    int grid_id(const pvar_t &dim) const {
        int id = -1;
        for (auto &b : blocks_) {
            if (b.dim != dim) continue;
            if (id == -1) id = b.grid_id;
            ir_assert(b.grid_id == id);
        }
        ir_assert(id != -1);
        return id;
    }

    expr_t grid_var(const pvar_t &dim) const {
        for (auto &info : dim_infos_) {
            if (info.dim == dim) return info.grid_var;
        }
        ir_error_not_expected() << "Grid variable not found: " << dim;
        return expr_t();
    }

    int dim_size(const expr_t &grid_var) const {
        for (auto &info : dim_infos_) {
            if (info.grid_var.is_same(grid_var)) return info.size;
        }
        ir_error_not_expected() << "Grid variable not found: " << grid_var;
        return -1;
    }

    int dim_size(const pvar_t &dim) const { return dim_size(grid_var(dim)); }

    bool is_grid_var(const expr_t &grid_var) const {
        for (auto &info : dim_infos_) {
            if (info.grid_var.is_same(grid_var)) return true;
        }
        return false;
    }

    void finalize(const pvar_tile_t &grid_tile) {
        for (auto &d : grid_tile) {
            int inner_block = 1;
            for (auto &b : blocks_) {
                if (b.dim == d) inner_block *= b.size;
            }
            dim_t outer = utils::div_up(grid_tile[d], inner_block);
            int id = (inner_block != 1 ? grid_id(d) : 0);
            dim_infos_.emplace_back(d, grid_tile[d]);
            if (outer != 1) add(d, outer, id);
        }
        for (auto &info : dim_infos_) {
            int nblocks = 0;
            block_t *first_block = nullptr;
            for (auto &b : blocks_) {
                if (b.dim != info.dim) continue;
                if (!first_block) first_block = &b;
                nblocks++;
            }
            if (nblocks == 1) {
                first_block->size = std::min(first_block->size, info.size);
            }
        }
    }

    std::string str() const {
        std::ostringstream oss;
        for (int id = 0; id < 3; id++) {
            if (id != 0) oss << ",";
            for (auto &b : blocks_) {
                if (b.grid_id != id) continue;
                oss << b.dim << b.size;
            }
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    // Ordered from innermost to outermost.
    std::vector<block_t> blocks_;
    std::vector<dim_info_t> dim_infos_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
