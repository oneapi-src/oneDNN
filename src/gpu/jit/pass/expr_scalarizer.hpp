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

#ifndef GPU_JIT_PASS_EXPR_SCALARIZER_HPP
#define GPU_JIT_PASS_EXPR_SCALARIZER_HPP

#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class expr_scalarizer_t : public ir_mutator_t {
public:
    expr_scalarizer_t(int elems, int idx,
            const object_map_t<expr_t, std::vector<expr_t>> &vec_vars)
        : elems_(elems), idx_(idx), vec_vars_(vec_vars) {}

    object_t _mutate(const cast_t &obj) override {
        if (obj.is_bool_vec_u16()) return obj;
        auto type = obj.type;
        auto expr = mutate(obj.expr);
        if (!type.is_scalar()) {
            ir_assert(type.elems() == elems_) << expr;
            type = type.scalar();
        }
        return cast_t::make(type, expr, obj.saturate);
    }

    object_t _mutate(const var_t &obj) override {
        if (obj.type.is_scalar()) return obj;

        auto it = vec_vars_.find(obj);
        ir_assert(it != vec_vars_.end()) << "Can't find variable: " << obj;
        ir_assert(int(it->second.size()) == elems_);
        return it->second[idx_];
    }

    object_t _mutate(const shuffle_t &obj) override {
        expr_t new_obj = ir_mutator_t::_mutate(obj);
        auto &shuffle = new_obj.as<shuffle_t>();
        ir_assert(shuffle.type.elems() == elems_) << new_obj;
        return new_obj[idx_];
    }

private:
    int elems_;
    int idx_;
    const object_map_t<expr_t, std::vector<expr_t>> &vec_vars_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
