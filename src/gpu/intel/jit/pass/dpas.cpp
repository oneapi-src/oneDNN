/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/intel/jit/pass/dpas.hpp"

#include "gpu/intel/jit/ir/fma.hpp"
#include "gpu/intel/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class mul_mutator_t : public ir_mutator_t {
public:
    struct entry_t {
        stmt_t stmt;
        bool is_dpas = false;
        int dpas_sdepth = 0;
        int dpas_rcount = 0;

        bool is_dpas_8x8() const {
            return is_dpas && dpas_sdepth == 8 && dpas_rcount == 8;
        }
    };

    object_t _mutate(const stmt_group_t &obj) override {
        if (obj.label != stmt_label_t::mul()) return ir_mutator_t::_mutate(obj);
        auto body = mutate_mul(obj.body);
        return stmt_group_t::make(obj.label, body);
    }

    stmt_t mutate_mul(const stmt_t &stmt) const {
        auto stmt_vec = flatten_statements(stmt);
        std::vector<entry_t> entries;
        for (auto &s : stmt_vec) {
            entries.emplace_back();
            auto &e = entries.back();
            e.stmt = s;
            e.is_dpas = is_func_call<dpas_t>(s) && !dpas_t::is_dp4a_call(s);
            if (e.is_dpas) {
                auto &dpas = s.as<func_call_t>().func.as<dpas_t>();
                e.dpas_sdepth = dpas.sdepth;
                e.dpas_rcount = dpas.rcount;
            }
        }
        return mutate_mul_impl(entries);
    }

    virtual stmt_t mutate_mul_impl(
            const std::vector<entry_t> &entries) const = 0;
};

class dpas_atomic_mutator_t : public mul_mutator_t {
public:
    stmt_t mutate_mul_impl(const std::vector<entry_t> &entries) const override {
        stmt_t ret;
        for (size_t i = 0; i < entries.size(); i++) {
            auto s = entries[i].stmt;
            if (i != entries.size() - 1 && entries[i].is_dpas
                    && entries[i + 1].is_dpas) {
                auto &cur_src1 = dpas_t::arg_src1(entries[i].stmt);
                auto &next_src1 = dpas_t::arg_src1(entries[i + 1].stmt);
                auto &cur_src2 = dpas_t::arg_src2(entries[i].stmt);
                auto &next_src2 = dpas_t::arg_src2(entries[i + 1].stmt);
                auto &cur_src2_base = cur_src2.as<ptr_t>().base;
                auto &next_src2_base = next_src2.as<ptr_t>().base;
                if (cur_src1.is_equal(next_src1)
                        && cur_src2_base.is_equal(next_src2_base)) {
                    auto atomic_attr = instruction_modifier_attr_t::make(
                            ngen_proxy::InstructionModifier().with_atomic());
                    auto &call = s.as<func_call_t>();
                    auto *attr
                            = call.attr.as_ptr<instruction_modifier_attr_t>();
                    if (!attr || !attr->mod.is_atomic) {
                        s = atomic_attr.apply_to(s);
                    }
                }
            }
            ret = ret.append(s);
        }
        return ret;
    }
};

stmt_t inject_dpas_atomic(const stmt_t &stmt, bool filter_by_label) {
    if (filter_by_label) return dpas_atomic_mutator_t().mutate(stmt);
    auto ret = dpas_atomic_mutator_t().mutate_mul(stmt);
    return ret;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
