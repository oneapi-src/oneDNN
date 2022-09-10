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

#include "gpu/jit/pass/peephole.hpp"

#include "gpu/jit/pass/simplify.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class peephole_optimizer_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &obj) override {
        auto old_obj = ir_mutator_t::_mutate(obj);
        auto new_obj
                = simplify_rewrite_with_ternary(old_obj, /*recursive=*/false);
        auto *ternary = new_obj.as_ptr<ternary_op_t>();
        if (!ternary) return std::move(new_obj);

        switch (ternary->op_kind) {
            case op_kind_t::_add3: {
                bool ok = true;
                // Allowed form: add3(dword/word, dword/word, dword/word).
                ok &= add3_type_ok(ternary->a);
                ok &= add3_type_ok(ternary->b);
                ok &= add3_type_ok(ternary->c);
                ok &= !is_const(ternary->a);
                ok &= !is_const(ternary->b);
                if (!ok) new_obj = old_obj;
                break;
            }
            case op_kind_t::_mad: {
                bool ok = false;
                if (try_int_mad(ternary))
                    ok = true;
                else if (try_float_mad(ternary))
                    ok = true;
                if (!ok) new_obj = old_obj;
                break;
            }
            default: ir_error_not_expected();
        }
        return std::move(new_obj);
    }

private:
    static type_t real_type(const expr_t &e) {
        auto *imm = e.as_ptr<int_imm_t>();
        if (!imm) return e.type();
        if (int_imm_t::try_shrink_type<int16_t>(imm->value))
            return type_t::s16();
        if (int_imm_t::try_shrink_type<int32_t>(imm->value))
            return type_t::s32();
        return type_t::s64();
    }

    static bool try_int_mad(const ternary_op_t *ternary) {
        auto a_type = real_type(ternary->a);
        auto b_type = real_type(ternary->b);
        auto c_type = real_type(ternary->c);
        bool ok = true;
        // Allowed form: mad(dword, dword, word).
        ok &= utils::one_of(a_type, type_t::s32(), type_t::u32());
        ok &= utils::one_of(b_type, type_t::s32(), type_t::u32());
        ok &= utils::one_of(c_type, type_t::s16(), type_t::u16());
        return ok;
    }

    static bool try_float_mad(const ternary_op_t *ternary) {
        auto op_ok = [](const expr_t &e) {
            if (is_const(e) || is_const_broadcast(e)) return false;
            if (!e.type().is_f32()) return false;
            return true;
        };
        if (!op_ok(ternary->a)) return false;
        if (!op_ok(ternary->b)) return false;
        if (!op_ok(ternary->c)) return false;
        return true;
    }

    static bool add3_type_ok(const expr_t &e) {
        auto t = real_type(e);
        if (!t.is_scalar()) return false;
        switch (t.kind()) {
            case type_kind_t::s32:
            case type_kind_t::u32: return !is_const(e);
            case type_kind_t::s16:
            case type_kind_t::u16: return true;
            default: return false;
        }
    }
};

stmt_t optimize_peephole(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = peephole_optimizer_t().mutate(s);
    trace_pass("optimize_peephole", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
