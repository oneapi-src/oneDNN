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

#include "gpu/jit/pass/bank_conflict.hpp"

#include "gpu/jit/ir/fma.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// FIXME: Use convolution-agnostic mechanism to skip zero-points related calls.
static bool is_zero_points_call(const stmt_t &s) {
    auto is_zp_var = [&](const expr_t &e) {
        auto &base = get_base(e);
        auto &name = base.as<var_t>().name;
        return name.find("zp_") == 0;
    };
    if (is_func_call<dpas_t>(s)) {
        auto &src1 = dpas_t::arg_src1(s);
        auto &src2 = dpas_t::arg_src2(s);
        return is_zp_var(src1) || is_zp_var(src2);
    }
    if (is_func_call<mad_t>(s)) {
        auto &src1 = mad_t::arg_src1(s);
        auto &src2 = mad_t::arg_src2(s);
        return is_zp_var(src1) || is_zp_var(src2);
    }
    return false;
}

class bank_conflict_attribute_injector_t : public ir_mutator_t {
public:
    object_t _mutate(const alloc_t &obj) override {
        all_buf_sizes_.emplace(obj.buf, obj.size);

        auto new_obj = ir_mutator_t::_mutate(obj);
        if (bufs_.count(obj.buf) == 0) return new_obj;

        init_attr();

        auto new_attrs = obj.attrs;
        new_attrs.push_back(attr_);
        auto &body = new_obj.as<alloc_t>().body;
        return alloc_t::make(obj.buf, obj.size, obj.kind, new_attrs, body);
    }

    object_t _mutate(const func_call_t &obj) override {
        if (is_frozen) return ir_mutator_t::_mutate(obj);
        if (is_zero_points_call(obj)) return ir_mutator_t::_mutate(obj);

        bool is_mad = obj.func.is<mad_t>();
        bool is_dpas = obj.func.is<dpas_t>();
        auto *send = obj.func.as_ptr<send_t>();
        bool is_load = send && (send->is_load() || send->is_load_2d());

        if (is_mad || is_dpas) {
            auto dst_buf = ptr_base(obj.args[0]);
            auto src0_buf = ptr_base(obj.args[1]);
            auto src1_buf = ptr_base(obj.args[2]);
            auto src2_buf = ptr_base(obj.args[3]);

            // src0 may be null in some cases, skip it.
            if (!src0_buf.is_empty()) bufs_.insert(src0_buf);
            bufs_.insert(src1_buf);
            bufs_.insert(src2_buf);

            instructions_.insert(obj);
        } else if (is_load) {
            // Returns minimal 2^B so that there is x such that:
            //   x * 2^B <= a <= b < (x + 1) * 2^B
            auto min_pow2_span = [](int a, int b) {
                int same_left_bits = 0;
                for (int i = 31; i >= 0; i--) {
                    int b0 = ((uint32_t)a >> i) & 0x1;
                    int b1 = ((uint32_t)b >> i) & 0x1;
                    if (b0 != b1) break;
                    same_left_bits++;
                }
                return 1 << (32 - same_left_bits);
            };
            auto &buf = send_t::arg_reg_buf(obj);
            auto &base = (is_var(buf) ? buf : buf.as<ptr_t>().base);
            int off = (is_var(buf) ? 0 : to_cpp<int>(buf.as<ptr_t>().off));
            int size = send->payload_size();
            int span = min_pow2_span(off, off + size - 1);
            int &min_block_size = all_buf_min_block_sizes[base];
            min_block_size = std::max(min_block_size, span);
        }
        return ir_mutator_t::_mutate(obj);
    }

private:
    void init_attr() {
        if (!attr_.is_empty()) return;

        is_frozen = true;
        std::vector<stmt_t> instructions;
        for (auto &s : instructions_)
            instructions.push_back(s);

        std::vector<expr_t> buf_vec;
        std::vector<int> buf_sizes;
        std::vector<int> buf_min_block_sizes;
        for (auto &buf : bufs_) {
            buf_vec.push_back(buf);
            buf_sizes.push_back(all_buf_sizes_.at(buf));
            auto it = all_buf_min_block_sizes.find(buf);
            int min_block_size
                    = (it == all_buf_min_block_sizes.end() ? 0 : it->second);
            buf_min_block_sizes.push_back(min_block_size);
        }
        attr_ = bank_conflict_attr_t::make(
                buf_vec, buf_sizes, buf_min_block_sizes, instructions);
    }

    static expr_t ptr_base(const expr_t &e) {
        if (e.is<var_t>()) return e;
        auto *ptr = e.as_ptr<ptr_t>();
        if (ptr) return e.as<ptr_t>().base;
        return expr_t();
    }

    object_map_t<expr_t, int> all_buf_sizes_;
    object_map_t<expr_t, int> all_buf_min_block_sizes;
    object_eq_set_t<expr_t> bufs_;
    object_eq_set_t<stmt_t> instructions_;
    bool is_frozen = false;

    alloc_attr_t attr_;
};

stmt_t inject_bank_conflict_attribute(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = bank_conflict_attribute_injector_t().mutate(s);
    trace_pass("inject_bank_conflict_attribute", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
