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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_DISPATCH_KEY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_DYNAMIC_DISPATCH_KEY_HPP
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <compiler/ir/sc_data_format.hpp>
#include <runtime/dynamic_dispatch/ops/impl_type.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace runtime {
union dispatch_key;
}
class tunable_op_t;
struct context_t;
class sc_op;
// the common base class of op_dispatch_key_t and combind_op_dispatch_key_t.
struct op_dispatch_key_base_t {
    virtual ~op_dispatch_key_base_t() {}
    // set dispatch key for op.
    virtual void set_op_dispatch_key(const std::shared_ptr<sc_op> &node,
            const std::shared_ptr<context_t> &ctx) const = 0;
    virtual std::vector<runtime::dispatch_key>
    convert_to_runtime_format_vec() const = 0;
};

// the dispatch key type for lowering. Will be used in a map in lowering. The
// key is this struct and the value is kernel.
struct op_dispatch_key_t : public op_dispatch_key_base_t {
    // Currently only need for tunable op. Size is same as in_out_formats, and
    // illustrate the config of input/outputs. E.g matmul_core config (M, N,
    // K)[32, 16, 64], we got {{32, 64}, {64, 16}, {32, 16}}.
    std::vector<std::vector<sc_dim>> var_block_;
    // a vector of input/output formats, order is input 0,1,..., output 0,1,...
    std::vector<sc_data_format_t> in_out_formats_;
    // the op can be dispatched as padding or not.
    // todo(cuij): find if the field could be deleted if we complete query by
    // juxtaposed. If we don't need query of combination any more.
    int impl_ = impl_kind_t::normal;
    op_dispatch_key_t() = default;
    virtual ~op_dispatch_key_t() {}
    op_dispatch_key_t(const std::vector<sc_data_format_t> &formats,
            int impl = impl_kind_t::normal)
        : in_out_formats_(formats), impl_(impl) {}
    op_dispatch_key_t(const std::vector<std::vector<sc_dim>> &var_block,
            const std::vector<sc_data_format_t> &formats, bool impl = false)
        : var_block_(var_block), in_out_formats_(formats), impl_(impl) {}
    bool operator==(const op_dispatch_key_t &other) const;
    bool operator!=(const op_dispatch_key_t &other) const;
    void set_op_dispatch_key(const std::shared_ptr<sc_op> &node,
            const std::shared_ptr<context_t> &ctx) const override;
    std::vector<runtime::dispatch_key>
    convert_to_runtime_format_vec() const override;
};

struct impl_op_dispatch_key_t : public op_dispatch_key_base_t {
    int impl_ = impl_kind_t::normal;
    int repeat_;
    virtual ~impl_op_dispatch_key_t() {}
    // impl kernel dispatch uses same table of outer kernel dispatch, so set
    // repeat here to match the number of outer kernel keys.
    impl_op_dispatch_key_t(int impl, int repeat)
        : impl_(impl), repeat_(repeat) {}
    void set_op_dispatch_key(const std::shared_ptr<sc_op> &node,
            const std::shared_ptr<context_t> &ctx) const override;
    std::vector<runtime::dispatch_key>
    convert_to_runtime_format_vec() const override;
};

struct combined_op_dispatch_key_t : public op_dispatch_key_base_t {
    combined_op_dispatch_key_t() = default;
    combined_op_dispatch_key_t(std::initializer_list<op_dispatch_key_t> keys)
        : keys_(std::move(keys)) {}
    combined_op_dispatch_key_t(std::vector<op_dispatch_key_t> &&keys)
        : keys_(std::move(keys)) {}
    bool operator==(const combined_op_dispatch_key_t &other) const;
    bool operator!=(const combined_op_dispatch_key_t &other) const;
    op_dispatch_key_t &operator[](size_t i) { return keys_[i]; }
    const op_dispatch_key_t &operator[](size_t i) const { return keys_[i]; }

    size_t size() const { return keys_.size(); }
    void set_op_dispatch_key(const std::shared_ptr<sc_op> &node,
            const std::shared_ptr<context_t> &ctx) const override;
    std::vector<runtime::dispatch_key>
    convert_to_runtime_format_vec() const override;
    // explict op_dispatch_key_t here, not impl_op_dispatch_key_t because it
    // could not be combined.
    std::vector<op_dispatch_key_t> keys_;
};

struct dispatch_key_cmper_t {
    bool operator()(
            const op_dispatch_key_t &key0, const op_dispatch_key_t &key1) const;
};

struct impl_dispatch_key_cmper_t {
    bool operator()(const impl_op_dispatch_key_t &key0,
            const impl_op_dispatch_key_t &key1) const;
};

struct combined_dispatch_key_cmper_t {
    bool operator()(const combined_op_dispatch_key_t &key0,
            const combined_op_dispatch_key_t &key1) const;
};

// common base struct for dispatch_key_set_t and combined_dispatch_key_set_t.
struct disaptch_key_set_t;
struct dispatch_key_set_base_t {
    virtual ~dispatch_key_set_base_t() {}
    virtual size_t size() const = 0;
    virtual void for_each_key_process(
            const std::function<void(const op_dispatch_key_base_t *)> &callback)
            = 0;
    virtual std::set<op_dispatch_key_t, dispatch_key_cmper_t> &get_inner_set()
            = 0;
    virtual std::shared_ptr<dispatch_key_set_base_t> copy() const = 0;
};

struct dispatch_key_set_t : public dispatch_key_set_base_t {
    using inner_set_t = std::set<op_dispatch_key_t, dispatch_key_cmper_t>;
    dispatch_key_set_t() = default;
    dispatch_key_set_t(const inner_set_t &set) : set_(set) {}
    size_t size() const override { return set_.size(); }
    void for_each_key_process(
            const std::function<void(const op_dispatch_key_base_t *)> &callback)
            override;
    inner_set_t &get_inner_set() override;
    std::shared_ptr<dispatch_key_set_base_t> copy() const override;
    inner_set_t set_;
};

struct impl_dispatch_key_set_t : public dispatch_key_set_base_t {
    using inner_set_t
            = std::set<impl_op_dispatch_key_t, impl_dispatch_key_cmper_t>;
    impl_dispatch_key_set_t() = default;
    impl_dispatch_key_set_t(const inner_set_t &set) : set_(set) {}
    size_t size() const override { return set_.size(); }
    void for_each_key_process(
            const std::function<void(const op_dispatch_key_base_t *)> &callback)
            override;
    std::set<op_dispatch_key_t, dispatch_key_cmper_t> &get_inner_set() override;
    std::shared_ptr<dispatch_key_set_base_t> copy() const override;
    inner_set_t set_;
};

struct combined_dispatch_key_set_t : public dispatch_key_set_base_t {
    using inner_set_t = std::set<combined_op_dispatch_key_t,
            combined_dispatch_key_cmper_t>;
    // modified inp is for compatibility of fused op as its internal tuanble op
    // and fusible op are in seperate graphs.
    combined_dispatch_key_set_t(
            const std::vector<std::shared_ptr<sc_op>> &inputs,
            const std::shared_ptr<sc_op> &modified_inp = nullptr);
    combined_dispatch_key_set_t(
            const std::vector<std::shared_ptr<dispatch_key_set_base_t>>
                    &dispatch_sets);
    combined_dispatch_key_set_t(const inner_set_t &set) : set_(set) {}
    void internal_construct(
            const std::vector<std::shared_ptr<dispatch_key_set_base_t>>
                    &dispatch_sets,
            const std::vector<std::shared_ptr<sc_op>> &inputs
            = std::vector<std::shared_ptr<sc_op>>(),
            const std::shared_ptr<sc_op> &modified_inp = nullptr);
    size_t size() const override { return set_.size(); }
    void for_each_key_process(
            const std::function<void(const op_dispatch_key_base_t *)> &callback)
            override;
    std::set<op_dispatch_key_t, dispatch_key_cmper_t> &get_inner_set() override;
    std::shared_ptr<dispatch_key_set_base_t> copy() const override;
    inner_set_t set_;
};

std::vector<int> get_default_impl_dispatch_candidates();
std::vector<int> get_dynamic_impl_dispatch_candidates(
        tunable_op_t *op, const std::shared_ptr<context_t> &ctx);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
