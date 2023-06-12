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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BODY_GENERATOR_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_BODY_GENERATOR_HPP
#include <memory>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/graph/tensor_detail.hpp>
#include <compiler/ir/sc_stmt.hpp>
#include <unordered_map>
#include <util/general_object.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

using config_ptr = reflection::shared_general_object_t;
class fusion_manager;
class sc_op;
struct graph_tensor;
struct tensor_slice;

namespace tuner {
struct config_space;
using config_space_ptr = std::unique_ptr<tuner::config_space>;
} // namespace tuner

/**
 * The generator base class to generate IR for the body of an Op
 * */
struct body_generator_base_t {
    sc_op *owner_;
    std::vector<logical_tensor_t> in_tensors_;
    std::vector<logical_tensor_t> out_tensors_;
    // extra parameter for internal func pointer.
    expr single_core_func_param_;
    /**
     * simply judge the config is valid or not, then we needn't to generate
     * others in graph
     * */
    virtual bool is_valid_config(
            const context_ptr &ctx, const void *config) const {
        return true;
    }
    /**
     * Generates the tensor IR to the current IR builder.
     * @param ctx the context
     * @param config the configuration
     * @param fusion the fusion manager. The generator should push the anchors
     * to the fusion manager
     * @param inputs the input args of the Op
     * @param outputs the output tensors of the Op
     * @param loops the for-loops to be later scheduled by schedule_loops()
     * dispatch.
     * @return generate status, e.g. success.
     * */
    virtual bool generate(context_ptr ctx, const void *config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const = 0;
    /**
     * Get the single core calculation function e.g. wrapped brgemm.
     */
    virtual func_t get_single_core_func(context_ptr ctx, const void *config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const {
        return nullptr;
    }

    virtual std::vector<expr> get_extra_args_from_func(const func_t &f) const {
        throw std::runtime_error("Unimplemented");
    }

    virtual float get_gflop() const = 0;

    sc_data_type_t get_in_dtypes(size_t idx) const {
        return in_tensors_.at(idx).dtype_;
    }

    sc_data_type_t get_out_dtypes(size_t idx) const {
        return out_tensors_.at(idx).dtype_;
    }

    void set_single_core_func_param(const expr &single_core_func_param) {
        single_core_func_param_ = single_core_func_param;
    }

    //   std::vector<sc_data_type_t> infer_out_dtypes() const {
    //     if (in_tensors_.size()
    //       && (in_tensors_.at(0).dtype_ == datatypes::u8
    //         || in_tensors_.at(1).dtype_ == datatypes::s8)) {
    //       return {datatypes::s32};
    //     } else {
    //       return {datatypes::f32};
    //     }
    //   }

    /**
     * Returns the type-erased default config. You can use `get()` method in
     * the returned object to get the pointer, which can be used in `generate`
     * */
    virtual config_ptr get_default_config(context_ptr ctx) const = 0;

    using config_ptr_vec = std::vector<config_ptr>;
    using impl_kind_map = std::unordered_map<std::vector<int64_t>, int>;
    virtual config_ptr_vec get_dynamic_config_candidates(
            const context_ptr &ctx) const {
        return config_ptr_vec();
    }

    virtual std::vector<uint64_t> convert_config_to_keys(
            const config_ptr &config) const {
        throw std::runtime_error("Unimplement");
    }

    virtual void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const = 0;

    virtual ~body_generator_base_t() = default;

    body_generator_base_t(sc_op *owner,
            std::vector<logical_tensor_t> &&in_tensors,
            std::vector<logical_tensor_t> &&out_tensors)
        : owner_(owner)
        , in_tensors_(std::move(in_tensors))
        , out_tensors_(std::move(out_tensors)) {}

    body_generator_base_t(sc_op *owner,
            const std::vector<logical_tensor_t> &in_tensors,
            const std::vector<logical_tensor_t> &out_tensors)
        : owner_(owner), in_tensors_(in_tensors), out_tensors_(out_tensors) {}
};

using body_generator_ptr = std::unique_ptr<body_generator_base_t>;
template <typename TConfig>
struct body_generator_t : public body_generator_base_t {
    virtual bool is_valid_config(
            const context_ptr &ctx, const TConfig &config) const {
        return true;
    }
    bool is_valid_config(
            const context_ptr &ctx, const void *config) const override {
        return is_valid_config(ctx, *reinterpret_cast<const TConfig *>(config));
    }
    virtual bool generate(context_ptr ctx, const TConfig &config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const = 0;

    bool generate(context_ptr ctx, const void *config, fusion_manager *fusion,
            const std::vector<expr> &inputs, const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override {
        return generate(ctx, *reinterpret_cast<const TConfig *>(config), fusion,
                inputs, outputs, loops);
    }

    virtual func_t get_single_core_func(context_ptr ctx, const TConfig &config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const {
        return nullptr;
    }

    func_t get_single_core_func(context_ptr ctx, const void *config,
            fusion_manager *fusion, const std::vector<expr> &inputs,
            const std::vector<expr> &outputs,
            std::vector<for_loop> &loops) const override {
        return get_single_core_func(ctx,
                *reinterpret_cast<const TConfig *>(config), fusion, inputs,
                outputs, loops);
    }

    virtual void schedule_loops(context_ptr ctx, const TConfig &config,
            stmt body, std::vector<for_loop> &fors) const = 0;

    void schedule_loops(context_ptr ctx, const void *config, stmt body,
            std::vector<for_loop> &fors) const override {
        schedule_loops(
                ctx, *reinterpret_cast<const TConfig *>(config), body, fors);
    }

    body_generator_t(sc_op *owner, const std::vector<logical_tensor_t> &ins,
            const std::vector<logical_tensor_t> &outs)
        : body_generator_base_t {owner, ins, outs} {}

    body_generator_t(sc_op *owner, std::vector<logical_tensor_t> &&ins,
            std::vector<logical_tensor_t> &&outs)
        : body_generator_base_t {owner, std::move(ins), std::move(outs)} {}
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
