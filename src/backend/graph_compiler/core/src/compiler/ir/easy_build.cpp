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

#include "easy_build.hpp"
#include <utility>
#include <compiler/config/context.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace builder {
for_range_simulator_t range(const std::string &name, for_loop &out, expr min,
        expr extent, expr step, for_type type, int num_threads) {
    return for_range_simulator_t(builder::get_current_builder(), &out, name,
            std::move(min), std::move(extent), std::move(step), type,
            num_threads);
}

for_range_simulator_t range_nobind(const std::string &name, expr min,
        expr extent, expr step, for_type type, int num_threads) {
    return for_range_simulator_t(builder::get_current_builder(), nullptr, name,
            std::move(min), std::move(extent), std::move(step), type,
            num_threads);
}

for_range_simulator_t range(for_loop &out, expr min, expr extent, expr step,
        for_type type, int num_threads) {
    return for_range_simulator_t(builder::get_current_builder(), &out,
            "!!!unamed", std::move(min), std::move(extent), std::move(step),
            type, num_threads);
}

for_range_simulator_t range(
        expr min, expr extent, expr step, for_type type, int num_threads) {
    return for_range_simulator_t(builder::get_current_builder(), nullptr,
            "!!!unamed", std::move(min), std::move(extent), std::move(step),
            type, num_threads);
}

func_simulator_t _make_func_simulator(const std::string &name, func_t *outfunc,
        sc_data_type_t dtype, std::vector<std::vector<expr>> &&args) {
    std::vector<expr> flattened;
    for (auto &a : args) {
        for (auto &arg : a) {
            flattened.emplace_back(std::move(arg));
        }
    }
    return func_simulator_t(name, outfunc, dtype, std::move(flattened));
}

std::vector<expr> _make_arg(
        const char *name, sc_data_type_t dtype, const std::vector<int> &args) {
    expr ret;
    if (args.empty()) {
        ret = builder::make_var(dtype, name);
    } else {
        std::vector<expr> dims;
        dims.reserve(args.size());
        for (auto i : args) {
            dims.emplace_back(i);
        }
        ret = builder::make_tensor(name, dims, dtype);
    }
    return std::vector<expr> {ret};
}

std::vector<expr> _make_arg(const char *name, sc_data_type_t dtype,
        std::initializer_list<unsigned long> args) { // NOLINT,
    // We must use unsigned long here to let g++ and MSVC to correctly let UL
    // number literals find correct overload version of function.
    expr ret;
    if (args.size() == 0) {
        ret = builder::make_var(dtype, name);
    } else {
        std::vector<expr> dims;
        dims.reserve(args.size());
        for (auto i : args) {
            dims.emplace_back(i);
        }
        ret = builder::make_tensor(name, dims, dtype);
    }
    return std::vector<expr> {ret};
}

std::vector<expr> _make_arg(
        const char *name, sc_data_type_t dtype, const std::vector<expr> &args) {
    expr ret;
    if (args.empty()) {
        ret = builder::make_var(dtype, name);
    } else {
        ret = builder::make_tensor(name, args, dtype);
    }
    return std::vector<expr> {ret};
}

std::vector<expr> _make_arg(const char *name, sc_data_type_t dtype,
        std::initializer_list<int> args) {
    return _make_arg(name, dtype, std::vector<int>(args));
}

std::vector<expr> _make_arg(const char *name, sc_data_type_t dtype) {
    return std::vector<expr> {builder::make_var(dtype, name)};
}

func_t _decl_func(const std::string &name, sc_data_type_t dtype,
        std::vector<std::vector<expr>> &&args) {
    std::vector<expr> flattened;
    for (auto &a : args) {
        for (auto &arg : a) {
            flattened.emplace_back(std::move(arg));
        }
    }
    return builder::make_func(name, flattened, stmt(), dtype);
}

} // namespace builder
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
