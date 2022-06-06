/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#ifdef _WIN32
#include <windows.h>
#endif

#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "oneapi/dnnl/dnnl_graph.hpp"

#include "test_allocator.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
#endif

#define EXAMPLE_SWITCH_TYPE(type_enum, type_key, ...) \
    switch (type_enum) { \
        case dnnl::graph::logical_tensor::data_type::f32: { \
            using type_key = float; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::f16: { \
            using type_key = int16_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::bf16: { \
            using type_key = uint16_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::u8: { \
            using type_key = uint8_t; \
            __VA_ARGS__ \
        } break; \
        case dnnl::graph::logical_tensor::data_type::s8: { \
            using type_key = int8_t; \
            __VA_ARGS__ \
        } break; \
        default: \
            throw std::runtime_error( \
                    "Not supported data type in current example."); \
    }

// get exact data handle from tensor according to the data type
void *get_handle_from_tensor(const dnnl::graph::tensor &src,
        dnnl::graph::logical_tensor::data_type dtype) {
    switch (dtype) {
        case dnnl::graph::logical_tensor::data_type::f32:
            return src.get_data_handle<float>();
            break;
        case dnnl::graph::logical_tensor::data_type::f16:
            return src.get_data_handle<int16_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::bf16:
            return src.get_data_handle<uint16_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::u8:
            return src.get_data_handle<uint8_t>();
            break;
        case dnnl::graph::logical_tensor::data_type::s8:
            return src.get_data_handle<int8_t>();
            break;
        default:
            throw std::runtime_error(
                    "Not supported data type in current example.");
    }
}

dnnl::graph::op::kind opstr2kind(std::string kind) {
    const std::unordered_map<std::string, dnnl::graph::op::kind> op_map = {
            {"Abs", dnnl::graph::op::kind::Abs},
            {"Add", dnnl::graph::op::kind::Add},
            {"AvgPool", dnnl::graph::op::kind::AvgPool},
            {"AvgPoolBackprop", dnnl::graph::op::kind::AvgPoolBackprop},
            {"BatchNormInference", dnnl::graph::op::kind::BatchNormInference},
            {"BatchNormForwardTraining",
                    dnnl::graph::op::kind::BatchNormForwardTraining},
            {"BatchNormTrainingBackprop",
                    dnnl::graph::op::kind::BatchNormTrainingBackprop},
            {"BiasAddBackprop", dnnl::graph::op::kind::BiasAddBackprop},
            {"Clamp", dnnl::graph::op::kind::Clamp},
            {"ClampBackprop", dnnl::graph::op::kind::ClampBackprop},
            {"Concat", dnnl::graph::op::kind::Concat},
            {"Convolution", dnnl::graph::op::kind::Convolution},
            {"ConvolutionBackpropData",
                    dnnl::graph::op::kind::ConvolutionBackpropData},
            {"ConvolutionBackpropFilters",
                    dnnl::graph::op::kind::ConvolutionBackpropFilters},
            {"ConvTranspose", dnnl::graph::op::kind::ConvTranspose},
            {"ConvTransposeBackpropData",
                    dnnl::graph::op::kind::ConvTransposeBackpropData},
            {"ConvTransposeBackpropFilters",
                    dnnl::graph::op::kind::ConvTransposeBackpropFilters},
            {"Divide", dnnl::graph::op::kind::Divide},
            {"Elu", dnnl::graph::op::kind::Elu},
            {"EluBackprop", dnnl::graph::op::kind::EluBackprop},
            {"Erf", dnnl::graph::op::kind::Erf},
            {"Exp", dnnl::graph::op::kind::Exp},
            {"GELU", dnnl::graph::op::kind::GELU},
            {"GELUBackprop", dnnl::graph::op::kind::GELUBackprop},
            {"HardSwish", dnnl::graph::op::kind::HardSwish},
            {"HardSwishBackprop", dnnl::graph::op::kind::HardSwishBackprop},
            {"HardTanh", dnnl::graph::op::kind::HardTanh},
            {"HardTanhBackprop", dnnl::graph::op::kind::HardTanhBackprop},
            {"LayerNorm", dnnl::graph::op::kind::LayerNorm},
            {"LayerNormBackprop", dnnl::graph::op::kind::LayerNormBackprop},
            {"Log", dnnl::graph::op::kind::Log},
            {"LogSoftmax", dnnl::graph::op::kind::LogSoftmax},
            {"LogSoftmaxBackprop", dnnl::graph::op::kind::LogSoftmaxBackprop},
            {"MatMul", dnnl::graph::op::kind::MatMul},
            {"Maximum", dnnl::graph::op::kind::Maximum},
            {"MaxPool", dnnl::graph::op::kind::MaxPool},
            {"MaxPoolBackprop", dnnl::graph::op::kind::MaxPoolBackprop},
            {"Minimum", dnnl::graph::op::kind::Minimum},
            {"Multiply", dnnl::graph::op::kind::Multiply},
            {"Pow", dnnl::graph::op::kind::Pow},
            {"PowBackprop", dnnl::graph::op::kind::PowBackprop},
            {"PReLU", dnnl::graph::op::kind::PReLU},
            {"PReLUBackprop", dnnl::graph::op::kind::PReLUBackprop},
            {"ReduceL1", dnnl::graph::op::kind::ReduceL1},
            {"ReduceL2", dnnl::graph::op::kind::ReduceL2},
            {"ReduceMax", dnnl::graph::op::kind::ReduceMax},
            {"ReduceMean", dnnl::graph::op::kind::ReduceMean},
            {"ReduceMin", dnnl::graph::op::kind::ReduceMin},
            {"ReduceProd", dnnl::graph::op::kind::ReduceProd},
            {"ReduceSum", dnnl::graph::op::kind::ReduceSum},
            {"ReLU", dnnl::graph::op::kind::ReLU},
            {"ReLUBackprop", dnnl::graph::op::kind::ReLUBackprop},
            {"Round", dnnl::graph::op::kind::Round},
            {"Sigmoid", dnnl::graph::op::kind::Sigmoid},
            {"SigmoidBackprop", dnnl::graph::op::kind::SigmoidBackprop},
            {"SoftMax", dnnl::graph::op::kind::SoftMax},
            {"SoftMaxBackprop", dnnl::graph::op::kind::SoftMaxBackprop},
            {"SoftPlus", dnnl::graph::op::kind::SoftPlus},
            {"SoftPlusBackprop", dnnl::graph::op::kind::SoftPlusBackprop},
            {"Sqrt", dnnl::graph::op::kind::Sqrt},
            {"SqrtBackprop", dnnl::graph::op::kind::SqrtBackprop},
            {"Square", dnnl::graph::op::kind::Square},
            {"SquaredDifference", dnnl::graph::op::kind::SquaredDifference},
            {"Subtract", dnnl::graph::op::kind::Subtract},
            {"Tanh", dnnl::graph::op::kind::Tanh},
            {"TanhBackprop", dnnl::graph::op::kind::TanhBackprop},
            {"Wildcard", dnnl::graph::op::kind::Wildcard},
            {"BiasAdd", dnnl::graph::op::kind::BiasAdd},
            {"Interpolate", dnnl::graph::op::kind::Interpolate},
            {"Index", dnnl::graph::op::kind::Index},
            {"InterpolateBackprop", dnnl::graph::op::kind::InterpolateBackprop},
            {"PowBackpropExponent", dnnl::graph::op::kind::PowBackpropExponent},
            {"End", dnnl::graph::op::kind::End},
            {"Quantize", dnnl::graph::op::kind::Quantize},
            {"Dequantize", dnnl::graph::op::kind::Dequantize},
            {"Reorder", dnnl::graph::op::kind::Reorder},
            {"TypeCast", dnnl::graph::op::kind::TypeCast},
            {"StaticReshape", dnnl::graph::op::kind::StaticReshape},
            {"StaticTranspose", dnnl::graph::op::kind::StaticTranspose},
            {"DynamicReshape", dnnl::graph::op::kind::DynamicReshape},
            {"DynamicTranspose", dnnl::graph::op::kind::DynamicTranspose},
            {"DynamicQuantize", dnnl::graph::op::kind::DynamicQuantize},
            {"DynamicDequantize", dnnl::graph::op::kind::DynamicDequantize},
            {"Sign", dnnl::graph::op::kind::Sign},
            {"Negative", dnnl::graph::op::kind::Negative},
            {"Reciprocal", dnnl::graph::op::kind::Reciprocal}};
    const auto it = op_map.find(kind);
    if (it != op_map.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Unsupported opkind.\n");
    }
}

dnnl::graph::engine::kind engine_kind_str2kind(std::string kind) {
    if (kind == "cpu") {
        return dnnl::graph::engine::kind::cpu;
    } else if (kind == "gpu") {
        return dnnl::graph::engine::kind::gpu;
    } else {
        return dnnl::graph::engine::kind::any;
    }
}

dnnl::graph::graph::fpmath_mode fpmath_mode_kind_str2kind(std::string kind) {
    if (kind == "bf16") {
        return dnnl::graph::graph::fpmath_mode::bf16;
    } else if (kind == "f16") {
        return dnnl::graph::graph::fpmath_mode::f16;
    } else if (kind == "any") {
        return dnnl::graph::graph::fpmath_mode::any;
    } else if (kind == "tf32") {
        return dnnl::graph::graph::fpmath_mode::tf32;
    } else {
        return dnnl::graph::graph::fpmath_mode::strict;
    }
}

dnnl::graph::engine::kind parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return dnnl::graph::engine::kind::cpu;
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        std::string engine_kind_str = argv[1];
        if (engine_kind_str == "cpu") {
            return dnnl::graph::engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            return dnnl::graph::engine::kind::gpu;
        } else {
            throw std::runtime_error("only support cpu and gpu engine\n");
        }
    }
    // If all above fails, the example should be ran properly
    std::cout << "Inappropriate engine kind." << std::endl
              << "Please run the example like this: " << argv[0] << " [cpu|gpu]"
              << "." << std::endl;
    exit(1);
}

inline dnnl_graph_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty()
            ? 0
            : std::accumulate(dims.begin(), dims.end(), (dnnl_graph_dim_t)1,
                    std::multiplies<dnnl_graph_dim_t>());
}

// fill the memory according to the given value
//  src -> target memory buffer
//  total_size -> total number of bytes of this buffer
//  val -> fixed value for initialization
template <typename T>
void fill_buffer(void *src, size_t total_size, int val) {
    size_t num_elem = static_cast<size_t>(total_size / sizeof(T));
    T *src_casted = static_cast<T *>(src);
    // can be implemented through OpenMP
    for (size_t i = 0; i < num_elem; ++i)
        *(src_casted + i) = static_cast<T>(val);
}

struct logical_id_manager {
    static logical_id_manager &get() {
        static logical_id_manager id_mgr;
        return id_mgr;
    }

    size_t operator[](const std::string &arg) {
        const auto &it = knots_.find(arg);
        if (it != knots_.end()) return it->second;
        if (frozen_) {
            std::cout << "Unrecognized argument [" << arg << "]!\n";
            std::abort();
        }
        const auto &new_it = knots_.emplace(arg, knots_.size());
        if (new_it.second) {
            return new_it.first->second;
        } else {
            std::cout << "New argument [" << arg
                      << "] is failed to be added to knots.\n";
            std::abort();
        }
    }

    void freeze() { frozen_ = true; }

private:
    logical_id_manager() : frozen_(false) {};

    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

template <typename T>
static void compare_data(T *dst, T *ref, size_t size, float rtol = 1e-5f,
        float atol = 1e-6f, bool equal_nan = false) {
    auto cal_error = [&](const float dst, const float ref) -> bool {
        const float diff_f32 = dst - ref;
        const float gap = rtol
                        * (std::abs(ref) > std::abs(dst) ? std::abs(ref)
                                                         : std::abs(dst))
                + atol;
        bool good = std::abs(diff_f32) <= gap;
        return good;
    };

    for (size_t i = 0; i < size; ++i) {
        if (std::isfinite(dst[i]) && std::isfinite(ref[i])) {
            const float ref_f32 = static_cast<float>(ref[i]);
            const float dst_f32 = static_cast<float>(dst[i]);
            if (!cal_error(dst_f32, ref_f32)) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        } else {
            bool cond = (dst[i] == ref[i]);
            if (equal_nan) { cond = std::isnan(dst[i]) && std::isnan(ref[i]); }
            if (!cond) {
                printf("expected = %s, actual = %s\n",
                        std::to_string(ref[i]).c_str(),
                        std::to_string(dst[i]).c_str());
                throw std::runtime_error(
                        "output result is not equal to excepted "
                        "results");
            }
        }
    }
}

#ifdef DNNL_GRAPH_WITH_SYCL
template <typename dtype>
void fill_buffer(
        cl::sycl::queue &q, void *usm_buffer, size_t length, dtype value) {
    dtype *usm_buffer_casted = static_cast<dtype *>(usm_buffer);
    q.parallel_for(cl::sycl::range<1>(length), [=](cl::sycl::id<1> i) {
         int idx = (int)i[0];
         usm_buffer_casted[idx] = value;
     }).wait();
}
#endif

void custom_setenv(const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    SetEnvironmentVariable(name, value);
#else
    ::setenv(name, value, overwrite);
#endif
}

int getenv(const char *name, char *buffer, int buffer_size) {
    if (name == nullptr || buffer_size < 0
            || (buffer == nullptr && buffer_size > 0))
        return INT_MIN;

    int result = 0;
    int term_zero_idx = 0;
    size_t value_length = 0;

#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, buffer, buffer_size);
#else
    const char *value = ::getenv(name);
    value_length = value == nullptr ? 0 : strlen(value);
#endif

    if (value_length > INT_MAX)
        result = INT_MIN;
    else {
        int int_value_length = (int)value_length;
        if (int_value_length >= buffer_size) {
            result = -int_value_length;
        } else {
            term_zero_idx = int_value_length;
            result = int_value_length;
#ifndef _WIN32
            if (value) strncpy(buffer, value, buffer_size - 1);
#endif
        }
    }

    if (buffer != nullptr) buffer[term_zero_idx] = '\0';
    return result;
}

#endif
