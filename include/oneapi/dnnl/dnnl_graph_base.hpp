/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

/// @file
/// Utilities for C++ API

#ifndef ONEAPI_DNNL_DNNL_GRAPH_BASE_HPP
#define ONEAPI_DNNL_DNNL_GRAPH_BASE_HPP

#include <stdexcept>
#include <string>
#include <type_traits>

#include "oneapi/dnnl/dnnl_graph_types.h"

/// @addtogroup dnnl_graph_api
/// @{

namespace dnnl {
namespace graph {

/// @addtogroup dnnl_graph_api_utils Utilities
/// Utility types and definitions
/// @{

/// dnnl graph exception class.
///
/// This class captures the status returned by a failed C API function and
/// the error message from the call site.
struct error : public std::exception {
    dnnl_graph_result_t result;
    std::string detailed_message;

    /// Constructs an instance of an exception class.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    error(dnnl_graph_result_t result, const std::string &message)
        : result(result)
        , detailed_message(message + ": " + result2str(result)) {}

    /// Convert dnnl_graph_result_t to string.
    ///
    /// @param result The error status returned by a C API function.
    /// @return A string that describes the error status
    std::string result2str(dnnl_graph_result_t result) {
        switch (result) {
            case dnnl_graph_result_success: return "success";
            case dnnl_graph_result_not_ready: return "not ready";
            case dnnl_graph_result_error_device_not_found:
                return "device not found";
            case dnnl_graph_result_error_unsupported: return "unsupported";
            case dnnl_graph_result_error_invalid_argument:
                return "invalid argument";
            case dnnl_graph_result_error_compile_fail: return "compile fail";
            case dnnl_graph_result_error_invalid_index: return "invalid index";
            case dnnl_graph_result_error_invalid_graph: return "invalid graph";
            case dnnl_graph_result_error_invalid_shape: return "invalid shape";
            case dnnl_graph_result_error_invalid_type: return "invalid type";
            case dnnl_graph_result_error_invalid_op: return "invalid op";
            case dnnl_graph_result_error_miss_ins_outs:
                return "miss inputs or outputs";
            default: return "unknown error";
        }
    }

    /// Returns the explanatory string.
    ///
    /// @return A const char * that describes the error status
    const char *what() const noexcept override {
        return detailed_message.c_str();
    }

    /// Checks the return status and throws an error in case of failure.
    ///
    /// @param result The error status returned by a C API function.
    /// @param message The error message.
    static void check_succeed(
            dnnl_graph_result_t result, const std::string &message) {
        if (result != dnnl_graph_result_success) throw error(result, message);
    }
};

template <bool B>
using requires = typename std::enable_if<B, bool>::type;

template <typename T, requires<std::is_same<T, float>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_f32;
}

template <typename T, requires<std::is_same<T, int8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_s8;
}

template <typename T, requires<std::is_same<T, uint8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_u8;
}

// TODO(wuxun): now use int16 to simulate float16, need fix in the future
template <typename T, requires<std::is_same<T, int16_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_f16;
}

// TODO(wuxun): now use uint16 to simulate Bfloat16, need fix in the future
template <typename T, requires<std::is_same<T, uint16_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_bf16;
}

template <typename T,
        requires<!std::is_same<T, float>::value
                && !std::is_same<T, int16_t>::value
                && !std::is_same<T, uint16_t>::value
                && !std::is_same<T, int8_t>::value
                && !std::is_same<T, uint8_t>::value> = true>
constexpr dnnl_graph_data_type_t get_data_type() {
    return dnnl_graph_data_type_undef;
}

/// @} dnnl_graph_api_utils

} // namespace graph
} // namespace dnnl

/// @} dnnl_graph_api

#endif
