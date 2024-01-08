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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_SOFTMAX_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_SOFTMAX_REF_HPP

#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <test_utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

enum class test_op_type {
    SOFTMAX = 0,
    SOFTMAX_BWD,
    LOG_SOFTMAX,
    LOG_SOFTMAX_BWD
};

inline sc_dim product(const sc_dims &vec) {
    sc_dim res = 1;
    for (auto e : vec) {
        res *= e;
    }
    return res;
}

inline sc_dim flatten_index(const sc_dims &index, const sc_dims &range) {
    int index_size = index.size();
    sc_dim acc = 1;
    sc_dim result = 0;
    for (auto i = index_size - 1; i >= 0; i--) {
        if (i == index_size - 1) {
            result += index[i];
        } else {
            acc *= range[i + 1];
            result += index[i] * acc;
        }
    }
    return result;
}

inline static sc_dims find_rd_axis(
        const sc_dims &input_dims, std::vector<int> &keep_axis) {
    sc_dims axis;
    if (keep_axis.size() == 1 && keep_axis[0] == -1) {
        for (size_t i = 0; i < input_dims.size() - 1; i++) {
            axis.push_back(i);
        }
    } else {
        for (size_t i = 0; i < input_dims.size(); i++) {
            if (std::find(keep_axis.begin(), keep_axis.end(), i)
                    == keep_axis.end()) {
                axis.push_back(i);
            }
        }
    }
    return axis;
}

inline static void get_rd_dims(
        sc_dims &axis, const sc_dims &input_dims, sc_dims &reduced_dims) {
    if (axis.size() == input_dims.size()) {
        reduced_dims.push_back(1);
    } else {
        if (axis.size() == 1) {
            reduced_dims.push_back(input_dims[axis[0]]);
        } else {
            for (auto a : axis) {
                reduced_dims.push_back(input_dims[a]);
            }
        }
    }
}

inline std::vector<float> ref_softmax(const std::vector<float> &data,
        const sc_dims &input_dims, std::vector<int> keep_axis,
        test_op_type test_type = test_op_type::SOFTMAX,
        const std::vector<float> &inp2 = {}) {
    sc_dims axis = find_rd_axis(input_dims, keep_axis);
    std::sort(axis.begin(), axis.end());
    const int num_of_loops = input_dims.size();
    sc_dims lp_vars(num_of_loops, 0);
    std::vector<float> ret(data.size());

    // exp or mul
    for (unsigned i = 0; i < data.size(); i++) {
        switch (test_type) {
            case test_op_type::SOFTMAX:
            case test_op_type::LOG_SOFTMAX: {
                ret[i] = exp(data[i]);
            } break;
            case test_op_type::SOFTMAX_BWD: {
                ret[i] = data[i] * inp2[i];
            } break;
            case test_op_type::LOG_SOFTMAX_BWD: {
                ret[i] = data[i];
            } break;
        }
    }

    // reduce
    sc_dims reduced_dims;
    get_rd_dims(axis, input_dims, reduced_dims);

    std::vector<float> reduce_result;
    reduce_result.resize(product(reduced_dims));

    std::function<void(int)> reduce;
    reduce = [&](int lp_index) {
        for (; lp_vars[lp_index] < input_dims[lp_index]; lp_vars[lp_index]++) {
            if (lp_index == num_of_loops - 1) {
                sc_dims reduce_vars;
                int reduced_index;
                if (axis.size() == 1) {
                    reduced_index = lp_vars[axis[0]];
                } else {
                    if (axis.size() == input_dims.size()) {
                        reduced_index = 0;
                    } else {
                        for (auto a : axis) {
                            reduce_vars.push_back(lp_vars[a]);
                        }
                        reduced_index
                                = flatten_index(reduce_vars, reduced_dims);
                    }
                }
                auto index = flatten_index(lp_vars, input_dims);
                reduce_result[reduced_index] += ret[index];
            } else {
                reduce(lp_index + 1);
            }
        }
        lp_vars[lp_index] = 0;
    };
    reduce(0);
    std::fill(lp_vars.begin(), lp_vars.end(), 0);

    // division or mul(data - rd_val)
    std::function<void(int)> divide_or_mul;
    divide_or_mul = [&](int lp_index) {
        for (; lp_vars[lp_index] < input_dims[lp_index]; lp_vars[lp_index]++) {
            if (lp_index == num_of_loops - 1) {
                sc_dims reduce_vars;
                int reduced_index;
                if (axis.size() == 1) {
                    reduced_index = lp_vars[axis[0]];
                } else {
                    if (axis.size() == input_dims.size()) {
                        reduced_index = 0;
                    } else {
                        for (auto a : axis) {
                            reduce_vars.push_back(lp_vars[a]);
                        }
                        reduced_index
                                = flatten_index(reduce_vars, reduced_dims);
                    }
                }
                auto index = flatten_index(lp_vars, input_dims);
                switch (test_type) {
                    case test_op_type::SOFTMAX: {
                        ret[index] /= reduce_result[reduced_index];
                    } break;
                    case test_op_type::LOG_SOFTMAX: {
                        ret[index] /= reduce_result[reduced_index];
                        ret[index] = std::log(ret[index]);
                    } break;
                    case test_op_type::SOFTMAX_BWD: {
                        ret[index] = inp2[index]
                                * (data[index] - reduce_result[reduced_index]);
                    } break;
                    case test_op_type::LOG_SOFTMAX_BWD: {
                        ret[index] = data[index]
                                - (std::exp(inp2[index])
                                        * reduce_result[reduced_index]);
                    } break;
                }
            } else {
                divide_or_mul(lp_index + 1);
            }
        }
        lp_vars[lp_index] = 0;
    };
    divide_or_mul(0);

    return ret;
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
