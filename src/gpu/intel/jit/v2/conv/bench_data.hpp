/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_V2_CONV_BENCH_DATA_HPP
#define GPU_INTEL_JIT_V2_CONV_BENCH_DATA_HPP

#include "gpu/intel/jit/v2/conv/kernel_desc.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

class bench_data_t {
public:
    int id = -1;
    kernel_desc_t kernel_desc;
    std::vector<problem_t> prbs;
    std::vector<uint64_t> times;

    bench_data_t() = default;
    explicit bench_data_t(int id, const kernel_desc_t &kernel_desc)
        : id(id), kernel_desc(kernel_desc) {}

    int size() const { return (int)prbs.size(); }
    explicit operator bool() const { return size() > 0; }

    void add(const problem_t &prb, uint64_t time) {
        prbs.push_back(prb);
        times.push_back(time);
    }

    std::string str() const;
};

class bench_data_set_t {
public:
    void add(const bench_data_t &bd) { vec_.push_back(bd); }
    int size() const { return (int)vec_.size(); }
    std::vector<bench_data_t>::const_iterator begin() const {
        return vec_.begin();
    }
    std::vector<bench_data_t>::const_iterator end() const { return vec_.end(); }
    std::vector<int> find_best_ids(int nbest) const;
    std::vector<bench_data_t> find_best(int nbest) const;

private:
    std::vector<int> find_best_idxs(int nbest) const;

    std::vector<bench_data_t> vec_;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
