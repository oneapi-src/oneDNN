/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include "gpu/intel/jit/v2/conv/bench_data.hpp"

#include <sstream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

std::string bench_data_t::str() const {
    std::ostringstream oss;
    for (int i = 0; i < size(); i++) {
        if (i > 0) oss << std::endl;
        double gops_sec = prbs[i].ops() / times[i].total;
        oss << "bench," << prbs[i].csv_str() << "," << times[i].total << ","
            << gops_sec;
    }
    return oss.str();
}

std::vector<int> bench_data_set_t::find_best_ids(int nbest) const {
    auto idxs = find_best_idxs(nbest);
    std::vector<int> ret;
    for (int idx : idxs)
        ret.push_back(vec_[idx].id);
    return ret;
}

std::vector<bench_data_t> bench_data_set_t::find_best(int nbest) const {
    auto idxs = find_best_idxs(nbest);
    std::vector<bench_data_t> ret;
    for (int idx : idxs)
        ret.push_back(vec_[idx]);
    return ret;
}

std::vector<int> bench_data_set_t::find_best_idxs(int _nbest) const {
    int nbest = std::min(_nbest, (int)vec_.size());
    if (nbest == 0) return {};
    int nprbs = vec_[0].size();
    uint64_t max_time = std::numeric_limits<uint64_t>::max();
    std::vector<uint64_t> best_times(nprbs, max_time);
    std::vector<uint64_t> cur_times(nprbs, max_time);
    for (auto &bd : vec_) {
        for (int i = 0; i < nprbs; i++) {
            best_times[i] = std::min(best_times[i], bd.times[i].total);
        }
    }
    std::unordered_set<int> best_idxs;
    for (int k = 0; k < nbest; k++) {
        double best_geomean = 0;
        int best_idx = -1;
        // Greedily select the kernel that gives the highest improvement in geomean.
        for (int i = 0; i < size(); i++) {
            if (best_idxs.count(i) > 0) continue;
            double geomean = 1.0;
            for (int j = 0; j < nprbs; j++) {
                double ratio = best_times[j]
                        / (double)std::min(
                                cur_times[j], vec_[i].times[j].total);
                geomean *= std::pow(ratio, 1.0 / nprbs);
            }
            if (geomean >= best_geomean) {
                best_geomean = geomean;
                best_idx = i;
            }
        }
        gpu_assert(best_idx != -1);
        for (int j = 0; j < nprbs; j++) {
            cur_times[j]
                    = std::min(cur_times[j], vec_[best_idx].times[j].total);
        }
        best_idxs.insert(best_idx);
    }
    return std::vector<int>(best_idxs.begin(), best_idxs.end());
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
