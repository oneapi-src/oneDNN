/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include <algorithm>
#include <functional>
#include <limits>
#include <math.h>

#include "../../target_machine.hpp"
#include "config.hpp"
#include <runtime/config.hpp>
#include <util/utils.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
extern "C" int get_matmul_dyn_cfg_single(int in, bool is_batch) {
    assert(in > 0);
    const int blk_step = 16;
    int blk = 16;
    bool has_no_tail = false;
    int padded_in = std::numeric_limits<int>::max();
    if (is_batch && in <= 16) {
        if (in <= 2) { return 2; }
        if (in <= 4) { return 4; }
        if (in <= 8) { return 8; }
        return 16;
    }
    for (int i = 1; i <= 4; i++) {
        int cur_blk = blk_step * i;
        if (cur_blk == 48) { continue; }
        int cur_num_blk = utils::divide_and_ceil(in, cur_blk);
        int cur_padded_in = cur_num_blk * cur_blk;
        if (in % cur_padded_in == 0) {
            has_no_tail = true;
            blk = cur_blk;
        } else if (!has_no_tail && in / (float)cur_padded_in >= 0.8) {
            blk = cur_blk;
            padded_in = cur_padded_in;
        } else if (!has_no_tail) {
            if (cur_padded_in <= padded_in) {
                blk = cur_blk;
                padded_in = cur_padded_in;
            }
        }
    }
    return blk;
}

// according to sub block candidates.
inline int get_mmm_sub_block_floor(const int x) {
    // todo: recover this in following prs.
    // if (x >= 8) { return 8; }
    if (x >= 4) { return 4; }
    return 1;
}
void get_managed_matmul_config(const runtime::target_machine_t &tm,
        int &M_split_num, int &N_split_num, int &M_sub_block, int &N_sub_block,
        int &K_sub_block, int &im_loop_order, const int M, const int N,
        const int K, const int iim_block, const int iin_block,
        const int iik_block, const int sizeofdtypeA, const int sizeofdtypeC,
        bool is_int8, bool is_f32, bool is_dynamic) {
    im_loop_order = 0;
    const int num_threads = runtime_config_t::get().get_num_threads();
    auto thread_factors = utils::get_factors(num_threads);
    float cost = std::numeric_limits<float>::max();
    int split_n = 1;
    bool is_special_fm
            = tm.cpu_flags_.family == 6 && tm.cpu_flags_.model == 143;
    auto cal_cost = [&](int i) {
        int num_M_block
                = utils::divide_and_ceil(M / iim_block, num_threads / i);
        int num_N_block = utils::divide_and_ceil(N / iin_block, i);
        int num_brgemm = num_M_block * num_N_block;
        int num_core = std::min(i, N / iin_block)
                * std::min(num_threads / i, M / iim_block);
        // Cost = Shape_efficient_weight *
        // (workload_balance + divide_X_plenty) / core_utilitizaiton
        // single core gemm prefers square shape for A and B.
        // For small workload, the A and B shape is not a key problem, but the
        // num_core and num_brgemm is important to performance. Use 2048 to
        // reduce the shape weight on small shape.
        float new_cost;
        float sew = 1024 + M * i / num_threads + N / i;
        if ((K >= 1024 && is_int8 && !tm.use_amx())
                || (K >= 512 && is_f32 && !is_special_fm)) {
            // Cost += empty_cores, making M_split_num * N_split_num closer to
            // num_threads
            float empty_cores = num_threads - i * (num_threads / i);
            if (((N >= 1024 && is_int8 && M <= 2 * N)
                        || (N >= 256 && is_f32 && M <= 64))
                    || (is_f32 && M <= 256 && N >= 1024 && K >= 1024)) {
                // give bigger splits on N when N is bigger
                new_cost = sew * (num_brgemm + num_threads / i / 2) / num_core
                        + empty_cores;
            } else if (N >= 256 && is_f32 && M <= 256) {
                new_cost = sew * (num_brgemm + i + num_threads / i * 2)
                                / num_core
                        + empty_cores;
            } else {
                new_cost = sew * (num_brgemm + i / 2) / num_core + empty_cores;
            }
        } else {
            // give bigger splits on N when N is bigger
            if (N >= 16 * M && N >= 4096 && !is_f32) {
                // TODO(xianhang): in some mlp shapes, only one or some few
                // layres have big N while others are small. Give bigger splits
                // on N would break fusion to some extent, which influences the
                // performance. The logic will be refactored after involving
                // graph-level loop up to make all the matmul use the same split
                // manner
                new_cost = sew * (num_brgemm + num_threads / i / 2) / num_core;
            } else {
                new_cost = sew * (num_brgemm + 8 * i) / num_core;
            }
        }

        if (new_cost < cost) {
            split_n = i;
            cost = new_cost;
        }
    };

    // different between dynamic and static, static may choice non-factor of
    // num_threads.
    if (!is_dynamic) {
        for (int i = 1; i <= num_threads; i++) {
            cal_cost(i);
        }
    } else {
        for (auto &i : thread_factors) {
            cal_cost(i);
        }
    }
    M_split_num = num_threads / split_n;
    N_split_num = split_n;
    if (is_int8 && N <= 512 && K <= 512) {
        // for int8 datatype and small N/Ks, we prefer to give splits only on M
        // when M is small, num_threadsx1x1 split is the same as 1x1x1 split,
        // which runs on single core
        M_split_num = num_threads;
        N_split_num = 1;
    } else if (N <= 192 && K <= 192) {
        // for other datatypes, we prefer to give splits only on M with much
        // smaller N/Ks
        M_split_num = num_threads;
        N_split_num = 1;
    } else if ((M == iim_block && is_special_fm)
            || (M == iim_block && !is_special_fm && num_threads <= 4)) {
        M_split_num = 1;
        if (num_threads <= 4) {
            // magic number = 4096, needs to be further discussed for pretty big
            // K magic number = 4, needs to be further discussed for different M
            if ((K < 4096 || M <= 4) && !is_f32) {
                N_split_num = num_threads;
            } else {
                if (K > N * 4 && thread_factors.size() > 2) {
                    N_split_num = num_threads / thread_factors.at(1);
                } else {
                    N_split_num = num_threads;
                }
            }
        } else {
            // for really small M with super big N and K, despites N is bigger
            // than K, giving splits on K has performance advantage
            auto possible_splits = thread_factors;
            auto split_idx = 0;
            if (is_int8) {
                if (K >= 4096 && N >= 4096 && M <= 4) {
                    auto split_idx = 1;
                    if (N >= K) {
                        split_idx = possible_splits.size() > 3 ? 2 : 1;
                    } else if (K >= 4 * N) {
                        split_idx = possible_splits.size() > 4 ? 3 : 1;
                    }
                    N_split_num = num_threads / possible_splits.at(split_idx);
                } else {
                    split_idx = K >= 4096
                            ? (N < 2 * K ? (
                                       N <= K / 2 && possible_splits.size() > 2
                                               ? 2
                                               : 1)
                                         : (K >= 4096 ? 1 : 0))
                            : 0;
                    N_split_num = num_threads / possible_splits.at(split_idx);
                }
            } else {
                // works well on bf16, needs to be further discussed for f32
                if (K >= 4096) {
                    if (M < 16) {
                        split_idx = possible_splits.size() > 6
                                ? (N >= 4 * K ? 2 : 3)
                                : (num_threads > 32 ? 1 : 0);
                        if (N >= 10 * K) {
                            if (M > 16 || num_threads <= 28) { split_idx = 1; }
                        }
                    } else {
                        split_idx = K >= 4 * N
                                ? (possible_splits.size() > 6
                                                ? 3
                                                : (num_threads > 32 ? 2 : 0))
                                : ((N >= 10 * K || num_threads <= 32) ? 0 : 1);
                    }
                    N_split_num = num_threads / possible_splits.at(split_idx);
                }
            }
        }
    } else if (K >= 8192) {
        // for really big K, we need to give splits on K
        if (M < N) {
            auto possible_splits = utils::get_factors(M_split_num);
            if (possible_splits.size() > 2 && N / M < 3) {
                M_split_num = M_split_num / possible_splits[1];
            } else {
                M_split_num = 1;
                int K_split_num = num_threads == 1 ? 1 : thread_factors.at(1);
                N_split_num = num_threads / K_split_num;
            }
        } else {
            auto possible_splits = utils::get_factors(N_split_num);
            if (possible_splits.size() > 2) {
                N_split_num = N_split_num / possible_splits[1];
            }
        }
    } else if (is_f32 && !is_special_fm && M <= 256 && N >= 256 && K >= 512) {
        // f32 special case
        // for small M, consider giving splits on big K
        if (K >= 1024 && K >= 2 * N && thread_factors.size() > 3) {
            // give bigger splits on K
            int K_split_num = thread_factors.at(2);
            N_split_num = N_split_num / K_split_num > 0
                    ? N_split_num / K_split_num
                    : N_split_num;
        } else if (thread_factors.size() > 2 && N >= 1024 && N != K) {
            int K_split_num = thread_factors.at(1);
            N_split_num = N_split_num / K_split_num > 0
                    ? N_split_num / K_split_num
                    : N_split_num;
        } else if (N == 256 && K == 512 && thread_factors.size() > 3) {
            // special requirements in dlrm shapes, will refactor logic after
            // involving graph-level loop up
            if (M_split_num >= 2) { M_split_num /= 2; }
            if (N_split_num >= 2) { N_split_num /= 2; }
        }
    } else if (M / iim_block < 2 && (N >= 16 * M || K >= 16 * M)) {
        // int8 special case
        if (is_int8 && !tm.use_amx()) {
            M_split_num = 1;
            int K_split_num = 1;
            if (K >= 16 * M) {
                K_split_num
                        = thread_factors.size() > 2 ? thread_factors.at(1) : 1;
            }
            N_split_num = num_threads / K_split_num;
        }
    }
    int single_M = utils::divide_and_ceil(
                           utils::divide_and_ceil(M, iim_block), M_split_num)
            * iim_block;
    int single_N = utils::divide_and_ceil(
                           utils::divide_and_ceil(N, iin_block), N_split_num)
            * iin_block;
    int single_K = K;
    int L2_size = static_cast<int>(tm.cpu_flags_.getDCacheSize(2));
    int single_K_threshold
            = (single_M * single_N * sizeofdtypeA < L2_size ? 2048 : 4096)
            / sizeofdtypeA;
    if (is_f32 && !is_special_fm && num_threads / M_split_num / N_split_num <= 2
            && M >= 128) {
        // if no split is given on K axis, bigger K_sub_block is required
        single_K_threshold /= 4;
    } else if (is_f32 && num_threads <= 4) {
        single_K_threshold /= 2;
    }
    if (single_K >= single_K_threshold) {
        K_sub_block = utils::divide_and_ceil(single_K, single_K_threshold);
        int K_split_num = num_threads / M_split_num / N_split_num;
        while (K / iik_block / K_split_num < K_sub_block && K_sub_block > 1) {
            K_sub_block--;
        }
        int L2_K = utils::divide_and_ceil(
                           utils::divide_and_ceil(single_K, iik_block),
                           K_sub_block)
                * iik_block;
        // sizeofdtypeA* (M * K) + sizeofdtypeB * (N * K) + sizeofdtypeC(M * N)
        // <= L2_size, let N == P * M, then (P + 1) * sizeofdtypeA * M * K +
        // sizeofdtypeC * M * P * M <= L2_size Then M = (sqrt(((P + 1) *
        // sizeofdtypeA * K) ^ 2 + 4 * P* sizeofdtypeC * L2_size) - (P + 1) *
        // sizeofdtypeA * K)/ (2 * P * sizeofdtypeC)
        int P = single_N > single_M
                ? (single_N / single_M > 16 ? (is_f32 ? 4 : 16)
                                            : single_N / single_M)
                : 1;
        int L2_MN = (sqrt(pow((P + 1) * sizeofdtypeA * L2_K, 2)
                             + 4 * P * sizeofdtypeC * L2_size)
                            - (P + 1) * sizeofdtypeA * L2_K)
                / (2 * P * sizeofdtypeC);
        M_sub_block = std::max(1, single_M / L2_MN);
        N_sub_block = std::max(1, single_N / L2_MN);
    } else {
        // sizeofdtypeA * M * K + sizeofdtypeB * N * K <= L2_size
        // let let N == P * M, then
        // M = L2_size / ((1 + P) * sizeofdtypeA * K)
        int P = single_N > single_M
                ? (single_N / single_M > 16 ? (is_f32 ? 4 : 16)
                                            : single_N / single_M)
                : 1;
        int L2_MN = L2_size / ((1 + P) * sizeofdtypeA * single_K);
        M_sub_block = std::max(1, single_M / L2_MN);
        N_sub_block = std::max(1, single_N / L2_MN);
        K_sub_block = 1;
    }
    while (M / iim_block / M_split_num < M_sub_block && M_sub_block > 1) {
        M_sub_block--;
    }
    while (N / iin_block / N_split_num < N_sub_block && N_sub_block > 1) {
        N_sub_block--;
    }
    if (is_dynamic) {
        M_sub_block = get_mmm_sub_block_floor(M_sub_block);
        N_sub_block = get_mmm_sub_block_floor(N_sub_block);
        K_sub_block = 1;
    }
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
