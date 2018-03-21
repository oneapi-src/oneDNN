/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"
#include "cpu_batch_normalization_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace bnorm_utils {
    void cache_balance(size_t working_set_size, int C_blks, int &C_blks_per_iter,
            int &iters) {
        int nthrs = omp_get_max_threads();
        int l3_size = get_cache_size(3, true) * nthrs / 2;

        C_blks_per_iter = l3_size / working_set_size;

        if (C_blks_per_iter == 0)
            C_blks_per_iter = 1;
        if (C_blks_per_iter > C_blks)
            C_blks_per_iter = C_blks;

        iters = (C_blks + C_blks_per_iter - 1) / C_blks_per_iter;
    }
    void thread_balance(bool do_blocking, int ithr, int nthr, int N, int C_blks,
            int &C_ithr, int &C_nthr, int &C_blk_s, int &C_blk_e, int &N_ithr,
            int &N_nthr, int &N_s, int &N_e) {
        if (nthr <= (int)C_blks) {
            C_ithr = ithr; C_nthr = nthr;
            N_ithr = 0; N_nthr = 1;
            N_s = 0; N_e = N;
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
        } else {
            if (do_blocking) {
                N_nthr = nstl::min((int)N, nthr);
                C_nthr = nstl::min((int)C_blks, nthr / N_nthr);
            } else {
                C_nthr = math::gcd(nthr, (int)C_blks);
                N_nthr = nstl::min((int)N, nthr / C_nthr);
            }
            if (ithr < C_nthr * N_nthr) {
                N_ithr = ithr % N_nthr;
                C_ithr = ithr / N_nthr;
                balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
                balance211(N, N_nthr, N_ithr, N_s, N_e);
            } else {
                N_ithr = C_ithr = -ithr;
                N_s = N_e = C_blk_s = C_blk_e = -1;
            }
        }
    }
};
}
}
}
