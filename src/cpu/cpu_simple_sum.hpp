/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_SIMPLE_SUM_HPP
#define CPU_SIMPLE_SUM_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct cpu_simple_sum_t: public c_compatible {
    typedef typename prec_trait<data_type>::type data_t;

    static bool applicable(const nstl::vector<cpu_memory_t::pd_t> &src_pds_,
                            const nstl::vector<double> &scale_,
                            cpu_memory_t::pd_t &dst_pds_)
    {
        const memory_desc_wrapper o_d(&dst_pds_);
        bool ok = true;
        for (size_t i = 0; i < src_pds_.size(); ++i) {
            const memory_desc_wrapper i_d(&src_pds_[i]);
            ok = ok && i_d.data_type() == data_type
                && o_d.data_type() == data_type && i_d.format() == o_d.format()
                && i_d.is_dense() && o_d.is_dense();
        }
        return ok;
    }

    static void execute(const nstl::vector<cpu_memory_t::pd_t> &src_pds_,
                        const nstl::vector<double> &scale_,
                        cpu_memory_t::pd_t &dst_pds_,
                        cpu_primitive_t *sum)
    {
        const int num_arrs = src_pds_.size();

        auto output = reinterpret_cast<data_t *>(sum->memory());
        const memory_desc_wrapper o_d(&dst_pds_);
        output += o_d.blk_off(0);
        const size_t nelems = o_d.nelems();

        const data_t *input_ptrs[num_arrs];
        for (int a = 0; a < num_arrs; ++a) {
            const memory_desc_wrapper i_d(&src_pds_[a]);

            input_ptrs[a] = reinterpret_cast<const data_t *>(
                    sum->input_memory(a)) + i_d.blk_off(0);
        }

        double scale = scale_[0];
        const data_t *input_ptr = &(input_ptrs[0][0]);

#       pragma omp parallel for schedule(static)
        for (size_t e = 0; e < nelems; ++e) {
             output[e] = scale*input_ptr[e];
        }
        for (int a = 1; a < num_arrs; ++a) {
            scale = scale_[a];
            input_ptr = &(input_ptrs[a][0]);
#           pragma omp parallel for schedule(static)
            for (size_t e = 0; e < nelems; ++e) {
                output[e] += scale*input_ptr[e];
            }
        }
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
