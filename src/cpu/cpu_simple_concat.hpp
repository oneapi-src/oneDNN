/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef CPU_SIMPLE_CONCAT_HPP
#define CPU_SIMPLE_CONCAT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

static void catch_me() {}

template <impl::data_type_t data_type>
struct cpu_simple_concat_t: public c_compatible {
    typedef typename prec_trait<data_type>::type data_t;

    static bool applicable(const nstl::vector<cpu_memory_t::pd_t> &src_pds_,
            const nstl::vector<cpu_memory_t::pd_t> &dst_pds_, int concat_dim) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };

        bool ok = concat_dim != 0;
        for (size_t i = 0; i < src_pds_.size(); ++i) {
            const memory_desc_wrapper i_d(&src_pds_[i]);
            const memory_desc_wrapper o_d(&dst_pds_[i]);
            ok = ok && i_d.data_type() == data_type
                && o_d.data_type() == data_type && i_d.format() == o_d.format()
                && is_dense_no_0(i_d) && is_dense_no_0(o_d);
        }
        return ok;
    }

    static void execute(const nstl::vector<cpu_memory_t::pd_t> &src_pds_,
            const nstl::vector<cpu_memory_t::pd_t> &dst_pds_,
            cpu_primitive_t *concat) {
        const int num_arrs = src_pds_.size();
        const data_t *input_ptrs[num_arrs];
        data_t *output_ptrs[num_arrs];
        size_t nelems_no_d0[num_arrs];
        size_t is[num_arrs];

        auto o_base_ptr = reinterpret_cast<data_t *>(concat->memory());

        for (int a = 0; a < num_arrs; ++a) {
            const memory_desc_wrapper i_d(&src_pds_[a]);
            const memory_desc_wrapper o_d(&dst_pds_[a]);

            input_ptrs[a] = reinterpret_cast<const data_t *>(
                    concat->input_memory(a)) + i_d.blk_off(0);
            output_ptrs[a] = o_base_ptr + o_d.blk_off(0);

            nelems_no_d0[a] = nelems_no_dim_0(i_d);
            is[a] = i_d.blocking_desc().strides[0][0];
        }

        const memory_desc_wrapper o_d(&dst_pds_[0]);
        const size_t N = o_d.dims()[0];
        const size_t os = o_d.blocking_desc().strides[0][0];

        catch_me();

#       pragma omp parallel for collapse(2) schedule(static)
        for (size_t n = 0; n < N; ++n) {
            for (int a = 0; a < num_arrs; ++a) {
                /* do coping */
                const data_t *i = &input_ptrs[a][is[a]*n];
                data_t *o = &output_ptrs[a][os*n];
                for (size_t e = 0; e < nelems_no_d0[a]; ++e) o[e] = i[e];
            }
        }
    }

private:
    static size_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static size_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        size_t max_size = 0;
        auto &blk = data_d.blocking_desc();
        for (int d = 1; d < data_d.ndims(); ++d) {
            auto block = blk.block_dims[d];
            max_size = nstl::max(max_size,
                    size_t(blk.padding_dims[d]/block)*blk.strides[0][d]);
            if (block > 1)
                max_size = nstl::max(max_size,
                        size_t(block*blk.strides[1][d]));
        }
        return max_size;
    }

};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
