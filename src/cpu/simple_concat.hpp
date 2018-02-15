/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef SIMPLE_CONCAT_HPP
#define SIMPLE_CONCAT_HPP

#include "cpu_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_concat_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    struct pd_t: public cpu_concat_pd_t {
        pd_t(const memory_desc_t *output_d, int n,
                int concat_dim, const cpu_memory_pd_t **input_pds,
                const primitive_attr_t *attr)
            : cpu_concat_pd_t(output_d, n, concat_dim, input_pds, attr)
        {}
        pd_t(const pd_t &rhs) : cpu_concat_pd_t(rhs) {}

        DECLARE_CPU_CONCAT_PD_T("simple:any", simple_concat_t);

        virtual status_t init() override {
            auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
                return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
            };

            bool ok = true
                && concat_dim_ != 0
                && cpu_concat_pd_t::init() == success
                && src_pds_.size() <= max_num_arrs;

            if (!ok) return unimplemented;

            for (size_t i = 0; i < src_pds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_pds_[i]);
                const memory_desc_wrapper o_d(&src_image_pds_[i]);
                ok = ok
                    && utils::everyone_is(data_type, i_d.data_type(),
                            o_d.data_type())
                    && i_d.format() == o_d.format()
                    && is_dense_no_0(i_d) && is_dense_no_0(o_d);
                }

            return ok ? success : unimplemented;
        }
    };

    simple_concat_t(const pd_t *conf, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*conf) {}

    virtual void execute(event_t *e) {
        execute();
        e->set_state(event_t::ready);
    }

    enum { max_num_arrs = 16 };
    typedef typename prec_traits<data_type>::type data_t;

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

    void execute();
    pd_t conf_;
};

}
}
}

#endif
