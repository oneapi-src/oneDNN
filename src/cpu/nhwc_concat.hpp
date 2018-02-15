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

#ifndef NHWC_CONCAT_HPP
#define NHWC_CONCAT_HPP

#include "cpu_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct nhwc_concat_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;
    typedef typename prec_traits<data_type>::type data_t;

    struct pd_t: public cpu_concat_pd_t {
        pd_t(const memory_desc_t *output_d, int n, int concat_dim,
                const cpu_memory_pd_t **input_pds, const primitive_attr_t *attr)
            : cpu_concat_pd_t(output_d, n, concat_dim, input_pds, attr)
        {}

        DECLARE_CPU_CONCAT_PD_T("nhwc:any", nhwc_concat_t);

        virtual status_t init() override {
            using namespace mkldnn::impl::memory_format;

            bool ok = true
                && cpu_concat_pd_t::init() == success
                && concat_dim_ == 1;

            for (size_t i = 0; i < src_pds_.size(); ++i) {
                const memory_desc_wrapper src_d(&src_pds_[i]);
                const memory_desc_wrapper img_d(&src_image_pds_[i]);
                ok = ok
                    && utils::everyone_is(data_type, src_d.data_type(),
                            img_d.data_type())
                    && utils::everyone_is(src_d.format(), img_d.format(), nhwc);

            }
            return ok ? success : unimplemented;
        }
    };

    nhwc_concat_t(const pd_t *conf, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*conf) {
            const int num_srcs = conf_.n_inputs();
            src = (const data_t **)malloc(num_srcs*sizeof(data_t *), 64);
            img = (data_t **)malloc(num_srcs*sizeof(data_t *), 64);
            ic = (int *)malloc(num_srcs*sizeof(int), 64);
        }

    ~nhwc_concat_t() {
        free(src);
        free(img);
        free(ic);
    }

    virtual void execute(event_t *e) {
        execute();
        e->set_state(event_t::ready);
    }


private:
    void execute();
    pd_t conf_;

    const data_t **src;
    data_t **img;
    int *ic;
};

}
}
}
#endif
