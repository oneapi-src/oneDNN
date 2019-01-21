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

#include "memory_tracking.hpp"

#include "cpu_concat_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_concat_t: public cpu_primitive_t {
    struct pd_t: public cpu_concat_pd_t {
        using cpu_concat_pd_t::cpu_concat_pd_t;

        pd_t(const pd_t &rhs): cpu_concat_pd_t(rhs)
        {
            for (size_t i = 0; i < sizeof(perm_) / sizeof(perm_[0]); i++) {
                perm_[i] = rhs.perm_[i];
                iperm_[i] = rhs.iperm_[i];
            }
        }

        DECLARE_CONCAT_PD_T("simple:any", simple_concat_t);

        status_t init() {
            const memory_desc_wrapper dst_d(dst_md());
            bool ok = true
                && cpu_concat_pd_t::init() == status::success
                && dst_d.ndims() <= 6;
            if (!ok) return status::unimplemented;

            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                const memory_desc_wrapper o_d(&src_image_mds_[i]);
                ok = ok
                    && utils::everyone_is(data_type, i_d.data_type(),
                            o_d.data_type())
                    && i_d.format() == o_d.format()
                    && !utils::one_of(i_d.format(), memory_format::blocked,
                            memory_format::wino_fmt)
                    && !i_d.is_additional_buffer();
                if (!ok) return status::unimplemented;
            }

            format_perm();

            // density check
            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                const memory_desc_wrapper o_d(&src_image_mds_[i]);
                ok = ok
                    && nelems_to_concat(i_d) == size_to_concat(i_d)
                    && nelems_to_concat(o_d) == size_to_concat(o_d);
                if (!ok) return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

        int perm_[TENSOR_MAX_DIMS];
        int iperm_[TENSOR_MAX_DIMS];

        size_t nelems_to_concat(const memory_desc_wrapper &data_d) const {
            const int ndims = data_d.ndims();
            auto &blk = data_d.blocking_desc();

            size_t nelems = 1;
            for (int i = perm_[concat_dim()]; i < ndims; i++)
                nelems *= data_d.dims()[iperm_[i]] / blk.block_dims[iperm_[i]];
            for (int i = 0; i < ndims; i++)
                nelems *= blk.block_dims[i];

            return nelems;
        }

    private:
        void format_perm() {
            const memory_desc_wrapper dst_d(dst_md());
            const int ndims = dst_d.ndims();

            strides_t strides;
            utils::array_copy(strides, dst_d.blocking_desc().strides[0], ndims);

            for (int i = 0; i < ndims; i++) iperm_[i] = i;

            for (int i = 0; i < ndims - 1; i++) {
                bool swapped = false;
                for (int j = 0; j < ndims - i - 1; j++) {
                    if (strides[j] < strides[j + 1]) {
                        nstl::swap(strides[j], strides[j + 1]);
                        nstl::swap(iperm_[j], iperm_[j + 1]);
                        swapped = true;
                    }
                }
                if (swapped == false)
                    break;
            }

            for (int i = 0; i < ndims; i++) perm_[iperm_[i]] = i;
        }

        size_t size_to_concat(const memory_desc_wrapper &data_d) const {
            size_t max_size = 0;
            auto &blk = data_d.blocking_desc();
            for (int d = perm_[concat_dim()]; d < data_d.ndims(); ++d) {
                auto block = blk.block_dims[iperm_[d]];
                max_size = nstl::max<size_t>(max_size,
                        size_t(blk.padding_dims[iperm_[d]] / block)
                        * blk.strides[0][iperm_[d]]);
                if (block > 1) max_size = nstl::max(max_size,
                        size_t(block * blk.strides[1][iperm_[d]]));
            }
            return max_size;
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_concat_iptrs, sizeof(data_t *) * n_inputs());
            scratchpad.book(key_concat_optrs, sizeof(data_t *) * n_inputs());
            scratchpad.book(key_concat_nelems, sizeof(size_t) * n_inputs());
            scratchpad.book(key_concat_istrides,
                    sizeof(strides_t) * n_inputs());
        }
    };

    simple_concat_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
