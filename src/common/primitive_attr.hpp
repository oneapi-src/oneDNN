/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#ifndef PRIMITIVE_ATTR_HPP
#define PRIMITIVE_ATTR_HPP

#include "mkldnn.h"

#include "utils.hpp"
#include "c_types_map.hpp"

namespace mkldnn {
namespace impl {

struct scales_t: public c_compatible {
    scales_t(): count_(1), mask_(0), scales_(scales_buf_)
    { set(1.); }

    scales_t(const scales_t &rhs): scales_t()
    { set(rhs.count_, rhs.mask_, rhs.scales_); }

    ~scales_t() { cleanup(); }

    scales_t &operator=(const scales_t &rhs) {
        if (&rhs == this)
            return *this;
        status_t status = set(rhs.count_, rhs.mask_, rhs.scales_);
        assert(status == status::success);
        (void)status;
        return *this;
    }

    status_t set(int count, int mask, const float *scales);
    status_t set(float single_scale) { return this->set(1, 0, &single_scale); }

    int count_;
    int mask_;
    float *scales_;

private:
    enum { scales_buf_size = 16 };
    alignas(64) float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr)
            impl::free(scales_);

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }
};

}
}

struct mkldnn_primitive_attr: public mkldnn::impl::c_compatible {
    mkldnn_primitive_attr()
        : round_mode_(mkldnn::impl::round_mode::nearest) {}

    mkldnn_primitive_attr *clone() const
    { return new mkldnn_primitive_attr(*this); }

    mkldnn::impl::status_t set_round_mode(
            mkldnn::impl::round_mode_t round_mode);

    mkldnn::impl::round_mode_t round_mode_;
    mkldnn::impl::scales_t output_scales_;
};

#endif
