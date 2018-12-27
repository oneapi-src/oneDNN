/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_MEMORY_HPP
#define CPU_MEMORY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_primitive.hpp"
#include "memory.hpp"
#include "memory_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;

struct cpu_memory_t: public memory_t {
    struct pd_t: public memory_pd_t {
        pd_t(engine_t *engine): memory_pd_t(engine) {}
        pd_t(engine_t *engine, const memory_desc_t *adesc)
            : memory_pd_t(engine, adesc) {}
        virtual ~pd_t() {}
        virtual pd_t *clone() const override {
            return new pd_t(engine(), desc());
        }
        virtual status_t create_primitive(
                primitive_t **primitive) const override {
            UNUSED(primitive);
            return mkldnn::impl::status::unimplemented;
        }
        virtual status_t create_memory(memory_t **memory) const override {
            return safe_ptr_assign<memory_t>(*memory, new cpu_memory_t(this));
        }
    };

    cpu_memory_t(const pd_t *apd)
        : memory_t(apd)
        , data_(nullptr) {}
    virtual ~cpu_memory_t() {}

    virtual status_t get_data_handle(void **handle) const override {
        *handle = static_cast<void *>(data_);
        return success;
    }

    virtual mkldnn::impl::status_t set_data_handle(void *handle) override {
        data_ = static_cast<char *>(handle);
        return zero_pad();
    }

    virtual mkldnn::impl::status_t zero_pad() const override;

private:
    const pd_t *pd() const { return (const pd_t *)memory_t::pd(); }
    char *data_;

    template <mkldnn::impl::data_type_t>
    mkldnn::impl::status_t typed_zero_pad() const;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
