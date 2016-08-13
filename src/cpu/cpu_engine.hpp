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

#ifndef CPU_ENGINE_HPP
#define CPU_ENGINE_HPP

#include <assert.h>

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "../common/engine.hpp"

namespace mkldnn { namespace impl { namespace cpu {

using namespace mkldnn::impl::status;

class cpu_engine: public engine {
private:
    bool _lazy;
public:
    cpu_engine(bool lazy)
        : engine(lazy ? engine_kind::cpu_lazy : engine_kind::cpu)
        , _lazy(lazy) {}
    virtual bool is_lazy() const { return _lazy; }
    virtual bool is_ok() const { return true; }
    virtual status_t submit(size_t n, primitive *primitives[],
            primitive **error_primitive) {
        assert(error_primitive);
        *error_primitive = 0;
        for (size_t i = 0; i < n; i++) {
            status_t rc = primitives[i]->execute();
            if (rc != success) {
                *error_primitive = primitives[i];
                return rc;
            }
        }
        return success;
    }

    virtual primitive_desc_init_f *get_primitive_inits() const;
    virtual reorder_primitive_desc_init_f *get_reorder_inits() const;
};

class cpu_engine_factory: public engine_factory {
private:
    bool _lazy;
public:
    cpu_engine_factory(bool lazy): _lazy(lazy) {}
    virtual size_t count() { return 1; }
    virtual engine_kind_t kind()
    { return _lazy ? engine_kind::cpu_lazy : engine_kind::cpu; }
    virtual status_t engine_create(engine **aengine, size_t index) {
        assert(index == 0);
        *aengine = new cpu_engine(_lazy);
        return success;
    };
};

extern cpu_engine_factory engine_factory;
extern cpu_engine_factory engine_factory_lazy;

}}}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
