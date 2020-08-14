/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef COMMON_STREAM_ATTR_HPP
#define COMMON_STREAM_ATTR_HPP

#include <cassert>
#include "dnnl.h"

#include "c_types_map.hpp"
#include "nstl.hpp"

struct dnnl_stream_attr : public dnnl::impl::c_compatible {
    dnnl_stream_attr(dnnl::impl::engine_kind_t kind) : kind_(kind) {}

    dnnl::impl::engine_kind_t get_engine_kind() { return kind_; }

private:
    dnnl::impl::engine_kind_t kind_;
};

#endif
