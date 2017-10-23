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

struct mkldnn_primitive_attr: public mkldnn::impl::c_compatible {
    mkldnn_primitive_attr() {}

    mkldnn_primitive_attr *clone() const
    { return new mkldnn_primitive_attr(*this); }
};

#endif
