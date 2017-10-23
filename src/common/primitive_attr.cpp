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

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

status_t mkldnn_primitive_attr_create(primitive_attr_t **attr) {
    if (attr == nullptr)
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            new mkldnn_primitive_attr);
}

status_t mkldnn_primitive_attr_clone(primitive_attr_t **attr,
        const primitive_attr_t *existing_attr) {
    if (any_null(attr, existing_attr))
        return invalid_arguments;

    return safe_ptr_assign<mkldnn_primitive_attr>(*attr,
            existing_attr->clone());
}

status_t mkldnn_primitive_attr_destroy(primitive_attr_t *attr) {
    if (attr)
        delete attr;

    return success;
}
