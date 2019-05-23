/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#ifndef PATCH_MKLDNN_HPP
#define PATCH_MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stdlib.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

#include "mkldnn.h"
#include "mkldnn.hpp"
#include "patch_mkldnn.h"
#endif

namespace mkldnn {

/// Base class for all computational primitives.
class patch_primitive: public primitive {
public:
    /// A proxy to C primitive kind enum
    enum class kind {
        resize_bilinear = mkldnn_resize_bilinear,
    };

    using primitive::primitive;
};

enum patch_query {
    resize_bilinear_d = mkldnn_query_resize_bilinear_d,
};

/// @addtogroup cpp_api_memory Memory
/// A primitive to describe data.
///
/// @addtogroup cpp_api_resize_bilinear resize_bilinear
/// A primitive to perform resize_bilinear.
///
/// @sa @ref c_api_resize_bilinear in @ref c_api
/// @{

struct resize_bilinear_forward : public patch_primitive {
    struct desc {
        mkldnn_resize_bilinear_desc_t data;

        desc(prop_kind aprop_kind,
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const int align_corners) {
            error::wrap_c_api(mkldnn_resize_bilinear_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &src_desc.data, &dst_desc.data,
                        &align_corners),
                    "could not init a forward resize_bilinear descriptor");
        }
    };

/// Primitive descriptor for resize_bilinear forward propagation.
    struct primitive_desc : public mkldnn::primitive_desc {
        primitive_desc() = default;

        primitive_desc(const desc &desc, const engine &e)
            : mkldnn::primitive_desc(&desc.data, nullptr, e, nullptr) {}

        primitive_desc(const desc &desc, const primitive_attr &attr, const engine &e)
            : mkldnn::primitive_desc(&desc.data, &attr, e, nullptr) {}

        /// Queries source memory descriptor.
        memory::desc src_desc() const {
            return query_md(query::src_md, 0);
        }

        /// Queries destination memory descriptor.
        memory::desc dst_desc() const {
            return query_md(query::dst_md, 0);
        }

        /// Queries workspace memory descriptor.
        memory::desc workspace_desc() const {
            return query_md(query::workspace_md, 0);
        }
    };

    resize_bilinear_forward() = default;

    resize_bilinear_forward(const primitive_desc &pd): patch_primitive(pd) {}
};

/// @}

/// @} C++ API

} // namespace mkldnn

#endif
