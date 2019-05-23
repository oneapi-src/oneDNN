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

#ifndef PATCH_MKLDNN_H
#define PATCH_MKLDNN_H

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include "patch_mkldnn_types.hpp"
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup c_api_resize_bilinear resize_bilinear
 * A primitive to perform resize_bilinear.
 * 
 * @{ */

/** Initializes a resize_bilinear descriptor @p bilinear_desc for forward propagation using
 * @p prop_kind (possible values are #mkldnn_forward_training or
 * #mkldnn_forward_inference), @p alg_kind, memory descriptors, and resize_bilinear
 * parameters in spatial domain: @p strides, @p kernel sizes, @p padding_l, @p
 * padding_r, and @p padding_kind.
 *
 * @note if @p padding_r is @c NULL, the padding is supposed to be symmetric
 *
 * @todo clarify! */
mkldnn_status_t MKLDNN_API mkldnn_resize_bilinear_forward_desc_init(
        mkldnn_resize_bilinear_desc_t *bilinear_desc, 
        mkldnn_prop_kind_t prop_kind, /*mkldnn_alg_kind_t alg_kind, */
        const mkldnn_memory_desc_t *src_desc, const mkldnn_memory_desc_t *dst_desc, 
        const int *align_corners);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
