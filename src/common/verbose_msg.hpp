/*******************************************************************************
* Copyright 2023 Intel Corporation
* Copyright 2023 Arm Ltd. and affiliates
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

#ifndef COMMON_VERBOSE_MSG_HPP
#define COMMON_VERBOSE_MSG_HPP

// log type strings
#define VERBOSE_error "error"
#define VERBOSE_create "create"
#define VERBOSE_exec "exec"

// log subtypes strings
#define VERBOSE_check ":check"
#define VERBOSE_dispatch ":dispatch"
#define VERBOSE_debug ":debug"
#define VERBOSE_profile ""
#define VERBOSE_external ":external"

// verbose messages
#define VERBOSE_PROFILING_UNSUPPORTED "profiling capabilities are not supported"

#define VERBOSE_NULL_ARG "one of the mandatory arguments is nullptr"
#define VERBOSE_BAD_ENGINE_KIND "bad engine kind"
#define VERBOSE_BAD_ALGORITHM "bad algorithm"
#define VERBOSE_BAD_PROPKIND "bad propagation kind"
#define VERBOSE_BAD_AXIS "bad axis"
#define VERBOSE_BAD_FLAGS "bad flags"
#define VERBOSE_BAD_PARAM "bad param %s"
#define VERBOSE_RUNTIMEDIM_UNSUPPORTED "runtime dimension is not supported"
#define VERBOSE_RUNTIMEDIM_INCONSISTENT \
    "runtime dimension %d is inconsistent across tensors"

#define VERBOSE_INVALID_BROADCAST "invalid broadcast semantic on %s:%d"
#define VERBOSE_INVALID_DATATYPE "invalid datatype for %s"

#define VERBOSE_EMPTY_TENSOR "tensor %s has no elements"
#define VERBOSE_INCONSISTENT_DIM "dimension %s:%d is inconsistent with %s:%d"
#define VERBOSE_INCONSISTENT_NDIMS \
    "tensors %s and %s have inconsistent number of dimensions"
#define VERBOSE_INCONSISTENT_DT "tensors %s and %s have inconsistent datatypes"
#define VERBOSE_INCONSISTENT_MDS "inconsistent %s and %s mds"
#define VERBOSE_INCONSISTENT_ALPHA_BETA \
    "alpha and beta parameters are not properly set"
#define VERBOSE_INCONSISTENT_PRB "problem is not mathematically consistent"
#define VERBOSE_BAD_NDIMS "%s has a bad number of dimensions %d"
#define VERBOSE_BAD_DIM "bad dimension %s:%d"

#define VERBOSE_UNSUPPORTED_ISA "unsupported isa"
#define VERBOSE_UNSUPPORTED_DT "unsupported datatype"
#define VERBOSE_UNSUPPORTED_MD_FLAG "unsupported %s md flags"
#define VERBOSE_UNSUPPORTED_ATTR "unsupported attribute"
#define VERBOSE_UNSUPPORTED_FPMATH_MODE "unsupported fpmath mode"
#define VERBOSE_UNSUPPORTED_POSTOP "unsupported post-ops"
#define VERBOSE_UNSUPPORTED_SCALES_CFG "unsupported scales configuration"
#define VERBOSE_UNSUPPORTED_ZP_CFG "unsupported zero-point configuration"
#define VERBOSE_UNSUPPORTED_BIAS_CFG "unsupported bias configuration"
#define VERBOSE_UNSUPPORTED_DT_CFG "unsupported datatype combination"

#define VERBOSE_UNSUPPORTED_TAG "unsupported format tag"
#define VERBOSE_UNSUPPORTED_TAG_S "unsupported format tag for %s"

#define VERBOSE_ISA_DT_MISMATCH \
    "datatype configuration not supported on this isa"
#define VERBOSE_BLOCKING_FAIL "blocking heuristic failed"
#define VERBOSE_SMALL_SHAPES "small shapes fall back"
#define VERBOSE_NONTRIVIAL_STRIDE "only trivial strides are supported"

#endif
