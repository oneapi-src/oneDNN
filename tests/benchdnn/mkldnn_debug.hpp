/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED

#ifndef MKLDNN_DEBUG_HPP
#define MKLDNN_DEBUG_HPP

#include "mkldnn.h"

mkldnn_data_type_t str2dt(const char *str);
mkldnn_format_tag_t str2fmt_tag(const char *str);

/* status */
const char *status2str(mkldnn_status_t status);

/* data type */
const char *dt2str(mkldnn_data_type_t dt);

/* format */
const char *fmt_tag2str(mkldnn_format_tag_t tag);

#endif
