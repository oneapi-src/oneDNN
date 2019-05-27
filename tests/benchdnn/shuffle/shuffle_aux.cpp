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

#include <stdlib.h>
#include <assert.h>
#include "shuffle/shuffle.hpp"

namespace shuffle {

void prb2str(const prb_t *p, char *buffer, bool canonical) {
    char dir_str[32] = "", dt_str[32] = "", tag_str[32] = "",
         group_str[32] = "", axis_str[32] = "", dims_str[max_desc_len] = "";

    if (p->dir != FWD_D)
        snprintf(dir_str, sizeof(dir_str), "--dir=%s ", dir2str(p->dir));
    if (p->dt != mkldnn_f32)
        snprintf(dt_str, sizeof(dt_str), "--dt=%s ", dt2str(p->dt));
    if (p->tag != mkldnn_nchw)
        snprintf(tag_str, sizeof(tag_str), "--tag=%s ", tag2str(p->tag));
    if (p->group != 1)
        snprintf(group_str, sizeof(group_str), "--group=" IFMT " ", p->group);
    if (p->axis != 1)
        snprintf(axis_str, sizeof(axis_str), "--axis=%d ", p->axis);
    dims2str(p->dims, dims_str);

    snprintf(buffer, max_prb_len, "%s%s%s%s%s%s", dir_str, dt_str, tag_str,
            group_str, axis_str, dims_str);
}

}
