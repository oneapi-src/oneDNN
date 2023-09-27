/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef BENCHDNN_OP_SETTING_HPP
#define BENCHDNN_OP_SETTING_HPP

#include "deserialize.hpp"

#include "binary/binary.hpp"
#include "bnorm/bnorm.hpp"
#include "concat/concat.hpp"
#include "conv/conv.hpp"
#include "custom_driver.hpp"
#include "deconv/deconv.hpp"
#include "eltwise/eltwise.hpp"
#include "lnorm/lnorm.hpp"
#include "matmul/matmul.hpp"
#include "pool/pool.hpp"
#include "prelu/prelu.hpp"
#include "reduction/reduction.hpp"
#include "reorder/reorder.hpp"
#include "resampling/resampling.hpp"
#include "softmax/softmax.hpp"

namespace graph {

#define DECLARE_GET_SETTING(driver) \
    namespace driver { \
    ::driver::settings_t get_setting(const deserialized_op &base_op_ref, \
            const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res); \
    }

DECLARE_GET_SETTING(binary);
DECLARE_GET_SETTING(bnorm);
DECLARE_GET_SETTING(concat);
DECLARE_GET_SETTING(conv);
DECLARE_GET_SETTING(custom);
DECLARE_GET_SETTING(deconv);
DECLARE_GET_SETTING(eltwise);
DECLARE_GET_SETTING(lnorm);
DECLARE_GET_SETTING(matmul);
DECLARE_GET_SETTING(pool);
DECLARE_GET_SETTING(prelu);
DECLARE_GET_SETTING(reduction);
DECLARE_GET_SETTING(reorder);
DECLARE_GET_SETTING(resampling);
DECLARE_GET_SETTING(softmax);

template <bool B>
using req = typename std::enable_if<B, bool>::type;

#define DECLARE_TEMPLATE_GET_SETTING(driver) \
    template <typename setting_t, \
            req<std::is_same<setting_t, ::driver::settings_t>::value> = true> \
    setting_t get_setting(const deserialized_op &base_op_ref, \
            const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) { \
        deserialized_op base_op = base_op_ref; \
        for (size_t i = 0; i < base_op.in_lts_.size(); i++) { \
            if (base_op.in_lts_[i].shape_.size() == 0) \
                base_op.in_lts_[i].shape_.emplace_back(1); \
            if (base_op.in_lts_[i].stride_.size() == 0) \
                base_op.in_lts_[i].stride_.emplace_back(1); \
        } \
        for (size_t i = 0; i < base_op.out_lts_.size(); i++) { \
            if (base_op.out_lts_[i].shape_.size() == 0) \
                base_op.out_lts_[i].shape_.emplace_back(1); \
            if (base_op.out_lts_[i].stride_.size() == 0) \
                base_op.out_lts_[i].stride_.emplace_back(1); \
        } \
        return driver::get_setting(base_op, rewrite_lt_ids, res); \
    }

// template to generate driver settings
DECLARE_TEMPLATE_GET_SETTING(binary);
DECLARE_TEMPLATE_GET_SETTING(bnorm);
DECLARE_TEMPLATE_GET_SETTING(concat);
DECLARE_TEMPLATE_GET_SETTING(conv);
DECLARE_TEMPLATE_GET_SETTING(custom);
DECLARE_TEMPLATE_GET_SETTING(deconv);
DECLARE_TEMPLATE_GET_SETTING(eltwise);
DECLARE_TEMPLATE_GET_SETTING(lnorm);
DECLARE_TEMPLATE_GET_SETTING(matmul);
DECLARE_TEMPLATE_GET_SETTING(pool);
DECLARE_TEMPLATE_GET_SETTING(prelu);
DECLARE_TEMPLATE_GET_SETTING(reduction);
DECLARE_TEMPLATE_GET_SETTING(reorder);
DECLARE_TEMPLATE_GET_SETTING(resampling);
DECLARE_TEMPLATE_GET_SETTING(softmax);

namespace eltwise {

bool get_flag_use_dst_for_bwd_compute(const deserialized_op &base_op_ref);

const std::unordered_map<std::string, ::eltwise::alg_t> &get_eltwise_kind_map();

} // namespace eltwise

} // namespace graph

#endif
