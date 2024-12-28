/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef UTILS_COMPARE_HPP
#define UTILS_COMPARE_HPP

#include <functional>

#include "dnn_types.hpp"
#include "dnnl_memory.hpp"

namespace compare {

bool compare_extreme_values(float a, float b);

struct compare_t {
    struct driver_check_func_args_t {
        driver_check_func_args_t(const dnn_mem_t &exp_mem,
                const dnn_mem_t &got_f32, const int64_t i,
                const dnnl_data_type_t data_type, const float trh);

        const dnnl_data_type_t dt = dnnl_data_type_undef;
        const int64_t idx = 0;
        const float exp_f32 = 0.f;
        const float exp = 0.f;
        const float got = 0.f;
        const float diff = 0.f;
        const float rel_diff = 0.f;
        const int64_t ulps_diff = 0;
        const float trh = 0.f;
    };

    compare_t() = default;

    void set_norm_validation_mode(bool un) { use_norm_ = un; }
    void set_threshold(float trh) { trh_ = trh; }
    void set_ulps_threshold(int64_t utrh) { ulps_trh_ = utrh; }
    void set_zero_trust_percent(float ztp) { zero_trust_percent_ = ztp; }
    void set_data_kind(data_kind_t dk) { kind_ = dk; }
    void set_op_output_has_nans(bool ohn) { op_output_has_nans_ = ohn; }
    void set_has_eltwise_post_op(bool hepo) { has_eltwise_post_op_ = hepo; }
    void set_has_prim_ref(bool hpr) { has_prim_ref_ = hpr; }

    // @param idx The index of compared element. Helps to obtain any element
    //     from any reference memory since it's in abx format.
    // @param got The value of library memory for index `idx`. Can't be obtained
    //     by `idx` directly since could have different memory formats.
    // @param diff The absolute difference between expected and got values.
    // @returns true if checks pass and false otherwise.
    using driver_check_func_t
            = std::function<bool(const driver_check_func_args_t &)>;
    void set_driver_check_function(const driver_check_func_t &dcf) {
        driver_check_func_ = dcf;
    }

    int compare(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
            const attr_t &attr, res_t *res) const;

private:
    // Switch between point-to-point and norm comparison.
    bool use_norm_ = false;
    // Threshold for a point-to-point comparison.
    float trh_ = 0.f;
    // The percent value of zeros allowed in the output.
    float default_zero_trust_percent_ = 30.f;
    float zero_trust_percent_ = default_zero_trust_percent_;
    // Kind specifies what tensor is checked. Not printed if default one.
    data_kind_t kind_ = DAT_TOTAL;
    // Driver-specific function that adds additional criteria for a test case to
    // pass.
    driver_check_func_t driver_check_func_;
    // Some operators may legally return NaNs. This member allows to work around
    // issues involving comparison with NaNs in the op output if additional
    // post-ops are involved.
    bool op_output_has_nans_ = false;
    // Graph driver can't use attributes as a criterion for certain checks but
    // they may be relevant in specific cases. This is a hint to utilize
    // additional checks despite attributes are not set.
    bool has_eltwise_post_op_ = false;
    // `fast_ref` enables optimized primitive to be used instead of
    // reference. In this case `ref_mem` should also be reordered to a plain
    // layout for proper comparison.
    bool has_prim_ref_ = false;
    // ULP, or unit in the last place, is a distance metric between exp and got
    // values in "the number of bits". Thus, some 8-bit values (either integer
    // or floating-point) 0xFE and 0xFF are 1 ulp apart, because
    // `llabs(0x11111110 - 0x11111111) = 1`. This metric is relative to the
    // given numbers because in most cases it would end up comparing
    // mantissa bits of two floating-point values, while the absolute difference
    // relies on actual values, and with higher exponent mask a difference in 1
    // ulp may be expressed by bigger numbers.
    // However, this doesn't suit much for comparing the numbers close to 0 as
    // their exponent bits will be different, and this difference will be
    // expressed with values of (1 << digits_dt) order, e.g, 256*N for bfloat16.
    // So far, this criterion will be used as an additional one to mark a point
    // passed, mostly for lower precision data types.
    // Using a value of 7, translating into a diversion in the last three bits,
    // should be fine.
    // TODO: consider an opportunity to switch to this metric instead of
    // rel_diff verification.
    int64_t ulps_trh_ = 0;

    // Internal validation methods under `compare` interface.
    int compare_p2p(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
            const attr_t &attr, res_t *res) const;
    int compare_norm(const dnn_mem_t &exp_mem, const dnn_mem_t &got_mem,
            const attr_t &attr, res_t *res) const;

    std::string get_kind_str() const {
        std::string kind_str;
        if (kind_ == DAT_TOTAL) return kind_str;

        kind_str = std::string("[") + std::string(data_kind2str(kind_))
                + std::string("]");
        return kind_str;
    }
};

} // namespace compare

#endif
