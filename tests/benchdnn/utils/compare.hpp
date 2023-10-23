/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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
        const float trh = 0.f;
    };

    compare_t() = default;

    void set_norm_validation_mode(bool un) { use_norm_ = un; }
    void set_threshold(float trh) { trh_ = trh; }
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
