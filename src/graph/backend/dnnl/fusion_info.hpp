/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef GRAPH_BACKEND_DNNL_FUSION_INFO_HPP
#define GRAPH_BACKEND_DNNL_FUSION_INFO_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/graph_attr.hpp"
#include "graph/interface/op.hpp"
#include "graph/interface/value.hpp"
#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/utils.hpp"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
#define VCHECK_FUSION_INFO(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, fusion_info, (cond), status, msg, \
            ##__VA_ARGS__);

enum class attr_type_t { QK, VS };

// This class is used to represent an op's fusion information, such as the post
// ops, the zero points or scales.
class fusion_info_t {
    using op_ptr = std::shared_ptr<op_t>;

    // A meta op is used to represent the op that has been fused away, like the
    // scales op or post-ops op.
    class meta_op_t {
    public:
        // for scales and zps
        meta_op_t(const op_ptr &op) : op_(op) {}
        // for post-eltwise
        meta_op_t(const op_ptr &op, float scale) : op_(op), scale_(scale) {}
        // for post-sum and post_binary
        meta_op_t(const op_ptr &op,
                const std::vector<size_t> &extra_input_indices, float scale,
                int32_t zp)
            : op_(op)
            , scale_(scale)
            , zp_(zp)
            , unfused_input_indices_(extra_input_indices) {}
        // for post-conv
        meta_op_t(const op_ptr &op,
                const std::vector<size_t> &extra_input_indices)
            : op_(op), unfused_input_indices_(extra_input_indices) {}

        bool operator==(const meta_op_t &other) const {
            return *op_ == *other.op_ && scale_ == other.scale_
                    && zp_ == other.zp_
                    && unfused_input_indices_ == other.unfused_input_indices_
                    && is_post_sum_ == other.is_post_sum_;
        }

        float get_scale() const { return scale_; }
        int32_t get_zp() const { return zp_; }
        const std::vector<size_t> &get_unfused_input_indices() const {
            return unfused_input_indices_;
        }

        const op_t *get_op() const { return op_.get(); }

        bool is_post_sum() const {
            return op_->get_kind() == op_kind::_binary && is_post_sum_;
        }

        bool is_post_binary() const {
            return op_->get_kind() == op_kind::_binary && !is_post_sum_;
        }

        void set_post_sum() { is_post_sum_ = true; }

    private:
        std::shared_ptr<op_t> op_;
        // used to represent post-eltwise and post-sum's scale
        float scale_ = 1.0f;
        // used to represent post-sum's zp
        int32_t zp_ = 0;
        // used to represent post-sum, post-binary and post-convolution's
        // unfused input index
        std::vector<size_t> unfused_input_indices_;
        // used to identify post-sum
        bool is_post_sum_ = false;
    };

public:
    friend dnnl::primitive_attr make_dnnl_primitive_attr(
            const op_ptr &op, const fusion_info_t &fusion_info);

    friend dnnl::primitive_attr make_dnnl_sdpa_primitive_attr(
            const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info,
            const attr_type_t attr_type);

    fusion_info_t() = default;

    bool operator==(const fusion_info_t &other) const {
        // input_zps_
        if (input_zps_.size() != other.input_zps_.size()) return false;
        for (const auto &pair : input_zps_) {
            size_t index = pair.first;
            const std::shared_ptr<meta_op_t> &meta_op = pair.second;
            auto it = other.input_zps_.find(index);
            if (it == other.input_zps_.end() || !(*meta_op == *it->second)) {
                return false;
            }
        }

        // output_zps_
        if ((output_zps_ == nullptr) != (other.output_zps_ == nullptr))
            return false;
        if (output_zps_ && other.output_zps_
                && !(*output_zps_ == *other.output_zps_))
            return false;

        // input_scales_
        if (input_scales_.size() != other.input_scales_.size()) return false;
        for (const auto &pair : input_scales_) {
            size_t index = pair.first;
            const std::shared_ptr<meta_op_t> &meta_op = pair.second;
            auto it = other.input_scales_.find(index);
            if (it == other.input_scales_.end() || !(*meta_op == *it->second)) {
                return false;
            }
        }

        // dst_scales_
        if ((dst_scales_ == nullptr) != (other.dst_scales_ == nullptr))
            return false;
        if (dst_scales_ && other.dst_scales_
                && !(*dst_scales_ == *other.dst_scales_))
            return false;

        // dropout_
        if ((dropout_ == nullptr) != (other.dropout_ == nullptr)) return false;
        if (dropout_ && other.dropout_ && !(*dropout_ == *other.dropout_))
            return false;

        // post_ops_
        if (post_ops_.size() != other.post_ops_.size()) return false;
        for (size_t i = 0; i < post_ops_.size(); ++i) {
            if (!(*post_ops_[i] == *other.post_ops_[i])) { return false; }
        }

        return true;
    }

    bool operator!=(const fusion_info_t &other) const {
        return !(*this == other);
    }

    // used to modify the fused arg scales, like modifying it's axis after
    // inserting reshape op
    op_t *get_mutable_scales(bool is_input, size_t index) {
        if (is_input) {
            if (input_scales_.find(index) == input_scales_.end())
                return nullptr;
            return const_cast<op_t *>(input_scales_.at(index)->get_op());
        } else {
            VCHECK_FUSION_INFO(
                    index == 0, nullptr, "index for output scales must be 0");
            if (!dst_scales_) return nullptr;
            return const_cast<op_t *>(dst_scales_->get_op());
        }
    }

    void set_zero_points(const op_ptr &op, bool is_input, size_t index) {
        auto fused_zps = std::make_shared<meta_op_t>(op);
        if (is_input) {
            input_zps_[index] = std::move(fused_zps);
        } else {
            output_zps_ = std::move(fused_zps);
        }
    }

    void set_runtime_scales(const op_ptr &op, bool is_input, size_t index) {
        auto fused_scales = std::make_shared<meta_op_t>(op);
        if (is_input) {
            input_scales_[index] = std::move(fused_scales);
        } else {
            dst_scales_ = std::move(fused_scales);
        }
    }

    void set_dropout(const op_ptr &op) {
        dropout_ = std::make_shared<meta_op_t>(op);
    }

    // used to modify the fused zps, like modifying it's axis after inserting
    // reshape op
    op_t *get_mutable_zero_points(bool is_input, size_t index) const {
        if (is_input) {
            if (input_zps_.find(index) == input_zps_.end()) return nullptr;
            return const_cast<op_t *>(input_zps_.at(index)->get_op());
        } else {
            VCHECK_FUSION_INFO(
                    index == 0, nullptr, "index for output zps must be 0");
            if (!output_zps_) return nullptr;
            return const_cast<op_t *>(output_zps_->get_op());
        }
    }

    void append_post_eltwise(const op_ptr &op, float scale = 1.0f) {
        post_ops_.emplace_back(std::make_shared<meta_op_t>(op, scale));
    }

    // the extra input means the unfused input that has been added to the fused
    // op, like the following case, we fuse a binary mul into the conv, the src1
    // of mul op is unfused, and it becomes the 3rd input of conv. So the extra
    // input indices should be this input's index 2.
    //
    //   src   wei            src  wei  src1
    //     \   /                 \  |   /
    //      conv   src1   ->       conv
    //         \   /                 |
    //          mul
    //           |
    void append_post_binary(const op_ptr &op,
            const std::vector<size_t> &extra_input_indices, float scale = 1.0f,
            int32_t zp = 0) {
        post_ops_.emplace_back(std::make_shared<meta_op_t>(
                op, extra_input_indices, scale, zp));
    }

    // the meaning of extra input is same as that in append_post_binary function
    void append_post_dw_conv(
            const op_ptr &op, const std::vector<size_t> &extra_input_indices) {
        post_ops_.emplace_back(
                std::make_shared<meta_op_t>(op, extra_input_indices));
    }

    const std::vector<std::shared_ptr<meta_op_t>> &get_post_ops() const {
        return post_ops_;
    }

    bool has_post_dw_conv() const {
        auto pos = std::find_if(post_ops_.begin(), post_ops_.end(),
                [](const std::shared_ptr<meta_op_t> &mop) {
            return mop->get_op()->get_kind() == op_kind::_convolution;
        });
        return pos != post_ops_.end();
    }

    const std::shared_ptr<meta_op_t> &get_post_dw_conv() const {
        auto pos = std::find_if(post_ops_.begin(), post_ops_.end(),
                [](const std::shared_ptr<meta_op_t> &mop) {
            return mop->get_op()->get_kind() == op_kind::_convolution;
        });

        VCHECK_FUSION_INFO(
                pos != post_ops_.end(), *pos, "cannot find post dw_conv");
        return *pos;
    }

    bool has_post_binary() const {
        auto pos = std::find_if(post_ops_.begin(), post_ops_.end(),
                [](const std::shared_ptr<meta_op_t> &mop) {
            return mop->is_post_binary();
        });
        return pos != post_ops_.end();
    }

    bool with_runtime_zero_points(bool is_input, size_t index) const {
        if (is_input) {
            if (input_zps_.find(index) == input_zps_.end()) return false;
            const op_t *zp_op
                    = const_cast<op_t *>(input_zps_.at(index)->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_zps)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_zps);
            } else {
                return false;
            }
        } else {
            if (!output_zps_) return false;
            const op_t *zp_op = const_cast<op_t *>(output_zps_->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_zps)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_zps);
            } else {
                return false;
            }
        }
    }

    bool with_runtime_scales(bool is_input, size_t index) const {
        if (is_input) {
            if (input_scales_.find(index) == input_scales_.end()) return false;
            const op_t *zp_op
                    = const_cast<op_t *>(input_scales_.at(index)->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_scales)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_scales);
            } else {
                return false;
            }
        } else {
            if (!dst_scales_) return false;
            const op_t *zp_op = const_cast<op_t *>(dst_scales_->get_op());
            if (zp_op->has_attr(op_attr::with_runtime_scales)) {
                return zp_op->get_attr<bool>(op_attr::with_runtime_scales);
            } else {
                return false;
            }
        }
    }

    bool with_dropout() const { return dropout_ != nullptr; }

private:
    std::unordered_map<size_t, std::shared_ptr<meta_op_t>> input_zps_;
    std::shared_ptr<meta_op_t> output_zps_;
    std::unordered_map<size_t, std::shared_ptr<meta_op_t>> input_scales_;
    std::shared_ptr<meta_op_t> dst_scales_;
    std::shared_ptr<meta_op_t> dropout_;
    std::vector<std::shared_ptr<meta_op_t>> post_ops_;
};

// Compute the quantization mask from any quantization-related op. Handles both
// op_attr::mask and legacy op_attr::qtype/axis for backward compatibility.
inline int get_quant_mask(const op_t *quant_op) {
    // Primary path: mask attribute is set.
    if (quant_op->has_attr(op_attr::mask))
        return static_cast<int>(quant_op->get_attr<int64_t>(op_attr::mask));

    // Fallback: derive mask from qtype/axis for backward compatibility.
    // If qtype is not set, it's per-tensor by default.
    if (!quant_op->has_attr(op_attr::qtype)) return 0;

    const auto &qtype = quant_op->get_attr<std::string>(op_attr::qtype);
    if (qtype == "per_tensor") return 0;

    if (qtype == "per_channel") {
        int64_t axis = quant_op->has_attr(op_attr::axis)
                ? quant_op->get_attr<int64_t>(op_attr::axis)
                : 1;
        if (axis < 0) axis += quant_op->get_input_logical_tensor(0).ndims;
        return 1LL << axis;
    }

    if (qtype == "per_group") {
        // For per_group, mask bit i is set when scale_dims[i] > 1.
        // Dynamic ops have the scale tensor as input 1.
        if (quant_op->num_inputs() >= 2) {
            const auto &data_lt = quant_op->get_input_logical_tensor(0);
            const auto &scale_lt = quant_op->get_input_logical_tensor(1);
            const int ndims = scale_lt.ndims;
            if (ndims > 0) {
                int mask = 0;
                for (int i = 0; i < ndims; ++i) {
                    // If both the scale dimension and data dimension are 1, we
                    // also set the mask bit for the dimension. For example,
                    // data (1, 16, 1, 1024) and scale (1, 16, 1, 32) would
                    // result in a mask 15.
                    if (scale_lt.dims[i] > 1 || data_lt.dims[i] == 1)
                        mask |= (1 << i);
                }
                // Primitive API requires mask to cover last two dims
                // when groups are specified.
                if (ndims >= 2) {
                    mask |= (1 << (ndims - 1)) | (1 << (ndims - 2));
                }
                return mask;
            }
        }
        if (utils::has_group_shape(quant_op)) {
            const auto &group_shape = quant_op->get_attr<std::vector<int64_t>>(
                    op_attr::group_shape);
            int mask = 0;
            const int ndims = static_cast<int>(group_shape.size());
            for (int i = 0; i < ndims; ++i) {
                if (group_shape[i] > 1) mask |= (1 << i);
            }
            if (ndims >= 2) { mask |= (1 << (ndims - 1)) | (1 << (ndims - 2)); }
            return mask;
        }
        return 1;
    }

    return 0;
}

// This function is used to make a dnnl::primitive_attr from the fusion info.
// Note that the op and fusion_info arguments must be matched since a fusion
// info make sense only when it belongs to a specific op.
dnnl::primitive_attr make_dnnl_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info);

// This function is used to make a dnnl::primitive_attr(QK and VS) for sdpa.
// The attr creation is similar to matmul attr, but fusion info of sdpa has
// different semantics, so we need to handle it differently.
dnnl::primitive_attr make_dnnl_sdpa_primitive_attr(
        const std::shared_ptr<op_t> &op, const fusion_info_t &fusion_info,
        const attr_type_t attr_type);

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
