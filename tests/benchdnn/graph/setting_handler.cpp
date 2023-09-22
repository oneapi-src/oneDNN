/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "setting_handler.hpp"

namespace graph {

#define DNN_GRAPH_CHECK_SETTINGS(ret, res) \
    if (!(ret)) { \
        BENCHDNN_PRINT(0, "error settings: [%s:%d] \n", __PRETTY_FUNCTION__, \
                __LINE__); \
        (res)->state = INVALID_ARGUMENTS; \
    }

dnnl_data_type_t convert_dt(const dnnl::graph::logical_tensor::data_type dt) {
    using graph_dt = dnnl::graph::logical_tensor::data_type;

    switch (dt) {
        case graph_dt::f16: return dnnl_f16;
        case graph_dt::bf16: return dnnl_bf16;
        case graph_dt::f32: return dnnl_f32;
        case graph_dt::s32: return dnnl_s32;
        case graph_dt::s8: return dnnl_s8;
        case graph_dt::u8: return dnnl_u8;
        // use u8 instead of boolean in the reference path
        // dnn_graph_mem_t will use the data type from the logical tensor and the u8 data handle
        case graph_dt::boolean: return dnnl_u8;
        case graph_dt::undef:
        default: return dnnl_data_type_undef;
    }
}

logical_tensor::data_type get_data_type(const std::string &data_type) {
    if (data_type == "f32") {
        return logical_tensor::data_type::f32;
    } else if (data_type == "f16") {
        return logical_tensor::data_type::f16;
    } else if (data_type == "s8") {
        return logical_tensor::data_type::s8;
    } else if (data_type == "u8") {
        return logical_tensor::data_type::u8;
    } else if (data_type == "bf16") {
        return logical_tensor::data_type::bf16;
    } else if (data_type == "s32") {
        return logical_tensor::data_type::s32;
    } else {
        return logical_tensor::data_type::undef;
    }
}

void assign_stride_padding_val(bool has_h, bool has_d, int64_t &w, int64_t &h,
        int64_t &d, const std::vector<int64_t> &val_, int64_t default_val) {
    if (has_d) { // 3d tensor, attr input is DHW
        d = val_[0];
        h = val_[1];
        w = val_[2];
    } else if (has_h) { // 2d tensor
        d = default_val;
        h = val_[0];
        w = val_[1];
    } else { // 1d tensor
        d = default_val;
        h = default_val;
        w = val_[0];
    }
};

void assign_dilation_val(bool has_h, bool has_d, int64_t &w, int64_t &h,
        int64_t &d, const std::vector<int64_t> &val_, int64_t default_val) {
    if (has_d) { // 3d tensor, attr input is DHW
        d = val_[0] - 1;
        h = val_[1] - 1;
        w = val_[2] - 1;
    } else if (has_h) { // 2d tensor
        d = default_val;
        h = val_[0] - 1;
        w = val_[1] - 1;
    } else { // 1d tensor
        d = default_val;
        h = default_val;
        w = val_[0] - 1;
    }
};

void assign_shape_val(int64_t &c, int64_t &w, int64_t &h, int64_t &d,
        const std::vector<int64_t> &ncx_shape) {
    auto ndims = ncx_shape.size();
    bool has_w = ndims > 2;
    bool has_h = ndims > 3;
    bool has_d = ndims > 4;
    // NCDHW
    c = ncx_shape[1];
    w = has_w ? ncx_shape[ndims - 1] : 1;
    h = has_h ? ncx_shape[ndims - 2] : 1;
    d = has_d ? ncx_shape[2] : 1;
};

bool get_driver_tag_by_idx(const deserialized_op &base_op_ref, std::string &tag,
        int idx = 0, bool from_output = false) {
    logical_tensor::dims strides = from_output
            ? base_op_ref.out_lts_[idx].stride_
            : base_op_ref.in_lts_[idx].stride_;
    if (base_op_ref.has_NXC_format()) {
        // convert the strides to data_format = NCX
        change_format_to_ncx(strides);
    }
    tag = strides2memory_tag(strides.size(), strides, true);
    return true;
}

bool get_driver_tag(const deserialized_op &base_op_ref, std::string &tag,
        bool from_output = false) {
    return get_driver_tag_by_idx(base_op_ref, tag, 0, from_output);
}

bool get_driver_stag_and_dtag(const deserialized_op &base_op_ref,
        std::string &stag, std::string &dtag, bool from_output = false) {
    bool ret = get_driver_tag(base_op_ref, stag, from_output);
    dtag = stag;
    return ret;
}

bool get_driver_axis(const deserialized_op &base_op_ref, int &axis) {
    int64_t val = 0;
    base_op_ref.get_attr_s64(val, "axis");
    axis = val >= 0
            ? val
            : val + static_cast<int>(base_op_ref.in_lts_.front().shape_.size());
    return true;
}

bool get_prb_dims(const deserialized_op &base_op_ref, prb_dims_t &prb_dims) {
    prb_dims.dims = base_op_ref.in_lts_.front().shape_;
    prb_dims.ndims = static_cast<int>(prb_dims.dims.size());
    return true;
}

// extend shape in src to match the ndims
// if the rank in tensor is less than ndims, we need to insert 1
void extend_dims(::graph::deserialized_lt &lt, size_t ndims) {
    size_t nelem = 1;
    for (size_t i = 0; i < lt.shape_.size(); i++) {
        nelem *= lt.shape_[i];
    }
    while (lt.shape_.size() < ndims) {
        lt.shape_.insert(lt.shape_.begin(), 1);
    }
    while (lt.stride_.size() < ndims) {
        lt.stride_.insert(lt.stride_.begin(), nelem);
    }
}

namespace custom {

::custom::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::custom::settings_t op_setting;
    auto opkind = opstr2kind(base_op_ref.kind_);
    switch (opkind) {
        case ::graph::op::kind::Select:
            op_setting.alg = ::custom::alg_t::SELECT;
            break;
        case ::graph::op::kind::StaticTranspose:
            op_setting.alg = ::custom::alg_t::TRANSPOSE;
            base_op_ref.get_attr_s64_vector(op_setting.order, "order");
            break;
        case ::graph::op::kind::StaticReshape:
            op_setting.alg = ::custom::alg_t::RESHAPE;
            break;
        default:
            op_setting.alg = ::custom::alg_t::ALG_UNKNOWN;
            assert(!"unknown alg");
            res->state = res_state_t::INVALID_ARGUMENTS;
            return op_setting;
    }
    for (size_t i = 0; i < base_op_ref.in_lts_.size(); i++) {
        auto arg = get_prim_arg_name_from_graph_op_input_offset(
                opkind, static_cast<int>(i));
        auto dim = base_op_ref.in_lts_[i].shape_;
        auto dt = convert_dt(base_op_ref.in_lts_[i].get_data_type());
        auto tag = strides2memory_tag(base_op_ref.in_lts_[i].stride_.size(),
                base_op_ref.in_lts_[i].stride_, false);

        // 0-dim means scalar input in graph, extend to 1-dim to match behavior.
        if (dim.empty()) {
            dim.push_back(1);
            tag = "a";
        }
        op_setting.arg_mds_[arg] = ::std::make_tuple(tag, dim, dt);
    }
    for (size_t i = 0; i < base_op_ref.out_lts_.size(); i++) {
        auto arg = get_prim_arg_name_from_graph_op_output_offset(
                opkind, static_cast<int>(i));
        auto dim = base_op_ref.out_lts_[i].shape_;
        auto dt = convert_dt(base_op_ref.out_lts_[i].get_data_type());
        auto tag = strides2memory_tag(base_op_ref.out_lts_[i].stride_.size(),
                base_op_ref.out_lts_[i].stride_, false);

        // 0-dim means scalar input in graph, extend to 1-dim to match behavior.
        if (dim.empty()) {
            dim.push_back(1);
            tag = "a";
        }
        op_setting.arg_mds_[arg] = ::std::make_tuple(tag, dim, dt);
    }
    return op_setting;
}

} // namespace custom

namespace binary {
bool get_binary_prb_vdims(
        const deserialized_op &base_op_ref, prb_vdims_t &prb_vdims) {
    // since base_op_ref is a copy from the original
    // it is safe to modify it
    deserialized_op &base_op = const_cast<deserialized_op &>(base_op_ref);

    auto &src0_dims = base_op.in_lts_[0].shape_;
    auto &src1_dims = base_op.in_lts_[1].shape_;
    auto &dst_dims = base_op.out_lts_[0].shape_;
    const auto &ndims = dst_dims.size();
    // use Add to implement BiasAdd, need to align channel dims of src1
    if (base_op_ref.kind_ == "BiasAdd") {
        if (ndims == 1 && src0_dims[0] != src1_dims[0] && src1_dims[0] != 1) {
            return false;
        }
        // src0: [M,N] ---> src1:[1,1] / [M,1] / [1,N]
        else if (ndims == 2) {
            if (src1_dims[0] == 1 || src1_dims[0] == src0_dims[0]) {
                src1_dims.insert(src1_dims.end(), 1);
            } else if (src1_dims[0] == src0_dims[1]) {
                src1_dims.insert(src1_dims.begin(), 1);
            } else {
                return false;
            }
        }
        // src0: [N,X,C] / [N,C,X] ---> src1:[1,1..,C] / [1,C,1..]
        else if (ndims > 2) {
            dims_t src1_dims_tmp(ndims, 1);
            // default NCX
            int64_t channel_idx = 1;
            if (base_op_ref.has_NXC_format()) { channel_idx = ndims - 1; }
            src1_dims_tmp[channel_idx] = src0_dims[channel_idx];
            src1_dims = src1_dims_tmp;

            // convert NXC to NCX
            if (base_op_ref.has_NXC_format()) {
                change_format_to_ncx(src0_dims, src1_dims, dst_dims);
            }
        }
    } else {
        ::graph::extend_dims(base_op.in_lts_[0], ndims);
        ::graph::extend_dims(base_op.in_lts_[1], ndims);
    }

    prb_vdims = prb_vdims_t({src0_dims, src1_dims});
    return true;
}

bool get_binary_sdt_and_ddt(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids,
        ::binary::settings_t &op_setting) {
    auto sdt0 = convert_dt(base_op_ref.in_lts_[0].get_data_type());
    auto sdt1 = convert_dt(base_op_ref.in_lts_[1].get_data_type());
    auto ddt = convert_dt(base_op_ref.out_lts_[0].get_data_type());

    if (rewrite_lt_ids.find(base_op_ref.in_lts_[0].id_) != rewrite_lt_ids.end())
        sdt0 = dnnl_f32;
    if (rewrite_lt_ids.find(base_op_ref.in_lts_[1].id_) != rewrite_lt_ids.end())
        sdt1 = dnnl_f32;
    if (rewrite_lt_ids.find(base_op_ref.out_lts_[0].id_)
            != rewrite_lt_ids.end())
        ddt = dnnl_f32;

    op_setting.sdt = {{sdt0, sdt1}};
    op_setting.ddt.front() = ddt;
    return true;
}

bool get_binary_stag_and_dtag(
        const deserialized_op &base_op_ref, ::binary::settings_t &op_setting) {
    std::string tag;
    get_driver_tag(base_op_ref, tag);
    // src0, src1, dst have same tag.
    op_setting.stag = {{tag, tag}};
    op_setting.dtag.front() = tag;
    return true;
}

bool get_binary_alg(const deserialized_op &base_op_ref, ::binary::alg_t &alg) {
    static const std::unordered_map<std::string, ::binary::alg_t>
            map_kind_to_alg {{"Add", ::binary::alg_t::ADD},
                    {"BiasAdd", ::binary::alg_t::ADD},
                    {"Divide", ::binary::alg_t::DIV},
                    {"Maximum", ::binary::alg_t::MAX},
                    {"Minimum", ::binary::alg_t::MIN},
                    {"Multiply", ::binary::alg_t::MUL},
                    {"Subtract", ::binary::alg_t::SUB}};

    const auto &op_kind = base_op_ref.kind_;
    if (map_kind_to_alg.find(op_kind) == map_kind_to_alg.end()) return false;
    alg = map_kind_to_alg.at(op_kind);

    return true;
}

::binary::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::binary::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            binary::get_binary_prb_vdims(base_op_ref, op_setting.prb_vdims),
            res);

    DNN_GRAPH_CHECK_SETTINGS(binary::get_binary_sdt_and_ddt(
                                     base_op_ref, rewrite_lt_ids, op_setting),
            res);

    DNN_GRAPH_CHECK_SETTINGS(
            binary::get_binary_stag_and_dtag(base_op_ref, op_setting), res);

    DNN_GRAPH_CHECK_SETTINGS(
            binary::get_binary_alg(base_op_ref, op_setting.alg.front()), res);

    return op_setting;
}

} // namespace binary

namespace bnorm {

bool get_bnorm_desc(const deserialized_op &base_op_ref, ::bnorm::desc_t &d) {
    const auto &src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
    d.mb = src_ncx_shape[0];
    d.ndims = static_cast<int>(src_ncx_shape.size());
    assign_shape_val(d.ic, d.iw, d.ih, d.id, src_ncx_shape);
    base_op_ref.get_attr_f32(d.eps, "epsilon");
    return true;
}

bool get_bnorm_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "BatchNormForwardTraining") {
        dir = dir_t::FWD_D;
    } else if (op_kind == "BatchNormInference") {
        dir = dir_t::FWD_I;
    } else if (op_kind == "BatchNormTrainingBackward") {
        if (base_op_ref.out_lts_.size() == 1) {
            dir = dir_t::BWD_D;
        } else if (base_op_ref.out_lts_.size() == 3) {
            dir = dir_t::BWD_DW;
        } else {
            return false;
        }
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
}

bool get_bnorm_dt(const deserialized_op &base_op_ref, dnnl_data_type_t &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    dt = convert_dt(base_op_ref.in_lts_.front().get_data_type());
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        dt = dnnl_f32;
    return true;
}

bool get_bnorm_flag(
        const deserialized_op &base_op_ref, ::bnorm::flags_t &flag) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "BatchNormForwardTraining") {
        if (base_op_ref.in_lts_.size() == 3) {
            flag = ::bnorm::NONE;
        } else if (base_op_ref.in_lts_.size() == 5) {
            flag = ::bnorm::USE_SCALE | ::bnorm::USE_SHIFT;
        } else {
            return false;
        }
    } else if (op_kind == "BatchNormInference") {
        flag = ::bnorm::GLOB_STATS | ::bnorm::USE_SCALE | ::bnorm::USE_SHIFT;
    } else if (op_kind == "BatchNormTrainingBackward") {
        if (base_op_ref.out_lts_.size() == 1) {
            flag = ::bnorm::GLOB_STATS;
        } else if (base_op_ref.out_lts_.size() == 3) {
            flag = ::bnorm::USE_SCALE | ::bnorm::USE_SHIFT;
        } else {
            return false;
        }
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
}

::bnorm::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::bnorm::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            bnorm::get_bnorm_desc(base_op_ref, op_setting.desc), res);
    DNN_GRAPH_CHECK_SETTINGS(
            bnorm::get_bnorm_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(bnorm::get_bnorm_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_tag(base_op_ref, op_setting.tag.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            bnorm::get_bnorm_flag(base_op_ref, op_setting.flags.front()), res);
    return op_setting;
}

} // namespace bnorm

namespace concat {

bool get_concat_prb_vdims(
        const deserialized_op &base_op_ref, prb_vdims_t &prb_vdims) {
    std::vector<dims_t> vdims;
    for (const auto &in : base_op_ref.in_lts_) {
        vdims.push_back(in.shape_);
    }
    prb_vdims = prb_vdims_t(vdims);
    return true;
}

bool get_concat_sdt_and_ddt(const deserialized_op &base_op_ref,
        ::concat::settings_t &op_setting,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    const auto &in_dt = convert_dt(base_op_ref.in_lts_.front().get_data_type());
    dnnl_data_type_t dt = in_dt;
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        dt = dnnl_f32;
    op_setting.sdt.front() = dt;
    op_setting.ddt.front() = dt;
    return true;
}

::concat::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::concat::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            concat::get_concat_prb_vdims(base_op_ref, op_setting.prb_vdims),
            res);

    DNN_GRAPH_CHECK_SETTINGS(concat::get_concat_sdt_and_ddt(
                                     base_op_ref, op_setting, rewrite_lt_ids),
            res);

    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_stag_and_dtag(base_op_ref,
                    op_setting.stag.front().front(), op_setting.dtag.front()),
            res);

    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_axis(base_op_ref, op_setting.axis.front()), res);

    return op_setting;
}

} // namespace concat

namespace conv {

bool get_conv_desc(const deserialized_op &base_op_ref, ::conv::desc_t &d) {
    d.g = 1;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

    std::string data_format {}, weights_format {};
    std::vector<int64_t> pads_begin {}, pads_end {}, strides {}, dilations {};
    int64_t g = 1;

    base_op_ref.get_attr_string(data_format, "data_format");
    base_op_ref.get_attr_string(weights_format, "weights_format");
    base_op_ref.get_attr_s64_vector(pads_begin, "pads_begin");
    base_op_ref.get_attr_s64_vector(pads_end, "pads_end");
    base_op_ref.get_attr_s64_vector(strides, "strides");
    base_op_ref.get_attr_s64_vector(dilations, "dilations");
    base_op_ref.get_attr_s64(g, "groups");

    logical_tensor::dims src_ncx_shape {}, wei_dims {}, dst_ncx_shape {};
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Convolution") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        wei_dims = base_op_ref.in_lts_[1].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, false);
    } else if (op_kind == "ConvolutionBackwardData") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, false);
        wei_dims = base_op_ref.in_lts_[1].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, true);
    } else if (op_kind == "ConvolutionBackwardWeights") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        wei_dims = base_op_ref.out_lts_[0].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(1, true);
    } else {
        assert(!"unexpected op_kind");
        return false;
    }

    d.ndims = static_cast<int>(src_ncx_shape.size());
    d.mb = src_ncx_shape[0];

    assign_shape_val(d.ic, d.iw, d.ih, d.id, src_ncx_shape);
    assign_shape_val(d.oc, d.ow, d.oh, d.od, dst_ncx_shape);

    bool has_h = d.ndims > 3;
    bool has_d = d.ndims > 4;
    if (weights_format == "OIX") {
        // oiw, oihw, oidhw
        d.kw = wei_dims[d.ndims - 1];
        d.kh = has_h ? wei_dims[d.ndims - 2] : 1;
        d.kd = has_d ? wei_dims[2] : 1;
    } else if (weights_format == "XIO") {
        // wio, hwio, dhwio
        d.kw = wei_dims[d.ndims - 3];
        d.kh = has_h ? wei_dims[d.ndims - 4] : 1;
        d.kd = has_d ? wei_dims[0] : 1;
    } else {
        return FAIL;
    }

    assign_stride_padding_val(has_h, has_d, d.sw, d.sh, d.sd, strides, 1);
    assign_dilation_val(has_h, has_d, d.dw, d.dh, d.dd, dilations, 0);
    assign_stride_padding_val(has_h, has_d, d.pw, d.ph, d.pd, pads_begin, 0);
    assign_stride_padding_val(
            has_h, has_d, d.pw_r, d.ph_r, d.pd_r, pads_end, 0);

    if (g > 1) { // has group
        d.g = g;
        d.has_groups = true;
    }

    if (d.has_groups && d.g <= 0) return false;
    if (d.ic == 0 || d.oc == 0) return false;
    if (d.sd <= 0 || d.sh <= 0 || d.sw <= 0) return false;

    return true;
}

bool get_conv_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Convolution") {
        dir = base_op_ref.in_lts_.size() > 2 ? dir_t::FWD_B : dir_t::FWD_I;
    } else if (op_kind == "ConvolutionBackwardData") {
        dir = dir_t::BWD_D;
    } else if (op_kind == "ConvolutionBackwardWeights") {
        dir = dir_t::BWD_W;
    } else {
        return false;
    }
    return true;
}

bool get_conv_dt(const deserialized_op &base_op_ref,
        std::vector<dnnl_data_type_t> &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    std::string src_dt {}, wei_dt {}, dst_dt {};
    auto in_lt0_dt = base_op_ref.in_lts_[0].data_type_;
    auto in_lt1_dt = base_op_ref.in_lts_[1].data_type_;
    auto out_lt_dt = base_op_ref.out_lts_[0].data_type_;

    if (rewrite_lt_ids.find(base_op_ref.in_lts_[0].id_) != rewrite_lt_ids.end())
        in_lt0_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.in_lts_[1].id_) != rewrite_lt_ids.end())
        in_lt1_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.out_lts_[0].id_)
            != rewrite_lt_ids.end())
        out_lt_dt = "f32";

    const auto &op_kind = base_op_ref.kind_;

    if (op_kind == "Convolution") {
        src_dt = in_lt0_dt;
        wei_dt = in_lt1_dt;
        dst_dt = out_lt_dt;
    } else if (op_kind == "ConvolutionBackwardData") {
        src_dt = out_lt_dt;
        wei_dt = in_lt1_dt;
        dst_dt = in_lt0_dt;
    } else if (op_kind == "ConvolutionBackwardWeights") {
        src_dt = in_lt0_dt;
        wei_dt = out_lt_dt;
        dst_dt = in_lt1_dt;
    } else {
        assert(!"unexpected op_kind");
        return false;
    }

    dt = {convert_dt(get_data_type(src_dt)), convert_dt(get_data_type(wei_dt)),
            convert_dt(get_data_type(dst_dt))};

    return true;
}

bool get_conv_wtag(const deserialized_op &base_op_ref, std::string &tag) {
    std::string weights_format {};
    if (!base_op_ref.get_attr_string(weights_format, "weights_format"))
        return false;

    logical_tensor::dims strides {}, shape {};
    if (base_op_ref.kind_ == "ConvolutionBackwardWeights") {
        strides = base_op_ref.out_lts_[0].stride_;
        shape = base_op_ref.out_lts_[0].shape_;
    } else {
        strides = base_op_ref.in_lts_[1].stride_;
        shape = base_op_ref.in_lts_[1].shape_;
    }

    if (weights_format == "XIO") {
        // convert the strides to data_format = OIX
        strides.insert(strides.begin(), strides[strides.size() - 1]);
        strides.insert(strides.begin() + 1, strides[strides.size() - 2]);
        strides.erase(strides.end() - 2, strides.end());
    }

    int64_t groups = 1;
    bool has_group = base_op_ref.get_attr_s64(groups, "groups");
    if (has_group && groups > 1) {
        // convert the strides from w/o group to strides w/ group
        dnnl::memory::dim shape_oc = weights_format == "XIO"
                ? shape[strides.size() - 1]
                : shape[0];
        dnnl::memory::dim stride_oc = strides[0];
        strides.insert(strides.begin(), stride_oc * shape_oc / groups);
    }
    size_t ndims = strides.size();
    tag = strides2memory_tag(ndims, strides, true);

    return true;
}

::conv::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::conv::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            conv::get_conv_desc(base_op_ref, op_setting.desc), res);
    DNN_GRAPH_CHECK_SETTINGS(
            conv::get_conv_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(conv::get_conv_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_stag_and_dtag(base_op_ref, op_setting.stag.front(),
                    op_setting.dtag.front(),
                    base_op_ref.kind_ == "ConvolutionBackwardData"),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            conv::get_conv_wtag(base_op_ref, op_setting.wtag.front()), res);

    return op_setting;
}

} // namespace conv

namespace deconv {

bool get_deconv_desc(const deserialized_op &base_op_ref, ::deconv::desc_t &d) {
    d.g = 1;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

    std::string data_format {}, weights_format {};
    std::vector<int64_t> pads_begin, pads_end, strides, dilations;
    int64_t g = 1;

    base_op_ref.get_attr_string(data_format, "data_format");
    base_op_ref.get_attr_string(weights_format, "weights_format");
    base_op_ref.get_attr_s64_vector(pads_begin, "pads_begin");
    base_op_ref.get_attr_s64_vector(pads_end, "pads_end");
    base_op_ref.get_attr_s64_vector(strides, "strides");
    base_op_ref.get_attr_s64_vector(dilations, "dilations");
    base_op_ref.get_attr_s64(g, "groups");

    logical_tensor::dims src_ncx_shape {}, wei_dims {}, dst_ncx_shape {};
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "ConvTranspose") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        wei_dims = base_op_ref.in_lts_[1].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, false);
    } else if (op_kind == "ConvTransposeBackwardData") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, false);
        wei_dims = base_op_ref.in_lts_[1].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, true);
    } else if (op_kind == "ConvTransposeBackwardWeights") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        wei_dims = base_op_ref.out_lts_[0].shape_;
        dst_ncx_shape = base_op_ref.get_NCX_shape(1, true);
    } else {
        assert(!"unexpected op_kind");
        return false;
    }

    d.ndims = static_cast<int>(src_ncx_shape.size());
    d.mb = src_ncx_shape[0];

    assign_shape_val(d.ic, d.iw, d.ih, d.id, src_ncx_shape);
    assign_shape_val(d.oc, d.ow, d.oh, d.od, dst_ncx_shape);

    bool has_h = d.ndims > 3;
    bool has_d = d.ndims > 4;
    if (weights_format == "IOX") {
        // iow, iohw, iodhw
        d.kw = wei_dims[d.ndims - 1];
        d.kh = has_h ? wei_dims[d.ndims - 2] : 1;
        d.kd = has_d ? wei_dims[2] : 1;
    } else if (weights_format == "XOI") {
        // woi, hwoi, dhwoi
        d.kw = wei_dims[d.ndims - 3];
        d.kh = has_h ? wei_dims[d.ndims - 4] : 1;
        d.kd = has_d ? wei_dims[0] : 1;
    } else {
        return false;
    }

    assign_dilation_val(has_h, has_d, d.dw, d.dh, d.dd, dilations, 0);
    assign_stride_padding_val(has_h, has_d, d.sw, d.sh, d.sd, strides, 1);
    assign_stride_padding_val(has_h, has_d, d.pw, d.ph, d.pd, pads_begin, 0);
    assign_stride_padding_val(
            has_h, has_d, d.pw_r, d.ph_r, d.pd_r, pads_end, 0);

    if (g > 1) { // has group
        d.g = g;
        d.has_groups = true;
    }
    return true;
}

bool get_deconv_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    bool ret = false;
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "ConvTranspose") {
        dir = base_op_ref.in_lts_.size() > 2 ? dir_t::FWD_B : dir_t::FWD_I;
        ret = true;
    } else if (op_kind == "ConvTransposeBackwardData") {
        dir = dir_t::BWD_D;
        ret = true;
    } else if (op_kind == "ConvTransposeBackwardWeights") {
        dir = dir_t::BWD_W;
        ret = true;
    } else {
        assert(!"unexpected op_kind");
        return false;
    }
    return ret;
}

bool get_deconv_dt(const deserialized_op &base_op_ref,
        std::vector<dnnl_data_type_t> &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    std::string src_dt {}, wei_dt {}, dst_dt {};
    auto in_lt0_dt = base_op_ref.in_lts_[0].data_type_;
    auto in_lt1_dt = base_op_ref.in_lts_[1].data_type_;
    auto out_lt_dt = base_op_ref.out_lts_[0].data_type_;

    if (rewrite_lt_ids.find(base_op_ref.in_lts_[0].id_) != rewrite_lt_ids.end())
        in_lt0_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.in_lts_[1].id_) != rewrite_lt_ids.end())
        in_lt1_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.out_lts_[0].id_)
            != rewrite_lt_ids.end())
        out_lt_dt = "f32";

    const auto &op_kind = base_op_ref.kind_;

    if (op_kind == "ConvTranspose") {
        src_dt = in_lt0_dt;
        wei_dt = in_lt1_dt;
        dst_dt = out_lt_dt;
    } else if (op_kind == "ConvTransposeBackwardData") {
        src_dt = out_lt_dt;
        wei_dt = in_lt1_dt;
        dst_dt = in_lt0_dt;
    } else if (op_kind == "ConvTransposeBackwardWeights") {
        src_dt = in_lt0_dt;
        wei_dt = out_lt_dt;
        dst_dt = in_lt1_dt;
    } else {
        assert(!"unexpected op_kind");
        return false;
    }

    dt = {convert_dt(get_data_type(src_dt)), convert_dt(get_data_type(wei_dt)),
            convert_dt(get_data_type(dst_dt))};

    return true;
}

bool get_deconv_wtag(const deserialized_op &base_op_ref, std::string &tag) {
    std::string weights_format {};
    if (!base_op_ref.get_attr_string(weights_format, "weights_format"))
        return false;
    if (weights_format != "XOI" && weights_format != "IOX") return false;

    logical_tensor::dims strides {}, shape {};
    if (base_op_ref.kind_ == "ConvTransposeBackwardWeights") {
        strides = base_op_ref.out_lts_[0].stride_;
        shape = base_op_ref.out_lts_[0].shape_;
    } else {
        strides = base_op_ref.in_lts_[1].stride_;
        shape = base_op_ref.in_lts_[1].shape_;
    }

    if (weights_format == "XOI") {
        // convert the strides to weights_format = OIX
        strides.insert(strides.begin(), strides[strides.size() - 2]);
        strides.insert(strides.begin() + 1, strides[strides.size() - 1]);
        strides.erase(strides.end() - 2, strides.end());
    } else if (weights_format == "IOX") {
        // convert the strides to filter_format = OIX
        std::swap(strides[0], strides[1]);
    }

    int64_t groups = 1;
    bool has_group = base_op_ref.get_attr_s64(groups, "groups");
    if (has_group && groups > 1) {
        // convert the strides from w/o group to strides w/ group
        dnnl::memory::dim shape_ic
                = weights_format == "XOI" ? shape[shape.size() - 1] : shape[0];
        dnnl::memory::dim stride_ic = strides[1];
        strides.insert(strides.begin(), stride_ic * shape_ic / groups);
    }
    const size_t ndims = strides.size();
    tag = strides2memory_tag(ndims, strides, true);

    return true;
}

::deconv::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::deconv::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            deconv::get_deconv_desc(base_op_ref, op_setting.desc), res);
    DNN_GRAPH_CHECK_SETTINGS(
            deconv::get_deconv_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(deconv::get_deconv_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_stag_and_dtag(base_op_ref, op_setting.stag.front(),
                    op_setting.dtag.front(),
                    base_op_ref.kind_ == "ConvTransposeBackwardData"),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            deconv::get_deconv_wtag(base_op_ref, op_setting.wtag.front()), res);

    return op_setting;
}

} // namespace deconv

namespace eltwise {
const std::unordered_map<std::string, ::eltwise::alg_t> &
get_eltwise_kind_map() {
    static const std::unordered_map<std::string, ::eltwise::alg_t> map_ {
            {"Abs", ::eltwise::alg_t::ABS},
            {"AbsBackward", ::eltwise::alg_t::ABS},
            {"Clamp", ::eltwise::alg_t::CLIP_V2},
            {"ClampBackward", ::eltwise::alg_t::CLIP_V2},
            {"Elu", ::eltwise::alg_t::ELU},
            {"EluBackward", ::eltwise::alg_t::ELU},
            {"Exp", ::eltwise::alg_t::EXP},
            {"GELU", ::eltwise::alg_t::GELU_ERF},
            {"GELUBackward", ::eltwise::alg_t::GELU_ERF},
            {"HardSigmoid", ::eltwise::alg_t::HARDSIGMOID},
            {"HardSigmoidBackward", ::eltwise::alg_t::HARDSIGMOID},
            {"HardSwish", ::eltwise::alg_t::HARDSWISH},
            {"HardSwishBackward", ::eltwise::alg_t::HARDSWISH},
            {"LeakyReLU", ::eltwise::alg_t::RELU},
            {"Log", ::eltwise::alg_t::LOG},
            {"Mish", ::eltwise::alg_t::MISH},
            {"MishBackward", ::eltwise::alg_t::MISH},
            {"Pow", ::eltwise::alg_t::POW},
            {"Reciprocal", ::eltwise::alg_t::POW},
            {"ReLU", ::eltwise::alg_t::RELU},
            {"ReLUBackward", ::eltwise::alg_t::RELU},
            {"Round", ::eltwise::alg_t::ROUND},
            {"Log", ::eltwise::alg_t::LOG},
            {"Sigmoid", ::eltwise::alg_t::LOGISTIC},
            {"SigmoidBackward", ::eltwise::alg_t::LOGISTIC},
            {"SoftPlus", ::eltwise::alg_t::SRELU},
            {"SoftPlusBackward", ::eltwise::alg_t::SRELU},
            {"Sqrt", ::eltwise::alg_t::SQRT},
            {"SqrtBackward", ::eltwise::alg_t::SQRT},
            {"Square", ::eltwise::alg_t::SQUARE},
            {"Tanh", ::eltwise::alg_t::TANH},
            {"TanhBackward", ::eltwise::alg_t::TANH},
    };
    return map_;
}

bool get_flag_use_dst_for_bwd_compute(const deserialized_op &base_op_ref) {
    const auto it = base_op_ref.attrs_.find("use_dst");
    if (it == base_op_ref.attrs_.end()) return false;
    return it->second.bool_value_;
}

bool get_eltwise_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind.rfind("Backward") == std::string::npos) {
        dir = dir_t::FWD_D;
    } else {
        dir = dir_t::BWD_D;
    }
    return true;
}

bool get_eltwise_dt(const deserialized_op &base_op_ref, dnnl_data_type_t &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    dt = convert_dt(base_op_ref.in_lts_.front().get_data_type());
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        dt = dnnl_f32;
    return true;
}

bool get_eltwise_alg(
        const deserialized_op &base_op_ref, ::eltwise::alg_t &alg) {
    static const std::unordered_map<std::string, ::eltwise::alg_t>
            map_kind_to_alg_dst {
                    {"ClampBackward", ::eltwise::alg_t::CLIP_V2_DST},
                    {"EluBackward", ::eltwise::alg_t::ELU_DST},
                    {"Exp", ::eltwise::alg_t::EXP_DST},
                    {"LeakyReLU", ::eltwise::alg_t::RELU_DST},
                    {"ReLUBackward", ::eltwise::alg_t::RELU_DST},
                    {"SigmoidBackward", ::eltwise::alg_t::LOGISTIC_DST},
                    {"SqrtBackward", ::eltwise::alg_t::SQRT_DST},
                    {"TanhBackward", ::eltwise::alg_t::TANH_DST}};
    const auto &op_kind = base_op_ref.kind_;
    if (get_flag_use_dst_for_bwd_compute(base_op_ref)) {
        if (map_kind_to_alg_dst.find(op_kind) == map_kind_to_alg_dst.end())
            return false;
        alg = map_kind_to_alg_dst.at(op_kind);
    } else {
        const auto &map_kind_to_alg = get_eltwise_kind_map();
        if (map_kind_to_alg.find(op_kind) == map_kind_to_alg.end())
            return false;
        alg = map_kind_to_alg.at(op_kind);
    }
    return true;
}

bool get_eltwise_alpha(const deserialized_op &base_op_ref, float &alpha) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Clamp" || op_kind == "ClampBackward") {
        base_op_ref.get_attr_f32(alpha, "min");
    } else if (op_kind == "Elu" || op_kind == "EluBackward"
            || op_kind == "LeakyReLU" || op_kind == "HardSigmoid"
            || op_kind == "HardSigmoidBackward") {
        base_op_ref.get_attr_f32(alpha, "alpha");
    } else if (op_kind == "Reciprocal") {
        alpha = 1; // Reciprocal is pow(-1)
    } else if (op_kind == "SoftPlus" || op_kind == "SoftPlusBackward") {
        // forced data type conversion due to discrepancy between setting and JSON file
        base_op_ref.get_attr_f32(alpha, "beta");
    } else if (op_kind == "HardSwish" || op_kind == "HardSwishBackward") {
        alpha = 1.f / 6.f;
    } else if (op_kind == "Pow") {
        alpha = 1; // alpha is constant 1 according to graph API Pow definition
    }
    return true;
}

bool get_eltwise_beta(const deserialized_op &base_op_ref, float &beta) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Reciprocal") {
        beta = -1; // Reciprocal is pow(-1)
    } else if (op_kind == "Clamp" || op_kind == "ClampBackward") {
        base_op_ref.get_attr_f32(beta, "max");
    } else if (op_kind == "HardSigmoid" || op_kind == "HardSigmoidBackward"
            || op_kind == "Pow") {
        base_op_ref.get_attr_f32(beta, "beta");
    } else if (op_kind == "HardSwish" || op_kind == "HardSwishBackward") {
        beta = 1.f / 2.f;
    }
    return true;
}

::eltwise::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::eltwise::settings_t op_setting;
    const auto &map_kind_to_alg = get_eltwise_kind_map();
    DNN_GRAPH_CHECK_SETTINGS(
            map_kind_to_alg.find(base_op_ref.kind_) != map_kind_to_alg.end(),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_prb_dims(base_op_ref, op_setting.prb_dims), res);
    DNN_GRAPH_CHECK_SETTINGS(
            eltwise::get_eltwise_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(eltwise::get_eltwise_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_tag(base_op_ref, op_setting.tag.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            eltwise::get_eltwise_alg(base_op_ref, op_setting.alg.front()), res);

    DNN_GRAPH_CHECK_SETTINGS(
            eltwise::get_eltwise_alpha(base_op_ref, op_setting.alpha.front()),
            res);

    DNN_GRAPH_CHECK_SETTINGS(
            eltwise::get_eltwise_beta(base_op_ref, op_setting.beta.front()),
            res);

    return op_setting;
}

} // namespace eltwise

namespace lnorm {

bool get_lnorm_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "LayerNorm") {
        bool keep_stats = false;

        base_op_ref.get_attr_bool(keep_stats, "keep_stats");

        const size_t out_size = base_op_ref.out_lts_.size();
        // output: src, mean(opt), var(opt)
        if (out_size == 1) {
            dir = dir_t::FWD_I;
            if (keep_stats) return false;
        } else if (out_size == 3) {
            dir = dir_t::FWD_D;
            if (!keep_stats) return false;
        } else {
            return false;
        }
    } else if (op_kind == "LayerNormBackward") {
        dir = dir_t::BWD_DW;
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
}

bool get_lnorm_dt(const deserialized_op &base_op_ref, dnnl_data_type_t &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    const auto &in_dt = base_op_ref.in_lts_.front().get_data_type();
    dt = convert_dt(in_dt);
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        dt = dnnl_f32;
    return true;
}

bool get_lnorm_flags(
        const deserialized_op &base_op_ref, ::bnorm::flags_t &flags) {
    bool use_affine = false;
    base_op_ref.get_attr_bool(use_affine, "use_affine");
    const auto &op_kind = base_op_ref.kind_;
    const size_t in_size = base_op_ref.in_lts_.size();
    if (op_kind == "LayerNorm") {
        // input: src, gamma(opt), beta(opt)
        if (use_affine) {
            if (in_size == 3) {
                flags = ::lnorm::USE_SCALE | ::lnorm::USE_SHIFT;
            } else {
                return false;
            }
        } else {
            if (in_size == 1) {
                flags = ::lnorm::NONE;
            } else {
                return false;
            }
        }
    } else if (op_kind == "LayerNormBackward") {
        // input: src, diff_dst, mean, var, gamma(opt), beta(opt)
        if (use_affine) {
            if (in_size == 6) {
                flags = ::lnorm::USE_SCALE | ::lnorm::USE_SHIFT;
            } else {
                return false;
            }
        } else {
            if (in_size == 4) {
                flags = ::lnorm::NONE;
            } else {
                return false;
            }
        }
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
}

::lnorm::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::lnorm::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            get_prb_dims(base_op_ref, op_setting.prb_dims), res);
    DNN_GRAPH_CHECK_SETTINGS(
            lnorm::get_lnorm_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(lnorm::get_lnorm_dt(base_op_ref,
                                     op_setting.dt[0].front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_tag(base_op_ref, op_setting.tag[0].front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            lnorm::get_lnorm_flags(base_op_ref, op_setting.flags.front()), res);

    return op_setting;
}

} // namespace lnorm

namespace matmul {

bool get_matmul_prb_vdims(
        const deserialized_op &base_op_ref, prb_vdims_t &prb_vdims) {

    deserialized_op &base_op = const_cast<deserialized_op &>(base_op_ref);

    auto &src_dims = base_op.in_lts_[0].shape_;
    auto &wei_dims = base_op.in_lts_[1].shape_;
    auto &dst_dims = base_op.out_lts_[0].shape_;
    const auto ndims = dst_dims.size();

    ::graph::extend_dims(base_op.in_lts_[0], ndims);
    ::graph::extend_dims(base_op.in_lts_[1], ndims);
    if (base_op.in_lts_.size() > 2) {
        ::graph::extend_dims(base_op.in_lts_[2], ndims);
    }

    // transpose
    bool transpose_a = false, transpose_b = false;
    base_op_ref.get_attr_bool(transpose_a, "transpose_a");
    base_op_ref.get_attr_bool(transpose_b, "transpose_b");
    if (ndims >= 2) {
        if (transpose_a) std::swap(src_dims[ndims - 1], src_dims[ndims - 2]);
        if (transpose_b) std::swap(wei_dims[ndims - 1], wei_dims[ndims - 2]);
        if (src_dims[ndims - 1] != wei_dims[ndims - 2]) return false;
    } else {
        if (src_dims[0] != wei_dims[0]) return false;
    }

    prb_vdims = prb_vdims_t({src_dims, wei_dims, dst_dims});
    prb_vdims.dst_dims[ndims - 2] = src_dims[ndims - 2];
    prb_vdims.dst_dims[ndims - 1] = wei_dims[ndims - 1];

    return true;
}

bool get_matmul_dt(const deserialized_op &base_op_ref,
        std::vector<dnnl_data_type_t> &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    auto src_dt = base_op_ref.in_lts_[0].data_type_;
    auto wei_dt = base_op_ref.in_lts_[1].data_type_;
    auto dst_dt = base_op_ref.out_lts_[0].data_type_;

    if (rewrite_lt_ids.find(base_op_ref.in_lts_[0].id_) != rewrite_lt_ids.end())
        src_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.in_lts_[1].id_) != rewrite_lt_ids.end())
        wei_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.out_lts_[0].id_)
            != rewrite_lt_ids.end())
        dst_dt = "f32";

    dt = {convert_dt(get_data_type(src_dt)), convert_dt(get_data_type(wei_dt)),
            convert_dt(get_data_type(dst_dt))};

    return true;
}

bool get_matmul_tags(const deserialized_op &base_op_ref, std::string &stag,
        std::string &wtag, std::string &dtag, const int &ndims) {
    logical_tensor::dims src_strides = base_op_ref.in_lts_[0].stride_;
    logical_tensor::dims wei_strides = base_op_ref.in_lts_[1].stride_;
    const logical_tensor::dims &dst_strides = base_op_ref.out_lts_[0].stride_;
    // transpose
    bool transpose_a = false, transpose_b = false;
    base_op_ref.get_attr_bool(transpose_a, "transpose_a");
    base_op_ref.get_attr_bool(transpose_b, "transpose_b");
    if (ndims >= 2) {
        if (transpose_a)
            std::swap(src_strides[ndims - 1], src_strides[ndims - 2]);
        if (transpose_b)
            std::swap(wei_strides[ndims - 1], wei_strides[ndims - 2]);
    }
    stag = strides2memory_tag(ndims, src_strides, true);
    wtag = strides2memory_tag(ndims, wei_strides, true);
    dtag = strides2memory_tag(ndims, dst_strides, true);
    return true;
}

bool get_matmul_bia_dt_mask(const deserialized_op &base_op_ref,
        dnnl_data_type_t &bia_dt, const dnnl_data_type_t dt, int &bia_mask) {
    bia_dt = dnnl_data_type_undef;
    if (base_op_ref.in_lts_.size() <= 2) return true;

    // bia_dt is the same as src_dt
    bia_dt = dt;
    const logical_tensor::dims &bias_shape = base_op_ref.in_lts_[2].shape_;
    const logical_tensor::dims &dst_shape = base_op_ref.out_lts_[0].shape_;
    if (bias_shape.size() != dst_shape.size()) {
        if (bias_shape.size() != 1) return false;
        auto iter
                = std::find(dst_shape.begin(), dst_shape.end(), bias_shape[0]);
        if (iter == dst_shape.end()) return false;
        size_t channel_dim = iter - dst_shape.begin();
        bia_mask = 1 << channel_dim;
    } else {
        bia_mask = 0;
        for (size_t k = 0; k < dst_shape.size(); ++k) {
            if (bias_shape[k] != 1 && bias_shape[k] != dst_shape[k])
                return false;
            if (bias_shape[k] == dst_shape[k]) bia_mask |= 1 << k;
        }
    }
    return true;
}

::matmul::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::matmul::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            matmul::get_matmul_prb_vdims(base_op_ref, op_setting.prb_vdims),
            res);
    DNN_GRAPH_CHECK_SETTINGS(matmul::get_matmul_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            matmul::get_matmul_bia_dt_mask(base_op_ref,
                    op_setting.bia_dt.front(), op_setting.dt.front()[0],
                    op_setting.bia_mask.front()),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            matmul::get_matmul_tags(base_op_ref, op_setting.stag.front(),
                    op_setting.wtag.front(), op_setting.dtag.front(),
                    op_setting.prb_vdims.ndims),
            res);

    return op_setting;
}

} // namespace matmul

namespace pool {

bool get_pool_desc(const deserialized_op &base_op_ref, ::pool::desc_t &d) {

    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

    const auto &op_kind = base_op_ref.kind_;
    std::string data_format {}, rounding_type {};
    std::vector<int64_t> pads_begin {}, pads_end {}, strides {}, kernel {},
            dilations {};

    base_op_ref.get_attr_string(data_format, "data_format");
    base_op_ref.get_attr_s64_vector(pads_begin, "pads_begin");
    base_op_ref.get_attr_s64_vector(pads_end, "pads_end");
    base_op_ref.get_attr_s64_vector(strides, "strides");
    base_op_ref.get_attr_s64_vector(kernel, "kernel");

    if (op_kind == "MaxPool" || op_kind == "MaxPoolBackward") {
        base_op_ref.get_attr_s64_vector(dilations, "dilations");
    }
    if (op_kind == "MaxPool" || op_kind == "AvgPool") {
        base_op_ref.get_attr_string(rounding_type, "rounding_type");
    }

    logical_tensor::dims src_ncx_shape {}, dst_ncx_shape {};

    if (op_kind == "MaxPool" || op_kind == "AvgPool") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, false);
    } else if (op_kind == "MaxPoolBackward") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, false);
        dst_ncx_shape = base_op_ref.get_NCX_shape(1, true);
        // Backward of maxpooling has two inputs
    } else if (op_kind == "AvgPoolBackward") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, false);
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, true);
    } else
        return false;

    d.ndims = static_cast<int>(src_ncx_shape.size());
    d.mb = src_ncx_shape[0];
    assign_shape_val(d.ic, d.iw, d.ih, d.id, src_ncx_shape);
    // for pooling, ic = oc
    assign_shape_val(d.ic, d.ow, d.oh, d.od, dst_ncx_shape);

    bool has_h = d.ndims > 3;
    bool has_d = d.ndims > 4;

    if (op_kind == "MaxPool" || op_kind == "MaxPoolBackward") {
        assign_dilation_val(has_h, has_d, d.dw, d.dh, d.dd, dilations, 0);
    }

    assign_stride_padding_val(has_h, has_d, d.sw, d.sh, d.sd, strides, 1);
    assign_stride_padding_val(has_h, has_d, d.kw, d.kh, d.kd, kernel, 1);
    assign_stride_padding_val(has_h, has_d, d.pw, d.ph, d.pd, pads_begin, 0);
    assign_stride_padding_val(
            has_h, has_d, d.pw_r, d.ph_r, d.pd_r, pads_end, 0);

    if (d.ic == 0) return false;
    if (d.sd <= 0 || d.sh <= 0 || d.sw <= 0) return false;

    return true;
}

bool get_pool_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    bool ret = false;
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "MaxPool" || op_kind == "AvgPool") {
        // we only implement inference
        dir = dir_t::FWD_I;
        ret = true;
    } else if (op_kind == "AvgPoolBackward" || op_kind == "MaxPoolBackward") {
        dir = dir_t::BWD_D;
        ret = true;
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return ret;
}

bool get_pool_dt(const deserialized_op &base_op_ref,
        std::vector<dnnl_data_type_t> &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    auto src_dt = base_op_ref.in_lts_[0].data_type_;
    auto dst_dt = base_op_ref.out_lts_[0].data_type_;
    if (rewrite_lt_ids.find(base_op_ref.in_lts_[0].id_) != rewrite_lt_ids.end())
        src_dt = "f32";
    if (rewrite_lt_ids.find(base_op_ref.out_lts_[0].id_)
            != rewrite_lt_ids.end())
        dst_dt = "f32";

    dt = {convert_dt(get_data_type(src_dt)), convert_dt(get_data_type(dst_dt))};

    return true;
}

bool get_pool_alg(const deserialized_op &base_op_ref, ::pool::alg_t &alg) {

    const auto op_kind_ = base_op_ref.kind_;
    if (op_kind_ == "MaxPool" || op_kind_ == "MaxPoolBackward") {
        alg = ::pool::alg_t::max;
    } else if (op_kind_ == "AvgPool" || op_kind_ == "AvgPoolBackward") {
        bool exclude_pad = false;
        std::string rounding_type {};
        base_op_ref.get_attr_bool(exclude_pad, "exclude_pad");
        base_op_ref.get_attr_string(rounding_type, "rounding_type");

        if (exclude_pad)
            alg = ::pool::alg_t::avg_np;
        else {
            if (op_kind_ == "AvgPool" && rounding_type == "ceil")
                return false;
            else
                alg = ::pool::alg_t::avg_p;
        }

    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
}

::pool::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::pool::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            pool::get_pool_desc(base_op_ref, op_setting.desc), res);
    DNN_GRAPH_CHECK_SETTINGS(
            pool::get_pool_alg(base_op_ref, op_setting.alg.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            pool::get_pool_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(pool::get_pool_dt(base_op_ref,
                                     op_setting.dt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_tag(base_op_ref, op_setting.tag.front()), res);

    return op_setting;
}

} //namespace pool

namespace prelu {

bool get_prelu_prb_vdims(
        const deserialized_op &base_op_ref, prb_vdims_t &prb_vdims) {

    auto src_dims = base_op_ref.in_lts_[0].shape_;
    auto wei_dims = base_op_ref.in_lts_[1].shape_;
    const auto &op_kind = base_op_ref.kind_;
    const auto &ndims = src_dims.size();

    if (op_kind == "PReLU" || op_kind == "PReLUBackward") {
        // handle broadcast
        bool per_channel_broadcast {false};
        base_op_ref.get_attr_bool(
                per_channel_broadcast, "per_channel_broadcast");
        if (ndims != wei_dims.size()) {
            if (!per_channel_broadcast || ndims < wei_dims.size()) return false;
            if (wei_dims.size() > 1 || base_op_ref.has_NXC_format()) {
                wei_dims.insert(wei_dims.begin(), ndims - wei_dims.size(), 1);
            } else {
                // NCX and wei_dims = 1
                wei_dims.insert(wei_dims.begin(), 1);
                wei_dims.insert(wei_dims.end(), ndims - wei_dims.size(), 1);
            }
        }
    }
    // convert from NXC to NCX
    if (base_op_ref.has_NXC_format()) {
        change_format_to_ncx(src_dims, wei_dims);
    }
    prb_vdims = prb_vdims_t({src_dims, wei_dims});

    return true;
}

bool get_prelu_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    bool ret = false;
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "PReLU") {
        dir = dir_t::FWD_D;
        ret = true;
    } else if (op_kind == "PReLUBackward") {
        dir = dir_t::BWD_DW;
        ret = true;
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return ret;
}

bool get_prelu_sdt(const deserialized_op &base_op_ref,
        std::vector<dnnl_data_type_t> &dt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    const auto &in_dt = base_op_ref.in_lts_.front().get_data_type();
    auto sdt = convert_dt(in_dt);
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        sdt = dnnl_f32;
    dt = {sdt, sdt};
    return true;
}

bool get_prelu_stag(
        const deserialized_op &base_op_ref, ::prelu::settings_t &op_setting) {
    std::string tag0, tag1;
    get_driver_tag_by_idx(base_op_ref, tag0);
    get_driver_tag_by_idx(base_op_ref, tag1, 1);
    op_setting.stag = {{tag0, tag1}};
    return true;
}

::prelu::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::prelu::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            prelu::get_prelu_prb_vdims(base_op_ref, op_setting.prb_vdims), res);
    DNN_GRAPH_CHECK_SETTINGS(
            prelu::get_prelu_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(prelu::get_prelu_sdt(base_op_ref,
                                     op_setting.sdt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            prelu::get_prelu_stag(base_op_ref, op_setting), res);

    return op_setting;
}

} // namespace prelu

namespace reduction {

bool get_reduction_prb_vdims(
        const deserialized_op &base_op_ref, prb_vdims_t &prb_vdims) {
    const auto &src_dims = base_op_ref.in_lts_[0].shape_;
    auto dst_dims = base_op_ref.out_lts_[0].shape_;

    std::vector<int64_t> axes {};
    int64_t ndims = src_dims.size();
    base_op_ref.get_attr_s64_vector(axes, "axes");
    // -ndims <= axis <= ndims-1
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < -ndims || axes[i] > ndims - 1) { return false; }
        // make axes >= 0
        if (axes[i] < 0) axes[i] += ndims;
    }

    bool keep_dims = false;
    base_op_ref.get_attr_bool(keep_dims, "keep_dims");
    // unsequeeze dst dims for primitive
    if (!keep_dims) {
        std::sort(axes.begin(), axes.end());
        for (const auto &axis : axes) {
            dst_dims.insert(dst_dims.begin() + axis, 1);
        }
    }

    prb_vdims.vdims = {src_dims, dst_dims};
    prb_vdims.dst_dims = src_dims;
    prb_vdims.ndims = static_cast<int>(src_dims.size());
    return true;
}

bool get_reduction_dt(const deserialized_op &base_op_ref, dnnl_data_type_t &sdt,
        dnnl_data_type_t &ddt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {

    auto sdt_ = convert_dt(base_op_ref.in_lts_.front().get_data_type());
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        sdt_ = dnnl_f32;
    sdt = sdt_;

    auto ddt_ = convert_dt(base_op_ref.out_lts_.front().get_data_type());
    if (rewrite_lt_ids.find(base_op_ref.out_lts_.front().id_)
            != rewrite_lt_ids.end())
        ddt_ = dnnl_f32;
    ddt = ddt_;
    return true;
}

bool get_reduction_alg(
        const deserialized_op &base_op_ref, ::reduction::alg_t &alg) {
    static const std::unordered_map<std::string, ::reduction::alg_t>
            map_kind_to_alg {{"ReduceSum", ::reduction::alg_t::sum},
                    {"ReduceProd", ::reduction::alg_t::mul},
                    {"ReduceMin", ::reduction::alg_t::min},
                    {"ReduceMax", ::reduction::alg_t::max},
                    {"ReduceMean", ::reduction::alg_t::mean},
                    {"ReduceL1", ::reduction::alg_t::norm_lp_power_p_sum},
                    {"ReduceL2", ::reduction::alg_t::norm_lp_sum}};
    const auto &op_kind = base_op_ref.kind_;
    alg = map_kind_to_alg.at(op_kind);
    return true;
}

bool get_reduction_p(const deserialized_op &base_op_ref, float &p) {
    const auto &op_kind = base_op_ref.kind_;
    p = (op_kind == "ReduceL2") ? 2.f : 1.f;
    return true;
}

::reduction::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::reduction::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(reduction::get_reduction_prb_vdims(
                                     base_op_ref, op_setting.prb_vdims),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            reduction::get_reduction_dt(base_op_ref, op_setting.sdt.front(),
                    op_setting.ddt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_stag_and_dtag(base_op_ref, op_setting.stag.front(),
                    op_setting.dtag.front()),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            reduction::get_reduction_alg(base_op_ref, op_setting.alg.front()),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            reduction::get_reduction_p(base_op_ref, op_setting.p.front()), res);

    return op_setting;
}

} // namespace reduction

namespace reorder {

bool get_reorder_dt(const deserialized_op &base_op_ref, dnnl_data_type_t &sdt,
        dnnl_data_type_t &ddt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    sdt = convert_dt(base_op_ref.in_lts_.front().get_data_type());
    ddt = convert_dt(base_op_ref.out_lts_.front().get_data_type());

    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        sdt = dnnl_f32;
    if (rewrite_lt_ids.find(base_op_ref.out_lts_.front().id_)
            != rewrite_lt_ids.end())
        ddt = dnnl_f32;

    return true;
}

bool get_reorder_stag_and_dtag(const deserialized_op &base_op_ref,
        std::string &stag, std::string &dtag) {
    bool ret = get_driver_stag_and_dtag(base_op_ref, stag, dtag);
    if (!ret) return false;
    ret = get_driver_tag(base_op_ref, dtag, true);
    return ret;
}

bool get_reorder_attrs(const deserialized_op &base_op_ref,
        attr_t::arg_scales_t &arg_scales, attr_t::zero_points_t &zp) {

    const auto &op_kind = base_op_ref.kind_;
    std::string qtype {};
    base_op_ref.get_attr_string(qtype, "qtype");
    int arg = 0;
    if (op_kind == "Dequantize" || op_kind == "DynamicDequantize")
        arg = DNNL_ARG_SRC;
    else if (op_kind == "Quantize" || op_kind == "DynamicQuantize")
        arg = DNNL_ARG_DST;
    else
        return false;

    // scale
    attr_t::policy_t scale_policy = attr_t::policy_t::COMMON;
    int64_t axis = 1;
    if (qtype == "per_channel") {
        // per dimension
        base_op_ref.get_attr_s64(axis, "axis");
        const auto ndims = base_op_ref.in_lts_.front().shape_.size();
        if (axis < 0) axis += ndims;
        if (axis == 0) {
            scale_policy = attr_t::PER_DIM_0;
        } else if (axis == 1) {
            scale_policy = attr_t::PER_DIM_1;
        } else if (axis == 2) {
            scale_policy = attr_t::PER_DIM_2;
        } else if (axis == 3) {
            scale_policy = attr_t::PER_DIM_3;
        } else {
            assert(!"unsupported axis");
        }
    }

    if (op_kind == "Dequantize" || op_kind == "Quantize") {
        std::vector<float> scales {};
        base_op_ref.get_attr_f32_vector(scales, "scales");
        arg_scales.set(arg, {scale_policy, scales.front()});
        std::vector<int64_t> zps;
        base_op_ref.get_attr_s64_vector(zps, "zps");
        // currently, zps only support per_tensor quantization in primitive
        zp.set(arg, attr_t::policy_t::COMMON, zps.front());
    } else if (op_kind == "DynamicDequantize" || op_kind == "DynamicQuantize") {
        //  TODO: benchdnn needs to alloc memory based on is_def() function.
        //  so add tmp value for per_tensor scales && zps to make is_def()
        //  return false to alloc memory.
        if (qtype == "per_tensor") {
            arg_scales.set(arg, {scale_policy, 2});
        } else {
            arg_scales.set(arg, {scale_policy});
        }
        // zps is optional for DynamicDequantize/DynamicQuantize, default is
        // symmetric quantization
        if (base_op_ref.in_lts_.size() == 3) {
            zp.set(arg, attr_t::policy_t::COMMON, 1);
        }
    }
    return true;
}

::reorder::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::reorder::settings_t op_setting;
    const auto op_kind = base_op_ref.kind_;

    DNN_GRAPH_CHECK_SETTINGS(
            get_prb_dims(base_op_ref, op_setting.prb_dims), res);

    DNN_GRAPH_CHECK_SETTINGS(
            reorder::get_reorder_dt(base_op_ref, op_setting.sdt.front(),
                    op_setting.ddt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            reorder::get_reorder_stag_and_dtag(base_op_ref,
                    op_setting.stag.front(), op_setting.dtag.front()),
            res);

    if (op_kind == "Dequantize" || op_kind == "Quantize"
            || op_kind == "DynamicDequantize" || op_kind == "DynamicQuantize") {
        DNN_GRAPH_CHECK_SETTINGS(reorder::get_reorder_attrs(base_op_ref,
                                         op_setting.scales.front(),
                                         op_setting.zero_points.front()),
                res);
    }
    return op_setting;
}

} // namespace reorder

namespace resampling {

bool get_resampling_desc(
        const deserialized_op &base_op_ref, ::resampling::desc_t &d) {
    std::string data_format {};
    base_op_ref.get_attr_string(data_format, "data_format");

    logical_tensor::dims src_ncx_shape {}, dst_ncx_shape {};
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Interpolate") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, true);
        dst_ncx_shape = base_op_ref.get_NCX_shape(0, false);
    } else if (op_kind == "InterpolateBackward") {
        src_ncx_shape = base_op_ref.get_NCX_shape(0, false);
        dst_ncx_shape = base_op_ref.get_NCX_shape(1, true);
    } else {
        return false;
    }

    d.ndims = static_cast<int>(src_ncx_shape.size());
    d.mb = src_ncx_shape[0];
    assign_shape_val(d.ic, d.iw, d.ih, d.id, src_ncx_shape);
    // for resampling, ic = oc
    assign_shape_val(d.ic, d.ow, d.oh, d.od, dst_ncx_shape);

    return true;
}

bool get_resampling_dir(const deserialized_op &base_op_ref, dir_t &dir) {

    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "Interpolate") {
        dir = ::dir_t::FWD_D;
    } else if (op_kind == "InterpolateBackward") {
        dir = ::dir_t::BWD_D;
    } else {
        return false;
    }

    return true;
}

bool get_resampling_dt(const deserialized_op &base_op_ref,
        dnnl_data_type_t &sdt, dnnl_data_type_t &ddt,
        const std::unordered_set<size_t> &rewrite_lt_ids) {

    const auto &inputs = base_op_ref.in_lts_;
    auto sdt_ = convert_dt(inputs.front().get_data_type());
    auto ddt_ = convert_dt(base_op_ref.out_lts_.front().get_data_type());

    // bf16-to-f32 rewrite
    if (rewrite_lt_ids.find(inputs.front().id_) != rewrite_lt_ids.end())
        sdt_ = dnnl_f32;
    if (rewrite_lt_ids.find(base_op_ref.out_lts_.front().id_)
            != rewrite_lt_ids.end())
        ddt_ = dnnl_f32;

    sdt = sdt_;
    ddt = ddt_;
    return true;
}

bool get_resampling_alg(
        const deserialized_op &base_op_ref, ::resampling::alg_t &alg) {
    std::string alg_value {};
    base_op_ref.get_attr_string(alg_value, "mode");
    if (alg_value == "linear" || alg_value == "bilinear"
            || alg_value == "trilinear") {
        alg = ::resampling::alg_t::linear;
    } else if (alg_value == "nearest") {
        alg = ::resampling::alg_t::nearest;
    } else {
        return false;
    }

    return true;
}

::resampling::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::resampling::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            resampling::get_resampling_desc(base_op_ref, op_setting.desc), res);
    DNN_GRAPH_CHECK_SETTINGS(
            resampling::get_resampling_dir(base_op_ref, op_setting.dir.front()),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            resampling::get_resampling_dt(base_op_ref, op_setting.sdt.front(),
                    op_setting.ddt.front(), rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_tag(base_op_ref, op_setting.tag.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            resampling::get_resampling_alg(base_op_ref, op_setting.alg.front()),
            res);

    return op_setting;
}

} //namespace resampling

namespace softmax {

bool get_softmax_dir(const deserialized_op &base_op_ref, dir_t &dir) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "SoftMax" || op_kind == "LogSoftmax") {
        dir = dir_t::FWD_D;
    } else if (op_kind == "SoftMaxBackward"
            || op_kind == "LogSoftmaxBackward") {
        dir = dir_t::BWD_D;
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
};

bool get_softmax_sdt_and_ddt(const deserialized_op &base_op_ref,
        ::softmax::settings_t &op_setting,
        const std::unordered_set<size_t> &rewrite_lt_ids) {
    const auto &in_dt = base_op_ref.in_lts_.front().get_data_type();
    dnnl_data_type_t dt = convert_dt(in_dt);
    if (rewrite_lt_ids.find(base_op_ref.in_lts_.front().id_)
            != rewrite_lt_ids.end())
        dt = dnnl_f32;
    op_setting.sdt.front() = dt;
    op_setting.ddt.front() = dt;
    return true;
}

bool get_softmax_alg(
        const deserialized_op &base_op_ref, ::softmax::alg_t &alg) {
    const auto &op_kind = base_op_ref.kind_;
    if (op_kind == "SoftMax" || op_kind == "SoftMaxBackward") {
        alg = ::softmax::alg_t::SOFTMAX;
    } else if (op_kind == "LogSoftmax" || op_kind == "LogSoftmaxBackward") {
        alg = ::softmax::alg_t::LOGSOFTMAX;
    } else {
        assert(!"unsupported op_kind");
        return false;
    }
    return true;
};

::softmax::settings_t get_setting(const deserialized_op &base_op_ref,
        const std::unordered_set<size_t> &rewrite_lt_ids, res_t *res) {
    ::softmax::settings_t op_setting;
    DNN_GRAPH_CHECK_SETTINGS(
            get_prb_dims(base_op_ref, op_setting.prb_dims), res);
    DNN_GRAPH_CHECK_SETTINGS(
            softmax::get_softmax_dir(base_op_ref, op_setting.dir.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(softmax::get_softmax_sdt_and_ddt(
                                     base_op_ref, op_setting, rewrite_lt_ids),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_stag_and_dtag(base_op_ref, op_setting.stag.front(),
                    op_setting.dtag.front()),
            res);
    DNN_GRAPH_CHECK_SETTINGS(
            softmax::get_softmax_alg(base_op_ref, op_setting.alg.front()), res);
    DNN_GRAPH_CHECK_SETTINGS(
            get_driver_axis(base_op_ref, op_setting.axis.front()), res);
    return op_setting;
}

} // namespace softmax

} // namespace graph
