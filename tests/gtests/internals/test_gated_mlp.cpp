/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>

#include "mlp_internal.hpp"
#include "test_utils.hpp"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>

#include <memory>
#include <random>

using namespace dnnl;
using tag = memory::format_tag;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

using mdt = memory::data_type;

static bool verbose = false; // enable for debug
static const int min_runs = 4;

struct mlp_dims_t {
    dim mb;
    dim ic;
    dim oc;

    bool do_quantize;
    int gateup_group_size;
    int down_group_size;

    memory::data_type wgu_wt;
    memory::data_type wgu_s_dt;
    memory::data_type wgu_zp_dt;

    memory::data_type wd_wt;
    memory::data_type wd_s_dt;
    memory::data_type wd_zp_dt;

    quantize_type qtype;
};

struct gmlp_tensors {
    memory m_x, m_w_gate, m_w_up, m_w_down;
    memory m_w_gate_quantized, m_w_up_quantized, m_w_down_quantized;
    memory m_w_gate_scales, m_w_up_scales, m_w_down_scales;
    memory m_w_gate_zp, m_w_up_zp, m_w_down_zp;
    // memory m_x_quantized, m_x_scale, m_x_zp;
    memory m_out, m_out_quantized;
    memory m_fc_gate, m_fc_up, m_fc_down;
    memory m_fc_gate_t;

    dnnl::primitive_attr gateup_attr_quantized, down_attr_quantized;
    memory::dims wgu_groups, wd_groups;
};

std::ostream &operator<<(std::ostream &ss, const mlp_dims_t &p) {
    ss << "mb_" << p.mb;
    ss << "_ic_" << p.ic;
    ss << "_oc_" << p.oc;

    std::string quant = p.do_quantize ? "_quant_" : "_noquant_";
    ss << quant;
    ss << "_gu_group_size_" << p.gateup_group_size;
    ss << "_gd_group_size_" << p.down_group_size;

    ss << "_wgu_wt_" << p.wgu_wt;
    if (p.wgu_wt != mdt::f16) {
        ss << "_wgu_sdt_" << p.wgu_s_dt;
        ss << "_wgu_zpdt_" << p.wgu_zp_dt;
    }

    ss << "_wd_wt_" << p.wd_wt;
    if (p.wd_wt != mdt::f16) {
        ss << "_wd_sdt_" << p.wd_s_dt;
        ss << "_wd_zpdt_" << p.wd_zp_dt;
    }

    if (p.wgu_wt != mdt::f16 || p.wd_wt != mdt::f16) {
        ss << "_qtype_" << p.qtype;
    }
    return ss;
}

std::string PrintToString(const ::testing::TestParamInfo<mlp_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

void fill_const(std::vector<float> &out, const float c) {
    for (int i = 0; i < out.size(); ++i) {
        out[i] = c;
    }
}

void fill_const(std::vector<float16_t> &out, const float c) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;
    const unsigned seed = 2;

    if (random_data_f.empty()) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    size_t chunk = std::min(nrand, out.size());
    for (int i = 0; i < out.size(); ++i) {
        out[i] = float16_t(c); //TMP matmul only
    }
    //for (size_t i = 0; i < out.size(); i += nrand) {
    //    size_t chunk = std::min(nrand, out.size() - i);
    //    std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    //}
}

void fill_lin(std::vector<float> &out) {
    for (int i = 0; i < out.size(); ++i) {
        out[i] = i;
    }
}

void fill_hceye(std::vector<float> &out, int ldi = 32) {
    for (int i = 0; i < out.size(); ++i) {
        out[i] = ((((i / ldi) % ldi == (i % ldi))) ? 1.f
                                                   : 0.f); //TMP matmul only
        //out[i] = ((( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only
    }
}
void fill_hceye(std::vector<float16_t> &out, int ldi = 32) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    for (int i = 0; i < out.size(); ++i) {
        //out[i] = half_cast<half>( (i/33) == (i%33) ? 1.f : 0.f); //TMP matmul only
        //
        //out[i] = half_cast<half>( (i/32)%32 == (i%32) ? 1.f : 0.f); //TMP matmul only

        out[i] = float16_t(
                (((i / ldi) % 32 == (i % 32))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((((i/ldi)%32  == ((i+2)%32)) || ( (i/ldi) == (i%32))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((((i/ldi)  == ((i+2)%ldi)) || ( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = half_cast<half>((( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only

        /*
        if((i/ldi == i % ldi) && i/ldi == 1) {
            out[i] = half_cast<half>(-1); //TMP matmul only
        }
        */

        //
        //out[i] = half_cast<half>( (i/64) == (i%64) ? 1.f : 0.f); //TMP matmul only
        //out[i] = 1.f;
    }
}

gmlp_tensors get_descriptors(dnnl::engine &eng, mlp_dims_t p) {
    gmlp_tensors out;

    // Prepare input and output shapes to construct the swiglu graph.
    const memory::dims O_proj_sz = {p.mb, p.ic};
    const memory::dims W_gate_sz = {p.ic, p.oc};
    const memory::dims W_up_sz = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic};
    //const memory::dims FC_gate_sz = {p.oc, p.mb};
    const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};

    // memory::dims scale_gateup_sz = {p.ic, p.oc };
    const memory::dims scales_gateup_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups: // 1 scale for entire row
                return memory::dims {
                        W_gate_sz[0] / p.gateup_group_size, W_gate_sz[1]};
            case quantize_type::per_token:
                return memory::dims {W_gate_sz[0], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1};
        }
        throw "wat?\n";
    }();

    auto dt = memory::data_type::f16;
    auto wgu_wt = (p.wgu_wt == mdt::undef)
            ? dt
            : p.wgu_wt; ///undef = non quantized , can be u8
    auto wgu_s_dt = (p.wgu_s_dt == mdt::undef) ? dt : p.wgu_s_dt;
    auto wgu_zp_dt = (p.wgu_zp_dt == mdt::undef) ? dt : p.wgu_zp_dt;

    auto wd_wt = (p.wd_wt == mdt::undef) ? dt : p.wd_wt;
    auto wd_s_dt = (p.wd_s_dt == mdt::undef) ? dt : p.wd_s_dt;
    auto wd_zp_dt = (p.wd_zp_dt == mdt::undef) ? dt : p.wd_zp_dt;

    auto FC_gate_md = memory::desc(FC_gate_sz, dt, tag::ab);
    auto FC_up_md = memory::desc(FC_up_sz, dt, tag::ab);
    auto FC_down_md = memory::desc(FC_down_sz, dt, tag::ab);

    auto FC_gate_md_t = memory::desc(FC_gate_sz, dt, tag::ba);
    // auto FC_gate_md_t = memory::desc(FC_gate_sz_t, dt, tag::ab);

    // clang-format off
    auto x_md      = memory::desc(O_proj_sz, dt, tag::ab);
    auto w_gate_md = memory::desc(W_gate_sz, dt, tag::ab);
    auto w_up_md   = memory::desc(W_up_sz,   dt, tag::ab);
    auto w_down_md = memory::desc(W_down_sz, dt, tag::ab);

    //auto x_qnt_md      = memory::desc(x_sz, dt, tag::ab);
    auto w_gate_qnt_md = memory::desc(W_gate_sz, wgu_wt, tag::ab);
    auto w_up_qnt_md   = memory::desc(W_up_sz,   wgu_wt, tag::ab);
    auto w_down_qnt_md = memory::desc(W_down_sz,  wd_wt, tag::ab);

    //auto x_scale_md      = memory::desc(O_proj_sz, dt, tag::ab);
    auto w_gate_scales_md = memory::desc(scales_gateup_sz, wgu_s_dt, tag::ab);
    auto w_up_scales_md   = memory::desc(scales_gateup_sz,   wgu_s_dt, tag::ab);
    auto w_down_scales_md = memory::desc(W_down_sz, wd_s_dt, tag::ab);

    //auto x_zp_md      = memory::desc(O_proj_sz, dt, tag::ab);
    auto w_gate_zp_md = memory::desc(scales_gateup_sz, wgu_zp_dt, tag::ab);
    auto w_up_zp_md   = memory::desc(scales_gateup_sz,   wgu_zp_dt, tag::ab);
    auto w_down_zp_md = memory::desc(W_down_sz, wd_zp_dt, tag::ab);

    auto output_md     = memory::desc(FC_gate_sz, dt, tag::ab);
    auto output_qnt_md = memory::desc(FC_gate_sz, dt, tag::ab);
    // clang-format on

    // Create memory objects
    out.m_x = memory(x_md, eng);
    out.m_w_gate = memory(w_gate_md, eng);
    out.m_w_up = memory(w_up_md, eng);
    out.m_w_down = memory(w_down_md, eng);

    //out.m_x_quantized      = memory(x_qnt_md, eng);
    out.m_w_gate_quantized = memory(w_gate_qnt_md, eng);
    out.m_w_up_quantized = memory(w_up_qnt_md, eng);
    out.m_w_down_quantized = memory(w_down_qnt_md, eng);

    //out.m_x_scale      = memory(x_scale_md, eng);
    out.m_w_gate_scales = memory(w_gate_scales_md, eng);
    out.m_w_up_scales = memory(w_up_scales_md, eng);
    out.m_w_down_scales = memory(w_down_scales_md, eng);

    //out.m_x_zp      = memory(x_zp_md, eng);
    out.m_w_gate_zp = memory(w_gate_zp_md, eng);
    out.m_w_up_zp = memory(w_up_zp_md, eng);
    out.m_w_down_zp = memory(w_down_zp_md, eng);

    out.m_fc_gate = memory(FC_gate_md, eng);
    out.m_fc_up = memory(FC_up_md, eng);
    out.m_fc_down = memory(FC_down_md, eng);

    out.m_out = memory(output_md, eng);
    out.m_out_quantized = memory(output_qnt_md, eng);

    out.m_fc_gate_t = memory(FC_gate_md_t, eng);

    // Allocate user data.
    std::vector<float> x_data(product(O_proj_sz));
    std::vector<float> w_gate_data(product(W_gate_sz));
    std::vector<float> w_up_data(product(W_up_sz));
    std::vector<float> w_down_data(product(W_down_sz));

    // std::vector<float>x_quantized_data(product(O_proj_sz), 1.f);
    std::vector<float> w_gate_quantized_data(product(W_gate_sz), 1.f);
    std::vector<float> w_up_quantized_data(product(W_up_sz), 1.f);
    std::vector<float> w_down_quantized_data(product(W_down_sz), 1.f);

    // std::vector<float>x_scale_data(product(O_proj_sz), 1.f);
    std::vector<float> w_gate_scales_data(product(W_gate_sz), 1.f);
    std::vector<float> w_up_scales_data(product(W_up_sz), 1.f);
    std::vector<float> w_down_scales_data(product(W_down_sz), 1.f);

    // std::vector<int>x_zp_data_signed(product(O_proj_sz), 0);
    std::vector<int> w_gate_zp_data_signed(product(W_gate_sz), 0);
    std::vector<int> w_up_zp_data_signed(product(W_up_sz), 0);
    std::vector<int> w_down_zp_data_signed(product(W_down_sz), 0);

    // std::vector<unsigned>x_zp_data_unsigned(product(O_proj_sz), 0);
    std::vector<unsigned> w_gate_zp_data_unsigned(product(W_gate_sz), 0);
    std::vector<unsigned> w_up_zp_data_unsigned(product(W_up_sz), 0);
    std::vector<unsigned> w_down_zp_data_unsigned(product(W_down_sz), 0);

    out.wgu_groups = {};
    out.wd_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups: {
            int gateup_mask = 1 << 1 | 1 << 0;
            int down_mask = 1 << 1 | 1 << 0;

            out.wgu_groups = {p.gateup_group_size, 1};
            out.wd_groups = {p.down_group_size, 1};

            /*
            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef) {
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, gateup_mask, {p.gateup_group_size, 1}, wgu_s_dt);
            }
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef) {
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, gateup_mask, {p.gateup_group_size, 1}, wgu_zp_dt);
            }

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef) {
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, down_mask, {p.down_group_size, 1}, wd_s_dt);
            }
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef) {
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, down_mask, {p.down_group_size, 1}, wd_zp_dt);
            }
            */
        } break;
        case quantize_type::per_token: {
            // faster dim | slower dim
            // dim 3, 2,  {1 , 0}
            int gateup_mask = 1; //  1 << 0;
            int down_mask = 1; //  1 << 0;

            /*
            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef) {
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, gateup_mask, {}, wgu_s_dt);
            }
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef) {
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, gateup_mask, {}, wgu_zp_dt);
            }

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef) {
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, down_mask, {}, wd_s_dt);
            }
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef) {
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, down_mask, {}, wd_zp_dt);
            }
            */
        } break;
        case quantize_type::per_tensor:
            /*
            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef) {
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, 0, {}, wgu_s_dt);
            }
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef) {
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, 0, {}, wgu_zp_dt);
            }

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef) {
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, 0, {}, wd_s_dt);
            }
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef) {
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, 0, {}, wd_zp_dt);
            }
            */
            break;
        case quantize_type::no_quantization: break;
    }

    fill_random(x_data, x_md);
    //fill_lin(x_data); //testdata

    fill_random_quantized(w_gate_quantized_data, w_gate_qnt_md,
            (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
    fill_random_quantized(w_up_quantized_data, w_up_qnt_md,
            (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
    fill_random_quantized(w_down_quantized_data, w_down_qnt_md,
            (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));

    if (p.qtype != quantize_type::no_quantization) {
        if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef) {
            fill_random_scales(w_gate_scales_data, w_gate_scales_md);
            //w_gate_scales_data[63] = 2.f;
            fill_random_scales(w_up_scales_data, w_up_scales_md);
        }
        //if (qtype == quantize_type::per_token) {
        if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef) {
            fill_random_quantized(w_gate_zp_data_signed, w_gate_zp_md);
            fill_random_quantized(w_gate_zp_data_unsigned, w_gate_zp_md);
            fill_random_quantized(w_up_zp_data_signed, w_up_zp_md);
            fill_random_quantized(w_up_zp_data_unsigned, w_up_zp_md);
        }
        //}
        if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef)
            fill_random_scales(w_down_scales_data, w_down_scales_md);
        if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef) {
            fill_random_quantized(w_down_zp_data_signed, w_down_zp_md);
            fill_random_quantized(w_down_zp_data_unsigned, w_down_zp_md);
        }
    }

    int wgu_group_size = p.gateup_group_size;
    int wd_group_size = p.down_group_size;

    if (p.qtype == quantize_type::per_tensor) {
        wgu_group_size = W_gate_sz[0] * W_gate_sz[1];
        wd_group_size = W_down_sz[0] * W_down_sz[1];
    }

    //vector<float> x_data, w_gate_data, w_up_data, w_down_data;
    //if(p.qtype == quantize_type::no_quantization) {
    if (!p.do_quantize) {
        printf("no quant init\n");
        fill_random(w_gate_data, w_gate_md);
        //fill_hceye(w_gate_data, p.ic); //testdata

        fill_random(w_up_data, w_up_md);
        fill_random(w_down_data, w_down_md);
    } else {
        if (wgu_wt == mdt::s4 || wgu_wt == mdt::s8) {
            printf("s4/s8 quant init\n");
            w_gate_data = dequantize(w_gate_quantized_data, w_gate_md,
                    w_gate_scales_md, w_gate_zp_data_signed, w_gate_scales_data,
                    wgu_group_size, p.qtype, out.wgu_groups, 0);

            w_up_data = dequantize(w_up_quantized_data, w_up_md, w_up_scales_md,
                    w_up_zp_data_signed, w_up_scales_data, wgu_group_size,
                    p.qtype, out.wgu_groups, 0);

            w_down_data = dequantize(w_down_quantized_data, w_down_md,
                    w_down_scales_md, w_down_zp_data_signed, w_down_scales_data,
                    wgu_group_size, p.qtype, out.wd_groups, 0);
        } else {
            printf("quant init\n");
            w_gate_data = dequantize(w_gate_quantized_data, w_gate_md,
                    w_gate_scales_md, w_gate_zp_data_unsigned,
                    w_gate_scales_data, wgu_group_size, p.qtype, out.wgu_groups,
                    0);
            w_up_data = dequantize(w_up_quantized_data, w_up_md, w_up_scales_md,
                    w_up_zp_data_unsigned, w_up_scales_data, wgu_group_size,
                    p.qtype, out.wgu_groups, 0);
            w_down_data = dequantize(w_down_quantized_data, w_down_md,
                    w_down_scales_md, w_down_zp_data_unsigned,
                    w_down_scales_data, wgu_group_size, p.qtype, out.wd_groups,
                    0);
        }
    }

    // Write data to tensor object's handle.
    write_to_dnnl_memory(x_data.data(), out.m_x);
    write_to_dnnl_memory(w_gate_data.data(), out.m_w_gate);
    write_to_dnnl_memory(w_up_data.data(), out.m_w_up);
    write_to_dnnl_memory(w_down_data.data(), out.m_w_down);

    write_to_dnnl_memory(w_gate_quantized_data.data(), out.m_w_gate_quantized);
    write_to_dnnl_memory(w_up_quantized_data.data(), out.m_w_up_quantized);
    write_to_dnnl_memory(w_down_quantized_data.data(), out.m_w_down_quantized);

    if (wgu_wt == mdt::s4 || wgu_wt == mdt::s8) {
        write_to_dnnl_memory(w_gate_zp_data_signed.data(), out.m_w_gate_zp);
        write_to_dnnl_memory(w_up_zp_data_signed.data(), out.m_w_up_zp);
        write_to_dnnl_memory(w_down_zp_data_signed.data(), out.m_w_down_zp);
    } else {
        write_to_dnnl_memory(w_gate_zp_data_unsigned.data(), out.m_w_gate_zp);
        write_to_dnnl_memory(w_up_zp_data_unsigned.data(), out.m_w_up_zp);
        write_to_dnnl_memory(w_down_zp_data_unsigned.data(), out.m_w_down_zp);
    }

    write_to_dnnl_memory(w_gate_scales_data.data(), out.m_w_gate_scales);
    write_to_dnnl_memory(w_up_scales_data.data(), out.m_w_up_scales);
    write_to_dnnl_memory(w_down_scales_data.data(), out.m_w_down_scales);

    printf("memory data types?? %d %d %d\n",
            out.m_w_gate_scales.get_desc().get_data_type(),
            out.m_w_up_scales.get_desc().get_data_type(),
            out.m_w_down_scales.get_desc().get_data_type());

    //transpose_strides(eng, out.m_fc_gate_t, out.m_fc_gate);

    /*
        transpose_strides(eng, out.m_key_scales_t, out.m_key_scales);
        transpose_strides(eng, out.m_key_t, out.m_key);
        transpose_strides(eng, out.m_key_t_quantized, out.m_key_quantized);
        transpose_strides(eng, out.m_value_t, out.m_value);
        transpose_strides(eng, out.m_value_t_quantized, out.m_value_quantized);
    */

    return out;
}

void print_test_case(memory::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

template <typename T>
void bench_gated_mlp_primitives(std::vector<T> &res, double &avg_time,
        gmlp_tensors &t, dnnl::engine &eng, dnnl::stream &strm,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // extract memory objects
    auto m_O_proj = t.m_x;
    auto m_W_gate = t.m_w_gate;
    auto m_W_up = t.m_w_up;
    auto m_W_down = t.m_w_down;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    // extract memory descriptors
    auto O_proj_md = t.m_x.get_desc();
    auto W_gate_md = t.m_w_gate.get_desc();
    auto W_up_md = t.m_w_up.get_desc();
    auto W_down_md = t.m_w_down.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    //printf("[%d %d] x [%d %d] = [%d %d]\n", O_proj_md.get_dims()[0], O_proj_md.get_dims()[1],
    //W_up_md.get_dims()[0], W_up_md.get_dims()[1],
    //FC_up_md.get_dims()[0], FC_up_md.get_dims()[1]);

    auto m_FC_gate_t = t.m_fc_gate_t;

    // fc_up
    primitive_attr bmm0_attr;

    //bmm0_attr.set_scales(DNNL_ARG_WEIGHTS,
    //(1<<0) + (1<<1), {testGRP, 1}, memory::data_type::f16);
    //bmm0_attr.set_scratchpad_mode(scratchpad_mode::user);

    auto bmm0_pd = matmul::primitive_desc(
            eng, O_proj_md, W_up_md, FC_up_md, bmm0_attr);
    auto bmm0_prim = matmul(bmm0_pd);

    // fc_gate -> swish -> mul
    primitive_attr bmm1_attr;
    //bmm1_attr.set_scratchpad_mode(scratchpad_mode::user); // TODO: needed? no threading in this example...
    post_ops bmm1_po;
    bmm1_po.append_eltwise(algorithm::eltwise_swish, 1.f, 1.f);
    bmm1_po.append_binary(algorithm::binary_mul, m_FC_up.get_desc());
    bmm1_attr.set_post_ops(bmm1_po);

    auto bmm1_pd = matmul::primitive_desc(
            eng, O_proj_md, W_gate_md, FC_gate_md, bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    primitive_attr bmm2_attr;
    //bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(
            eng, FC_gate_md, W_down_md, FC_down_md, bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    /*
    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, tag::a);

    // allocate intermediate memory
    auto m_score = memory(score_md, eng);
    auto m_scratchpad = memory(scratchpad_md, eng);
    */

    ////////////////TMP transpose for test
    //const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    //auto m_FC_gate_t = t.m_fc_gate_t;

    primitive_attr reorder_attr;
    auto reorder_pd = reorder::primitive_desc(
            eng, FC_gate_md, eng, FC_gate_md, reorder_attr);

    auto reorder_prim = reorder(reorder_pd);

    std::unordered_map<int, memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, m_FC_gate});
    reorder_args.insert({DNNL_ARG_DST, m_FC_gate_t});
    ///////////////////

    const auto loop = [&]() {
    ///////////////////
    //TMP!!!!
// test first MM only
#if 0
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate} });

        //reorder_prim.execute(strm, reorder_args);

//// TMP!!! mm + swish + elwise_mul
#elif 1
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                        {DNNL_ARG_DST, m_FC_up}});

        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_FC_up}});
        //reorder_prim.execute(strm, reorder_args);

#else //TEST full gated mlp
        bmm0_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_up},
                        {DNNL_ARG_DST, m_FC_up}});

        // each primitive will use all threads
        bmm1_prim.execute(strm,
                {{DNNL_ARG_SRC, m_O_proj}, {DNNL_ARG_WEIGHTS, m_W_gate},
                        {DNNL_ARG_DST, m_FC_gate},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                m_FC_up}});

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, m_FC_gate}, {DNNL_ARG_WEIGHTS, m_W_down},
                        {DNNL_ARG_DST, m_FC_down}});
#endif
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    if (verbose && product(FC_down_md.get_dims()) < (64 * 64) + 1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("resprim----------[%ld %ld]\n", p.mb, p.ic);
        printf("------inpA\n");
        print_mem(m_O_proj, "-prim");
        printf("------inpB\n");
        print_mem(m_W_gate, "-prim");
    }
#define TEST_VS_TRANSPOSE 1
#if TEST_VS_TRANSPOSE
    //transpose(eng, m_FC_gate_t, m_FC_gate);
    transpose_strides(eng, m_FC_gate_t, m_FC_gate);

    if (verbose && product(FC_down_md.get_dims()) < (64 * 64) + 1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("------tmpres\n");
        print_mem(m_FC_gate_t, "-prim");
    }

    float16_t *mapped_ptr_f16 = (float16_t *)m_FC_gate_t.map_data();
    res.resize(product(FC_gate_md.get_dims()));
    for (int i = 0; i < res.size(); ++i) {
        res[i] = mapped_ptr_f16[i];
    }
    m_FC_gate_t.unmap_data(mapped_ptr_f16);
#else
    if (verbose && product(FC_down_md.get_dims()) < (64 * 64) + 1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("------tmpres\n");
        print_mem(m_FC_gate, "-prim");
    }

    float16_t *mapped_ptr_f16 = (float16_t *)m_FC_gate.map_data();
    res.resize(product(FC_gate_md.get_dims()));
    for (int i = 0; i < res.size(); ++i) {
        res[i] = mapped_ptr_f16[i];
    }
    m_FC_gate.unmap_data(mapped_ptr_f16);
#endif
}

template <typename T>
void bench_gated_mlp_internal(std::vector<T> &res, double &avg_time,
        gmlp_tensors &t, dnnl::engine &eng, dnnl::stream strm,
        const mlp_dims_t &p, double time_limit = 0.) {

    using namespace dnnl::impl;
    printf("eng?\n");
    const bool quick_test = (time_limit == 0.);

    // Create memory objects
    auto m_O_proj = t.m_x;
    auto m_W_gate = t.m_w_gate;
    auto m_W_up = t.m_w_up;
    auto m_W_down = t.m_w_down;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    // Create memory objects
    auto O_proj_md = t.m_x.get_desc();
    auto W_gate_md = t.m_w_gate.get_desc();
    auto W_up_md = t.m_w_up.get_desc();
    auto W_down_md = t.m_w_down.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    // quantization memory
    auto m_W_gate_quant = t.m_w_gate_quantized;
    auto m_W_gate_scales = t.m_w_gate_scales;
    auto m_W_gate_zp = t.m_w_gate_zp;
    auto m_W_up_quant = t.m_w_up_quantized;
    auto m_W_up_scales = t.m_w_up_scales;
    auto m_W_up_zp = t.m_w_up_zp;

    auto m_W_gate_quant_md = t.m_w_gate_quantized.get_desc();
    auto m_W_gate_scales_md = t.m_w_gate_scales.get_desc();
    auto m_W_gate_zp_md = t.m_w_gate_zp.get_desc();
    auto m_W_up_quant_md = t.m_w_up_quantized.get_desc();
    auto m_W_up_scales_md = t.m_w_up_scales.get_desc();
    auto m_W_up_zp_md = t.m_w_up_zp.get_desc();

    const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    //const memory::dims FC_gate_sz_t = {p.mb, p.oc};
    auto FC_gate_md_t
            = memory::desc(FC_gate_sz_t, FC_gate_md.get_data_type(), tag::ab);
    auto m_FC_gate_t = memory(FC_gate_md_t, eng);

    if (verbose) {
        printf("memquant\n");
        //print_mem(t.m_w_gate, "-gen_desc_wgate");
        //print_mem(m_W_gate, "-gen_desc_wgate");
        print_mem(t.m_w_gate_quantized, "-gen_desc_wgate_quant");
        //print_mem(t.m_w_up_quantized, "-gen_desc_wgate_quant");
        //print_mem(m_W_gate_quant, "-gen_desc_wgate_quant");
        print_mem(t.m_w_gate_scales, "-gen_desc_wgate_scale");
        //print_mem(m_W_gate_scales, "-gen_desc_wgate_scale");
        //print_mem(t.m_w_gate_zp, "-gen_desc_wgate_zp");
        print_mem(m_W_gate_zp, "-gen_desc_wgate_zp");
    }

    //primitive_attr bmm0_attr;
    //bmm0_attr.set_scratchpad_mode(scratchpad_mode::user);
    //auto bmm0_pd = matmul::primitive_desc(
    //eng, O_proj_md, W_up_md, FC_up_md, bmm0_attr);
    //auto prim_fused_internal = matmul(bmm0_pd);

    primitive_attr gmlp_attr, gate_attr, up_attr, down_attr;
    //DNNL_ARG_ATTR_SCALES | DNNL_ARG_WTS_GATE,
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            //wts_gate scale+zp
            gate_attr.set_scales(DNNL_ARG_SRC_0, (1 << 0) + (1 << 1),
                    {1, p.gateup_group_size},
                    m_W_gate_scales_md.get_data_type());
            gate_attr.set_zero_points(
                    DNNL_ARG_WEIGHTS, //TODO: wat??? why no arg_src0,1,2,3...
                    (1 << 0) + (1 << 1), {1, p.gateup_group_size},
                    m_W_gate_zp_md.get_data_type());
            //wts_up scale+zp
            up_attr.set_scales(DNNL_ARG_SRC_1, (1 << 0) + (1 << 1),
                    {1, p.gateup_group_size}, m_W_up_scales_md.get_data_type());
            up_attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1),
                    {1, p.gateup_group_size}, m_W_up_zp_md.get_data_type());

            break;
        case quantize_type::per_tensor:
            //wts_gate scale+zp
            gate_attr.set_scales(DNNL_ARG_SRC_0, 0, {1, p.gateup_group_size},
                    m_W_gate_scales_md.get_data_type());
            gate_attr.set_zero_points(
                    DNNL_ARG_WEIGHTS, //TODO: wat??? why no arg_src0,1,2,3...
                    0, {1, p.gateup_group_size},
                    m_W_gate_zp_md.get_data_type());
            //wts_up scale+zp
            up_attr.set_scales(DNNL_ARG_SRC_1, 0, {1, p.gateup_group_size},
                    m_W_up_scales_md.get_data_type());
            up_attr.set_zero_points(DNNL_ARG_WEIGHTS, 0,
                    {1, p.gateup_group_size}, m_W_up_zp_md.get_data_type());

            break;

        case quantize_type::no_quantization: break;
        default: break;
    }

    auto gmlp_pd = [&]() {
        if (p.do_quantize) {
            //TODO: W_up_md_quant
            return gmlp::primitive_desc(eng, O_proj_md, m_W_gate_quant_md,
                    m_W_up_quant_md, W_down_md, FC_gate_md_t, gmlp_attr,
                    gate_attr, up_attr);
        } else {
            return gmlp::primitive_desc(eng, O_proj_md, W_gate_md, W_up_md,
                    W_down_md, FC_gate_md_t, gmlp_attr);
        }
    }();

    auto prim_fused_internal = gmlp(gmlp_pd);

    const auto loop = [&]() {
        if (p.do_quantize) {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC_0, m_O_proj},
                            {DNNL_ARG_SRC_1, m_W_gate_quant},
                            {DNNL_ARG_SRC_2, m_W_up_quant},
                            {DNNL_ARG_SRC_3, m_W_down},
                            //{DNNL_ARG_DST, m_FC_down}}); //CORRECT ARG
                            {DNNL_ARG_DST, m_FC_gate_t},
                            {DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_SCALES,
                                    m_W_gate_scales},
                            {DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_gate_zp},
                            {DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_SCALES,
                                    m_W_up_scales},
                            {DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_ZERO_POINTS,
                                    m_W_up_zp}}); // TMP ARG for mm test
        } else {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC_0, m_O_proj}, {DNNL_ARG_SRC_1, m_W_gate},
                            {DNNL_ARG_SRC_2, m_W_up},
                            {DNNL_ARG_SRC_3, m_W_down},
                            //{DNNL_ARG_DST, m_FC_down}}); //CORRECT ARG
                            {DNNL_ARG_DST,
                                    m_FC_gate_t}}); // TMP ARG for mm test
        }
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    loop();
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 5;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        loop();
        //print_mem(m_W_gate, "-ilolloopafnternal");
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "internal gmlp primitive runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    //transpose(eng, m_FC_gate, m_FC_gate_t);

    if (verbose && product(FC_down_md.get_dims()) < (64 * 64) + 1) {
        printf("resint----------[%ld %ld]\n", p.mb, p.ic);
        printf("------inpA\n");
        print_mem(m_O_proj, "-internal");
        printf("------inpB\n");
        print_mem(m_W_gate, "-internal");
        printf("------tmpres\n");
        print_mem(m_FC_gate_t, "-internal");
    }

    float16_t *mapped_ptr_f16 = (float16_t *)m_FC_gate_t.map_data();
    res.resize(product(FC_gate_md.get_dims()));
    //printf("linmem");
    for (int i = 0; i < res.size(); ++i) {
        //printf("%f ",mapped_ptr_f16[i].f());
        res[i] = mapped_ptr_f16[i];
    }
    m_FC_gate_t.unmap_data(mapped_ptr_f16);
}

const char *get_type_string(logical_tensor::data_type dt) {
    const char *type_string = "unknown";

#define TYPE_CASE(T) \
    if (dt == logical_tensor::data_type::T) type_string = #T;
    TYPE_CASE(f16);
    TYPE_CASE(f32);
    TYPE_CASE(bf16);
#undef TYPE_CASE

    return type_string;
}

void print_test_case(logical_tensor::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << get_type_string(dt);
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

enum class api_kind { primitive, graph, internal_hack };

template <typename T>
void bench(std::vector<T> &res, double &avg_time, gmlp_tensors &t, api_kind api,
        dnnl::engine &eng, dnnl::stream &strm, const mlp_dims_t &p,
        double time_limit = 0.) {

    try {
        if (api == api_kind::primitive) {
            bench_gated_mlp_primitives(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        } else if (api == api_kind::graph) {
            //bench_gated_mlp_graph(ekind, static_cast<logical_tensor::data_type>(dt),
            //p, time_limit);
            //get_mem_pool().clear();
        } else {
            bench_gated_mlp_internal(
                    res, avg_time, t, eng, strm, p, time_limit);
            strm.wait();
        }
    } catch (dnnl::error &e) {
        // Catch and report unimplemented cases.
        if (e.status == dnnl_unimplemented) {
            std::cout << "unsupported mlp" << std::endl;
        } else
            throw;
    }
}

template <typename T>
void check_memory(memory &gold, memory &test) {
    T *mapped_ptr_gold = (T *)gold.map_data();
    T *mapped_ptr_test = (T *)test.map_data();

    auto dims = gold.get_desc().get_dims();
    auto strides = gold.get_desc().get_strides();

    int mismatches = 0;
    int total = 0;
    float fthreshold = 0.f;
    if (std::is_same<T, float16_t>::value) {
        fthreshold = 0.001466f;
    } else {
        fthreshold = 0.0079f;
    }

    float max_diff = std::numeric_limits<float>::min();
    std::map<int, std::map<int, int>> hist;
    bool verbose = false;
    for_(int l = 0; l < dims[0]; l++)
    for_(int k = 0; k < dims[1]; k++)
    for_(int j = 0; j < dims[2]; j++)
    for (int i = 0; i < dims[3]; i++) {
        auto offset = l * strides[0] + k * strides[1] + j * strides[2]
                + i * strides[3];
        auto o_gold = (float)mapped_ptr_gold[offset];
        auto o_test = (float)mapped_ptr_test[offset];
        total++;

        float abs_diff = abs(o_gold - o_test);
        bool is_nan = isnan(o_gold) || isnan(o_test);

        bool is_mismatch = is_nan
                || (abs(o_gold) > 1.f ? abs_diff > abs(o_gold * fthreshold)
                                      : abs_diff > fthreshold);
        if (max_diff < abs_diff) {
            if (verbose) {
                printf("new max: gold: %f vs test: %f diff: %f\n", o_gold,
                        o_test, abs_diff);
            }
            max_diff = abs_diff;
        }
        if (is_mismatch) {
            hist[0][l]++;
            hist[1][k]++;
            hist[2][j]++;
            hist[3][i]++;
        }
        if ((is_mismatch && mismatches++ < 32) || is_nan) {
            if (verbose)
                fprintf(stderr,
                        "Mismatch at (%d,%d,%d,%d): test %f "
                        "vs. gold %f (diff: %f thresh: %f)\n",
                        l, k, j, i, o_test, o_gold, abs_diff,
                        (abs(o_gold) > 2.f ? abs(o_gold * fthreshold)
                                           : fthreshold));
        }
    }

    gold.unmap_data(mapped_ptr_gold);
    test.unmap_data(mapped_ptr_test);

    int threshold = total * 0.0006;

    ASSERT_LE(mismatches, threshold)
            << "max diff: " << max_diff << "out of: " << total;
}

class mlp_test : public ::testing::TestWithParam<mlp_dims_t> {
public:
    virtual void SetUp() override {
        p = GetParam();
        eng = dnnl::engine(engine::kind::gpu, 0);
        strm = dnnl::stream(eng);
        t = get_descriptors(eng, p);
    }

protected:
    mlp_dims_t p;
    dnnl::engine eng;
    dnnl::stream strm;
    gmlp_tensors t;
};

TEST_P(mlp_test, compare) {

    auto tensors = t;
    auto params = p;

    std::vector<float> resp, resi;
    std::vector<float16_t> resph, resih;
    double avg_time_int, avg_time_prim;

    printf("PRIMITIVE\n");
    bench(resph, avg_time_prim, tensors, api_kind::primitive, eng, strm, params,
            2000.0 /*ms*/);

    printf("INTERNAL\n");
    bench(resih, avg_time_int, tensors, api_kind::internal_hack, eng, strm,
            params, 2000.0 /*ms*/);

    if (resih.size() == 0) {
        printf("[WARNING] Empty output! internal kernel fail X_X\n");
        EXPECT_TRUE(false);
    }
    int n_mismatches = 0, n_matches = 0;
    printf("resih.size() %zu\n", resih.size());
    float max_diff = 0.0f, max_val, max_gold;
    for (int i = 0; i < resih.size(); ++i) {
        float abs_diff = std::abs(resih[i] - resph[i]);
        float rel_diff = std::abs((resih[i] - resph[i]) / resih[i]);
        if (abs_diff > 1e-4 && rel_diff > 5e-3) {

            if (isfinite(rel_diff) && (abs_diff) > max_diff) {
                max_diff = abs_diff;
                max_val = resih[i];
                max_gold = resph[i];
            }

            n_mismatches++;
            if (n_mismatches < 10)
                printf("mismatch @ %d, %f != %f\n", i, float(resih[i]),
                        float(resph[i])); //TODO: improve
        } else {
            if (std::abs(float16_t(resih[i])) > 5e-3) {
                n_matches++;
                if (n_matches < 10)
                    printf("vs @ %d, %f == %f\n", i, float(resih[i]),
                            float(resph[i])); //TODO: improve
            }
        }
    }
    printf("total mismatches: %d \n", n_mismatches);
    printf("avg time internal: %f vs %f avg time primitive, w/speedup of %f\n",
            avg_time_int, avg_time_prim, avg_time_prim / avg_time_int);

    int total_size = resph.size();
    int threshold = total_size * 0.0006;

    std::cout << "max diff: " << max_diff << ":  " << max_val
              << " != " << max_gold << std::endl;
    ASSERT_LE(n_mismatches, threshold) << "out of: " << total_size;
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(VEC,
    mlp_test,
                 // mb, seq_len, kv_grp_sz,  hd_num, hd_size, qry_num, kg_sz, vgrp_sz,       dt,      kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt, qtype
    testing::Values(
        // B = 1024
             mlp_dims_t{32,  32,   32,    false, // mb ic oc quant?
                         1, 1, // gateup, wd group size
                     mdt::f16, mdt::f16, mdt::f16, // dt wgateup
                     mdt::f16, mdt::f16, mdt::f16, // dt wd
                     quantize_type::per_token},
             mlp_dims_t{1024,  3584,   18944,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  3584,   4864,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  3584,   14336,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  3584,   27392,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
 //           mlp_dims_t{1024,  896,   18944,    false, // mb ic oc quant?  //TODO: 896 failing?
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1024,  896,   4864,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1024,  896,   14336,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1024,  896,   27392,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
             mlp_dims_t{1024,  4096,   18944,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  4096,   4864,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  4096,   14336,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1024,  4096,   27392,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},

             // B==1
             mlp_dims_t{1,  3584,   18944,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  3584,   4864,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  3584,   14336,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  3584,   27392,    false,
                         1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
 //           mlp_dims_t{1,  896,   18944,    false, // mb ic oc quant?  //TODO: 896 failing?
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1,  896,   4864,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1,  896,   14336,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
 //           mlp_dims_t{1,  896,   27392,    false,
 //                       1, 1,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   mdt::f16, mdt::f16, mdt::f16,
 //                   quantize_type::per_token},
             mlp_dims_t{1,  4096,   18944,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  4096,   4864,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  4096,   14336,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},
             mlp_dims_t{1,  4096,   27392,    false, 1, 1,
                     mdt::f16, mdt::f16, mdt::f16,
                     mdt::f16, mdt::f16, mdt::f16,
                     quantize_type::per_token},

       // B = 1024, quantized w=u8
            mlp_dims_t{32,  32,   32,   true, 8, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  3584,   18944,   true, 8, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 8, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 8, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 8, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},

            mlp_dims_t{1024,  3584,   18944,   true, 128, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 128, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 128, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 128, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},

        // B = 1024, quantized w=s8
            mlp_dims_t{32,  32,   32,   true, 8, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  3584,   18944,   true, 8, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 8, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 8, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 8, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},

            mlp_dims_t{1024,  3584,   18944,   true, 128, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 128, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 128, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 128, 1,
                    mdt::s8, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},

        // B = 1024, quantized w=u4
            mlp_dims_t{32,  32,   32,   true, 16, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  3584,   18944,   true, 8, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 8, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 8, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 8, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},

            //group size must be 8,16,32??? ;cannot work for 128 && u4
            mlp_dims_t{1024,  3584,   18944,   true, 32, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  896,   4864,   true, 32, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   14336,   true, 32, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{1024,  4096,   27392,   true, 32, 1,
                    mdt::u4, mdt::f16, mdt::u8,
                    mdt::u4, mdt::f16, mdt::u8,
                    quantize_type::per_token_with_groups},

            //additional 4bit quant
            mlp_dims_t{32,  32,   32,   true, 16, 1,
                    mdt::s4, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{32,  32,   32,   true, 16, 1,
                    mdt::u4, mdt::f16, mdt::s8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},
            mlp_dims_t{32,  32,   32,   true, 16, 1,
                    mdt::s4, mdt::f16, mdt::u8,
                    mdt::s8, mdt::f16, mdt::s8,
                    quantize_type::per_token_with_groups},

            // per tensor
            mlp_dims_t{1024,  3584,   18944,   true, 1, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_tensor},
            mlp_dims_t{1024,  896,   4864,   true, 1, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_tensor},
            mlp_dims_t{1024,  4096,   14336,   true, 1, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_tensor},
            mlp_dims_t{1024,  4096,   27392,   true, 1, 1,
                    mdt::u8, mdt::f16, mdt::u8,
                    mdt::u8, mdt::f16, mdt::u8,
                    quantize_type::per_tensor}

    //,
    ), &PrintToString);
