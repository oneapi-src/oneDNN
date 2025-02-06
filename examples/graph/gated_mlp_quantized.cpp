/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"

#include "oneapi/dnnl/experimental/dnnl_experimental.hpp" ///TMP FOR TESTING INTERNAL

#include "../half.hpp"
#include "graph_example_utils.hpp"

#include <dnnl_test_common.hpp>
#include <gtest/gtest.h>
#include <oneapi/dnnl/dnnl.hpp>

#include <memory>
#include <random>

//static bool verbose = true;
static bool verbose = false;

using namespace dnnl;
using tag = memory::format_tag;

using half_float::half;
using half_float::half_cast;

using namespace dnnl::graph;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

using std::vector;

using mdt = memory::data_type;

enum class quantize_type {
    no_quantization,
    per_tensor,
    per_token,
    per_token_with_groups
};

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
};

std::ostream &operator<<(std::ostream &ss, const quantize_type &qt) {
    switch (qt) {
        case quantize_type::no_quantization: ss << "no_quantization"; break;
        case quantize_type::per_tensor: ss << "per_tensor"; break;
        case quantize_type::per_token: ss << "per_token"; break;
        case quantize_type::per_token_with_groups:
            ss << "per_token_with_groups";
            break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const memory::data_type &dt) {
    switch (dt) {
        case mdt::f32: ss << "f32"; break;
        case mdt::s32: ss << "s32"; break;
        case mdt::f16: ss << "f16"; break;
        case mdt::s8: ss << "s8"; break;
        case mdt::u8: ss << "u8"; break;
        case mdt::s4: ss << "s4"; break;
        case mdt::u4: ss << "u4"; break;
        default: ss << "na"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const mlp_dims_t &p) {
    ss << "mb_" << p.mb;
    ss << "_ic_" << p.ic;
    ss << "_oc_" << p.oc;

    std::string quant = p.do_quantize ? "_quant_" : "_noquant_";
    ss << quant;
    ss << "_gu_group_size_" << p.gateup_group_size;
    ss << "_gd_group_size_" << p.down_group_size;

    ss << "_wgu_wt_" <<  p.wgu_wt;
    if (p.wgu_wt != mdt::f16) {
        ss << "_wgu_sdt_" <<  p.wgu_s_dt;
        ss << "_wgu_zpdt_" <<  p.wgu_zp_dt;
    }

    ss << "_wd_wt_" <<  p.wd_wt;
    if (p.wd_wt != mdt::f16) {
        ss << "_wd_sdt_"  <<  p.wd_s_dt;
        ss << "_wd_zpdt_" <<  p.wd_zp_dt;
    }

    if (p.wgu_wt != mdt::f16 || p.wd_wt != mdt::f16) { ss << "_qtype_" << p.qtype; }
    return ss;
}

std::string PrintToString(const ::testing::TestParamInfo<mlp_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

static const int min_runs = 4;

template <typename T>
vector<float> dequantize(const vector<float> &input, memory::desc &desc,
        primitive_attr &attr, memory::desc &scale_md,
        const vector<T> &zero_points, const vector<float> &scales,
        int group_size, quantize_type qtype = quantize_type::per_token,
        bool is_k = true) {

    vector<float> out(input.size());
    if (qtype == quantize_type::per_tensor) {
        printf("qqper tensor\n");
        for (size_t i = 0; i < input.size(); i++) {
            //printf("zp %f-s %f  ", static_cast<float>(zero_points[i / group_size]), static_cast<float>(scales[i / group_size]));
            out[i] = (input[i] - zero_points[i / group_size])
                    * scales[i / group_size];
        }
    } else {
        auto dims = desc.get_dims();
        auto strides = desc.get_strides();

        auto scales_dim = scale_md.get_dims();
        auto zp_dim = scale_md.get_dims();
        auto scales_strides = scale_md.get_strides();
        auto zp_strides = scale_md.get_strides();

        int sg = 0;
        int zg = 0;

        auto zp_attr = attr.get()->zero_points_;
        auto scales_attr = attr.get()->scales_;
        dnnl::impl::dim_t scales_groups[12];
        dnnl::impl::dim_t zp_groups[12];
        if (qtype == quantize_type::per_token_with_groups) {
            printf("qqper token w/group \n");
            std::copy(scales_attr.scales_[DNNL_ARG_WEIGHTS].group_dims_,
                    scales_attr.scales_[DNNL_ARG_WEIGHTS].group_dims_ + 12,
                    scales_groups);
            std::copy(zp_attr.get_groups(DNNL_ARG_WEIGHTS),
                    zp_attr.get_groups(DNNL_ARG_WEIGHTS) + 12, zp_groups);
        } else if (qtype == quantize_type::per_token) {
            printf("qqper token \n");
            if (is_k) {
                scales_groups[0] = dims[0];
                scales_groups[1] = 1;
                zp_groups[0] = dims[0];
                zp_groups[1] = 1;
            } else {
                scales_groups[0] = 1;
                scales_groups[1] = dims[1];
                zp_groups[0] = 1;
                zp_groups[1] = dims[1];
            }

        } else {
            printf("qqelse \n");

            scales_groups[0] = 1;
            scales_groups[1] = 1;
            zp_groups[0] = 1;
            zp_groups[1] = 1;
        }

        int scale_offset = 0;
        int zp_offset = 0;

        for (int j = 0; j < dims[0]; j++) {
            for (int i = 0; i < dims[1]; i++) {
                if (scales_groups[0] > 1) {
                    sg = scales_groups[0];
                    scale_offset =
                              j / sg * scales_strides[0]
                            + i * scales_strides[1];
                } else if (scales_groups[1] > 1) {
                    sg = scales_groups[1];
                    scale_offset =
                              j * scales_strides[0]
                            + i / sg * scales_strides[1];
                }
                if (zp_groups[1] > 1) {
                    zg = zp_groups[1];
                    zp_offset =
                            + j * zp_strides[0]
                            + i / zg * zp_strides[1];
                } else if (zp_groups[0] > 1) {
                    zg = zp_groups[0];
                    zp_offset = j / zg * zp_strides[0] + i;
                }
                int offset = j * strides[0] + i * strides[1];

                out[offset] = (input[offset] - zero_points[zp_offset])
                              * scales[scale_offset];
            }
        }

        //int groups = zero_points.size();
        //for (int g = 0; g < groups; g++) {
        //    for (int i = g * group_size; i < g * group_size + group_size; i++) {
        //        out[i] = (input[i] - zero_points[g]) * scales[g];
        //        printf("out: %f = (input: %f - zero_point: %d) * scale: %f\n",
        //                out[i], input[i], zero_points[g], scales[g]);
        //    }
        //}
    }
    return out;
}

void transpose(dnnl::engine eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    void *ptr2 = out.map_data();
    void *ptr1 = in.map_data();

    std::memcpy(ptr2, ptr1, in.get_desc().get_size());
    in.unmap_data(ptr1);
    out.unmap_data(ptr2);

    //dnnl::reorder(in, out).execute(s, in, out);
}

void transpose_strides(dnnl::engine eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    //void *ptr2 = out.map_data();
    //void *ptr1 = in.map_data();

    //std::memcpy(ptr2, ptr1, in.get_desc().get_size());
    //in.unmap_data(ptr1);
    //out.unmap_data(ptr2);

    if (out.get_desc().get_data_type() == mdt::u4
            || out.get_desc().get_data_type() == mdt::s4) {
        auto desc = in.get_desc();
        auto dims = desc.get_dims();
        auto strides = desc.get_strides();
        auto strides_t = out.get_desc().get_strides();

        char *mapped_ptr = (char *)in.map_data();
        char *mapped_ptr_t = (char *)out.map_data();
        for (int l = 0; l < dims[0]; l++) {
            for (int k = 0; k < dims[1]; k++) {
                for (int j = 0; j < dims[2]; j++) {
                    for (int i = 0; i < dims[3]; i++) {
                        int is_odd = i % 2;
                        int is_odd_t = j % 2;

                        auto offset = l * strides[0] + k * strides[1]
                                + j * strides[2] + i * strides[3];
                        offset /= 2;

                        auto offset_t = l * strides_t[0] + k * strides_t[1]
                                + j * strides_t[2] + i * strides_t[3];
                        offset_t /= 2;

                        auto &val = mapped_ptr[offset];
                        auto &val_t = mapped_ptr_t[offset_t];

                        char bits;
                        if (is_odd) {
                            bits = val & 0xf0;
                            bits >>= 4;
                        } else {
                            bits = val & 0x0f;
                        }
                        if (is_odd_t) {
                            val_t |= (bits << 4);
                        } else {
                            val_t |= bits;
                        }
                    }
                }
            }
        }
        in.unmap_data(mapped_ptr);
        out.unmap_data(mapped_ptr_t);
    } else {
        dnnl::reorder(in, out).execute(s, in, out);
        s.wait();
    }
}

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_real_distribution<float> dist_f(-2.0f, 2.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator);
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

template <typename T>
void fill_random_quantized(std::vector<T> &out, bool is_unsigned = false) {
    static std::vector<T> random_data_f;
    static std::vector<T> random_data_u;
    constexpr size_t nrand = 1024;

    if (random_data_f.empty() || random_data_u.empty()) {
        std::mt19937 generator;
        //std::uniform_int_distribution<int> dist_f(-7, 8); //TODO:whichrange?
        //std::uniform_int_distribution<unsigned> dist_u(0, 10);
        std::uniform_int_distribution<int> dist_f(-4, 4);
        std::uniform_int_distribution<unsigned> dist_u(0, 6);

        random_data_u.resize(nrand);
        for (auto &d : random_data_u) {
            d = dist_u(generator);
        }
        random_data_f.resize(nrand);
        for (auto &d : random_data_f) {
            d = dist_f(generator);
        }
    }

    if (std::is_same<unsigned, T>::value || is_unsigned) {
        for (size_t i = 0; i < out.size(); i += nrand) {
            size_t chunk = std::min(nrand, out.size() - i);
            std::memcpy(&out[i], random_data_u.data(), chunk * sizeof(T));
        }
    } else {
        for (size_t i = 0; i < out.size(); i += nrand) {
            size_t chunk = std::min(nrand, out.size() - i);
            std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(T));
        }
    }
}

void fill_random_scales(std::vector<float> &out) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

    if (random_data_f.empty()) {
        std::mt19937 generator;
        std::uniform_int_distribution<int> dist_f(-16, 16);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(generator) * 0.125f;
    }

    for (size_t i = 0; i < out.size(); i += nrand) {
        size_t chunk = std::min(nrand, out.size() - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}


void fill_const(std::vector<float> &out, const float c) {
   for(int i=0; i<out.size(); ++i) {
        out[i] = c; //TMP matmul only
   }
}
void fill_const(std::vector<half> &out, const float c) {
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
   for(int i=0; i<out.size(); ++i) {
        out[i] = half_cast<half>(c); //TMP matmul only
   }
   //for (size_t i = 0; i < out.size(); i += nrand) {
   //    size_t chunk = std::min(nrand, out.size() - i);
   //    std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
   //}
}

void fill_lin(std::vector<float> &out) {
   for(int i=0; i<out.size(); ++i) {
        out[i] = i;
   }
}

void fill_hceye(std::vector<float> &out, int ldi=32) {
   for(int i=0; i<out.size(); ++i) {
        out[i] = ((( (i/ldi)%ldi == (i%ldi))) ? 1.f : 0.f); //TMP matmul only
        //out[i] = ((( (i/ldi) == (i%ldi))) ? 1.f : 0.f); //TMP matmul only
   }
}
void fill_hceye(std::vector<half> &out, int ldi=32) {
    static std::vector<float> random_data_f;
    constexpr size_t nrand = 1037;

   for(int i=0; i<out.size(); ++i) {
        //out[i] = half_cast<half>( (i/33) == (i%33) ? 1.f : 0.f); //TMP matmul only
        //
        //out[i] = half_cast<half>( (i/32)%32 == (i%32) ? 1.f : 0.f); //TMP matmul only

        out[i] = half_cast<half>((( (i/ldi)%32 == (i%32))) ? 1.f : 0.f); //TMP matmul only
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

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, size_t seq_len) {
    const size_t pos = seq_len * 3 / 4;
    for (size_t i = 0; i < mask.size(); ++i) {
        if (i % seq_len < pos)
            mask[i] = 0.f;
        else
            mask[i] = -1 * std::numeric_limits<float>::infinity();
    }
}

// Read from handle, write to memory
template <typename T>
inline void write_to_dnnl_memory(const T *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

#ifdef DNNL_WITH_SYCL
    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)handle)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)handle)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, handle, size).wait();
            }
        }
        return;
    }
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    dnnl::stream s(eng);
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        if (mem.get_desc().get_data_type() != dnnl_f32
                && std::is_same<T, float>::value) {
            memory mem_f32_mem(
                    {mem.get_desc().get_dims(), memory::data_type::f32,
                            mem.get_desc().get_strides()},
                    eng);
            write_to_dnnl_memory<float>((const float *)handle, mem_f32_mem);
            dnnl::reorder(mem_f32_mem, mem).execute(s, mem_f32_mem, mem);
            s.wait();
        } else if (mem.get_desc().get_data_type() != dnnl_s32
                && std::is_same<T, int>::value) {
            memory mem_s32_mem(
                    {mem.get_desc().get_dims(), memory::data_type::s32,
                            mem.get_desc().get_strides()},
                    eng);
            write_to_dnnl_memory<int>((const int *)handle, mem_s32_mem);
            dnnl::reorder(mem_s32_mem, mem).execute(s, mem_s32_mem, mem);
            s.wait();
        } else if ((mem.get_desc().get_data_type() == dnnl_u8
                           || mem.get_desc().get_data_type() == dnnl_s8
                           || mem.get_desc().get_data_type() == dnnl_s4
                           || mem.get_desc().get_data_type() == dnnl_u4)
                && std::is_same<T, unsigned>::value) {
            memory mem_u32_mem(
                    {mem.get_desc().get_dims(), memory::data_type::s32,
                            mem.get_desc().get_strides()},
                    eng);
            write_to_dnnl_memory<unsigned>(
                    (const unsigned *)handle, mem_u32_mem);
            dnnl::reorder(mem_u32_mem, mem).execute(s, mem_u32_mem, mem);
            s.wait();
        } else if ((mem.get_desc().get_data_type() == dnnl_f32
                           && std::is_same<T, float>::value)
                || (mem.get_desc().get_data_type() == dnnl_s32
                        && std::is_same<T, int>::value)) {
            void *mapped_ptr = mem.map_data();
            if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
            mem.unmap_data(mapped_ptr);
        } else {
            // PC: this branch is identical to the one above
            void *mapped_ptr = mem.map_data();
            if (mapped_ptr) std::memcpy(mapped_ptr, handle, size);
            mem.unmap_data(mapped_ptr);
        }
        return;
    }
#endif

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}



gmlp_tensors get_descriptors(dnnl::engine &eng, mlp_dims_t p) {
    gmlp_tensors out;

    // Prepare input and output shapes to construct the swiglu graph.
    const memory::dims O_proj_sz  = {p.mb, p.ic};
    const memory::dims W_gate_sz  = {p.ic, p.oc};
    const memory::dims W_up_sz    = {p.ic, p.oc};
    const memory::dims W_down_sz  = {p.oc, p.ic};
    //const memory::dims FC_gate_sz = {p.oc, p.mb};
    const memory::dims FC_gate_sz = {p.mb, p.oc};
    const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    const memory::dims FC_up_sz = {p.mb, p.oc};
    const memory::dims FC_down_sz = {p.mb, p.ic};

    // memory::dims scale_gateup_sz = {p.ic, p.oc };
    const memory::dims scales_gateup_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims { 1, 1, 1, 1 };
            case quantize_type::per_token_with_groups: // 1 scale for entire row
                return memory::dims { W_gate_sz[0] / p.gateup_group_size, W_gate_sz[1]};
            case quantize_type::per_token:
                return memory::dims { W_gate_sz[0], 1 };
            case quantize_type::per_tensor: return memory::dims { 1, 1 };
        }
        throw "wat?\n";
    }();


    auto dt = memory::data_type::f16;
    auto wgu_wt    = (p.wgu_wt == mdt::undef) ?    dt : p.wgu_wt;   ///undef = non quantized , can be u8
    auto wgu_s_dt  = (p.wgu_s_dt == mdt::undef) ?  dt : p.wgu_s_dt;
    auto wgu_zp_dt = (p.wgu_zp_dt == mdt::undef) ? dt : p.wgu_zp_dt;

    auto wd_wt    = (p.wd_wt == mdt::undef) ?    dt : p.wd_wt;
    auto wd_s_dt  = (p.wd_s_dt == mdt::undef) ?  dt : p.wd_s_dt;
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
    out.m_x      = memory(x_md, eng);
    out.m_w_gate = memory(w_gate_md, eng);
    out.m_w_up   = memory(w_up_md, eng);
    out.m_w_down = memory(w_down_md, eng);

    //out.m_x_quantized      = memory(x_qnt_md, eng);
    out.m_w_gate_quantized = memory(w_gate_qnt_md, eng);
    out.m_w_up_quantized   = memory(w_up_qnt_md, eng);
    out.m_w_down_quantized = memory(w_down_qnt_md, eng);

    //out.m_x_scale      = memory(x_scale_md, eng);
    out.m_w_gate_scales = memory(w_gate_scales_md, eng);
    out.m_w_up_scales   = memory(w_up_scales_md, eng);
    out.m_w_down_scales = memory(w_down_scales_md, eng);

    //out.m_x_zp      = memory(x_zp_md, eng);
    out.m_w_gate_zp = memory(w_gate_zp_md, eng);
    out.m_w_up_zp   = memory(w_up_zp_md, eng);
    out.m_w_down_zp = memory(w_down_zp_md, eng);

    out.m_fc_gate       = memory(FC_gate_md, eng);
    out.m_fc_up         = memory(FC_up_md, eng);
    out.m_fc_down       = memory(FC_down_md, eng);

    out.m_out           = memory(output_md, eng);
    out.m_out_quantized = memory(output_qnt_md, eng);

    out.m_fc_gate_t     = memory(FC_gate_md_t, eng);

    // Allocate user data.
    std::vector<float>x_data(product(O_proj_sz));
    std::vector<float>w_gate_data(product(W_gate_sz));
    std::vector<float>w_up_data(product(W_up_sz));
    std::vector<float>w_down_data(product(W_down_sz));

    // std::vector<float>x_quantized_data(product(O_proj_sz), 1.f);
    std::vector<float>w_gate_quantized_data(product(W_gate_sz), 1.f);
    std::vector<float>w_up_quantized_data(product(W_up_sz), 1.f);
    std::vector<float>w_down_quantized_data(product(W_down_sz), 1.f);

    // std::vector<float>x_scale_data(product(O_proj_sz), 1.f);
    std::vector<float>w_gate_scales_data(product(W_gate_sz), 1.f);
    std::vector<float>w_up_scales_data(product(W_up_sz), 1.f);
    std::vector<float>w_down_scales_data(product(W_down_sz), 1.f);

    // std::vector<int>x_zp_data_signed(product(O_proj_sz), 0);
    std::vector<int>w_gate_zp_data_signed(product(W_gate_sz), 0);
    std::vector<int>w_up_zp_data_signed(product(W_up_sz), 0);
    std::vector<int>w_down_zp_data_signed(product(W_down_sz), 0);

    // std::vector<unsigned>x_zp_data_unsigned(product(O_proj_sz), 0);
    std::vector<unsigned>w_gate_zp_data_unsigned(product(W_gate_sz), 0);
    std::vector<unsigned>w_up_zp_data_unsigned(product(W_up_sz), 0);
    std::vector<unsigned>w_down_zp_data_unsigned(product(W_down_sz), 0);

    switch (p.qtype) {
        case quantize_type::per_token_with_groups: {
            int gateup_mask = 1 << 1 | 1 << 0;
            int down_mask   = 1 << 1 | 1 << 0;

            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef)
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, gateup_mask, {p.gateup_group_size, 1}, wgu_s_dt);
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef)
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, gateup_mask, {p.gateup_group_size, 1}, wgu_zp_dt);

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef)
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, down_mask, {p.down_group_size, 1}, wd_s_dt);
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef)
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, down_mask, {p.down_group_size, 1}, wd_zp_dt);
        } break;
        case quantize_type::per_token: {
            // faster dim | slower dim
            // dim 3, 2,  {1 , 0}
            int gateup_mask = 1; //  1 << 0;
            int down_mask   = 1; //  1 << 0;

            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef)
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, gateup_mask, {}, wgu_s_dt);
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef)
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, gateup_mask, {}, wgu_zp_dt);

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef)
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, down_mask, {}, wd_s_dt);
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef)
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, down_mask, {}, wd_zp_dt);
        } break;
        case quantize_type::per_tensor:
            if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef)
                out.gateup_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, 0, {}, wgu_s_dt);
            if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef)
                out.gateup_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, 0, {}, wgu_zp_dt);

            if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef)
                out.down_attr_quantized.set_scales(
                        DNNL_ARG_WEIGHTS, 0, {}, wd_s_dt);
            if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef)
                out.down_attr_quantized.set_zero_points(
                        DNNL_ARG_WEIGHTS, 0, {}, wd_zp_dt);
            break;
        case quantize_type::no_quantization: break;
    }

    fill_random(x_data);
    //fill_lin(x_data); //testdata

    fill_random_quantized(
            w_gate_quantized_data, (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
    fill_random_quantized(
            w_up_quantized_data, (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));
    fill_random_quantized(
            w_down_quantized_data, (wgu_wt == mdt::u4 || wgu_wt == mdt::u8));

    if (p.qtype != quantize_type::no_quantization) {
        if (wgu_wt != mdt::f16 && wgu_s_dt != mdt::undef) {
            fill_random_scales(w_gate_scales_data);
            //w_gate_scales_data[63] = 2.f;
            fill_random_scales(w_up_scales_data);
        }
        //if (qtype == quantize_type::per_token) {
        if (wgu_wt != mdt::f16 && wgu_zp_dt != mdt::undef) {
            fill_random_quantized(w_gate_zp_data_signed);
            fill_random_quantized(w_gate_zp_data_unsigned);
            fill_random_quantized(w_up_zp_data_signed);
            fill_random_quantized(w_up_zp_data_unsigned);
        }
        //}
        if (wd_wt != mdt::f16 && wd_s_dt != mdt::undef)
            fill_random_scales(w_down_scales_data);
        if (wd_wt != mdt::f16 && wd_zp_dt != mdt::undef) {
            fill_random_quantized(w_down_zp_data_signed);
            fill_random_quantized(w_down_zp_data_unsigned);
        }
    }

    int wgu_group_size = p.gateup_group_size;
    int wd_group_size  = p.down_group_size;

    if (p.qtype == quantize_type::per_tensor) {
        wgu_group_size = W_gate_sz[0] * W_gate_sz[1];
        wd_group_size  = W_down_sz[0] * W_down_sz[1];
    }

    //vector<float> x_data, w_gate_data, w_up_data, w_down_data;
    //if(p.qtype == quantize_type::no_quantization) {
    if(!p.do_quantize) {
        printf("no quant init\n");
        fill_random(w_gate_data);
        //fill_hceye(w_gate_data, p.ic); //testdata

        fill_random(w_up_data);
        fill_random(w_down_data);
    } else {
        if (wgu_wt == mdt::s4 || wgu_wt == mdt::s8) {
            printf("s4/s8 quant init\n");
            w_gate_data = dequantize(w_gate_quantized_data, w_gate_md,
                   out.gateup_attr_quantized, w_gate_scales_md, w_gate_zp_data_signed,
                   w_gate_scales_data, wgu_group_size, p.qtype);

            w_up_data = dequantize(w_up_quantized_data, w_up_md,
                   out.gateup_attr_quantized, w_up_scales_md, w_up_zp_data_signed,
                   w_up_scales_data, wgu_group_size, p.qtype);

            w_down_data = dequantize(w_down_quantized_data, w_down_md,
                   out.down_attr_quantized, w_down_scales_md, w_down_zp_data_signed,
                   w_down_scales_data, wgu_group_size, p.qtype);
        } else {
            printf("quant init\n");
            w_gate_data = dequantize(w_gate_quantized_data, w_gate_md,
                   out.gateup_attr_quantized, w_gate_scales_md, w_gate_zp_data_unsigned,
                   w_gate_scales_data, wgu_group_size, p.qtype);
            w_up_data = dequantize(w_up_quantized_data, w_up_md,
                   out.gateup_attr_quantized, w_up_scales_md, w_up_zp_data_unsigned,
                   w_up_scales_data, wgu_group_size, p.qtype);
            w_down_data = dequantize(w_down_quantized_data, w_down_md,
                   out.down_attr_quantized, w_down_scales_md, w_down_zp_data_unsigned,
                   w_down_scales_data, wgu_group_size, p.qtype);
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

    printf("memory data types?? %d %d %d\n", out.m_w_gate_scales.get_desc().get_data_type(),
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

void print_mem(const dnnl::memory &mem, std::string name = "") {
    auto eng = mem.get_engine();
    dnnl::stream s(eng);
    s.wait();
    auto desc = mem.get_desc();
    auto dims_og = desc.get_dims();
    auto strides_og = desc.get_strides();
    std::vector<int> dims    = { 1, 1, dims_og[0], dims_og[1] };
    std::vector<int> strides = { 0, 0, strides_og[0], strides_og[1] };
    printf("%sbegin-   ", name.c_str());
    printf("dims[%zu]: {%d, %d, %d, %d}  ", dims.size(), dims[0], dims[1],
            dims[2], dims[3]);
    printf("strides[%zu]: {%d, %d, %d, %d}\n", strides.size(), strides[0],
            strides[1], strides[2], strides[3]);
    void *mapped_ptr_ = (void *)mem.map_data();
    printf("    i:    ");
    for (int i = 0; i < dims[3]; i++) {
        switch ((int)desc.get_data_type()) {
            case dnnl_u4:
            case dnnl_s4: printf("%4d", i); break;
            case dnnl_u8:
            case dnnl_s8: printf("%4d", i); break;
            case dnnl_f32:
            case dnnl_f16: printf("%9d", i); break;
        }
    }
    printf("\n");

    switch ((int)desc.get_data_type()) {
        case dnnl_u4:
        case dnnl_s4: {
            char *mapped_ptr = (char *)mapped_ptr_;
            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d):", l, k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            auto offset = l * strides[0] + k * strides[1]
                                    + j * strides[2] + i * strides[3];
                            offset /= 2;
                            bool is_odd = (strides[3] == 1) ? i % 2 : j % 2;
                            if (is_odd) {
                                int bits = (mapped_ptr[offset] & 0xf0) >> 4;
                                if (desc.get_data_type() == dnnl_s4) {
                                    int sign = (bits & 0x08) ? -1 : 1;
                                    if (sign == -1) {
                                        bits = (bits & 0x07) - 8;
                                    } else {
                                        bits = (bits & 0x07);
                                    }
                                }

                                printf("%4d", bits);
                            } else {
                                int bits = (mapped_ptr[offset] & 0x0f);
                                if (desc.get_data_type() == dnnl_s4) {
                                    int sign = (bits & 0x08) ? -1 : 1;
                                    if (sign == -1) {
                                        bits = (bits & 0x07) - 8;
                                    } else {
                                        bits = (bits & 0x07);
                                    }
                                }

                                printf("%4d", bits);
                            }
                        }
                        printf("\n");
                    }
                }
            }
        } break;

        case dnnl_u8:
        case dnnl_s8: {
            char *mapped_ptr = (char *)mapped_ptr_;
            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d): ", l, k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            printf("%4d",
                                    (mapped_ptr[l * strides[0] + k * strides[1]
                                            + j * strides[2]
                                            + i * strides[3]]));
                        }
                        printf("\n");
                    }
                }
            }
        } break;
        case dnnl_f16: {
            using dnnl::impl::float16_t;
            float16_t *mapped_ptr = (float16_t *)mapped_ptr_;

            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %3d):", k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            printf("%+7.2f",
                                    (mapped_ptr[l * strides[0] + k * strides[1]
                                            + j * strides[2] + i * strides[3]]
                                                    .f()));
                        }
                        printf("\n");
                    }
                }
            }
        } break;
        case dnnl_f32: {
            using dnnl::impl::float16_t;
            float *mapped_ptr = (float *)mapped_ptr_;

            for (int l = 0; l < dims[0]; l++) {
                for (int k = 0; k < dims[1]; k++) {
                    for (int j = 0; j < dims[2]; j++) {
                        printf("(%2d, %2d, %3d):", l, k, j);
                        for (int i = 0; i < dims[3]; i++) {
                            printf("%+9.3f",
                                    (mapped_ptr[l * strides[0] + k * strides[1]
                                            + j * strides[2]
                                            + i * strides[3]]));
                        }
                        printf("\n");
                    }
                }
            }
        } break;
        default: throw std::runtime_error("Not supported");
    }
    mem.unmap_data(mapped_ptr_);
    printf("%send\n", name.c_str());
}

void print_test_case(memory::data_type dt, const mlp_dims_t &p) {
    std::cout << '[' << std::setw(4) << dnnl_dt2str(memory::convert_to_c(dt));
    std::cout << " mb = " << p.mb << ", ic = " << p.ic << ", oc = " << p.oc;
    std::cout << "] " << std::flush;
}

template<typename T>
void bench_gated_mlp_primitives(std::vector<T> &res, double &avg_time,
        gmlp_tensors &t,
        dnnl::engine &eng, dnnl::stream &strm,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);

    // extract memory objects
    auto m_O_proj  = t.m_x;
    auto m_W_gate  = t.m_w_gate;
    auto m_W_up    = t.m_w_up;
    auto m_W_down  = t.m_w_down;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up   = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    // extract memory descriptors
    auto O_proj_md  = t.m_x.get_desc();
    auto W_gate_md  = t.m_w_gate.get_desc();
    auto W_up_md    = t.m_w_up.get_desc();
    auto W_down_md  = t.m_w_down.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md   = t.m_fc_up.get_desc();
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
    runs = 10;
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

    if (verbose && product(FC_down_md.get_dims()) < (64*64) + 1) {
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

    if (verbose && product(FC_down_md.get_dims()) < (64*64) + 1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("------tmpres\n");
        print_mem(m_FC_gate_t, "-prim");
    }

    float16_t *mapped_ptr_f16 = (float16_t *)m_FC_gate_t.map_data();
    res.resize(product(FC_gate_md.get_dims()));
    for(int i=0; i<res.size(); ++i){
        res[i] = mapped_ptr_f16[i];
    }
    m_FC_gate_t.unmap_data(mapped_ptr_f16);
#else
    if (verbose && product(FC_down_md.get_dims()) < (64*64) + 1) {
        const memory::dims FC_down_sz = {p.mb, p.ic};
        printf("------tmpres\n");
        print_mem(m_FC_gate, "-prim");
    }

    float16_t *mapped_ptr_f16 = (float16_t *)m_FC_gate.map_data();
    res.resize(product(FC_gate_md.get_dims()));
    for(int i=0; i<res.size(); ++i){
        res[i] = mapped_ptr_f16[i];
    }
    m_FC_gate.unmap_data(mapped_ptr_f16);
#endif

}

template<typename T>
void bench_gated_mlp_internal(std::vector<T> &res, double &avg_time,
        gmlp_tensors &t,
        dnnl::engine &eng, dnnl::stream strm,
        const mlp_dims_t &p, double time_limit = 0.) {
    printf("eng?\n");
    const bool quick_test = (time_limit == 0.);

    // Create memory objects
    auto m_O_proj  = t.m_x;
    auto m_W_gate  = t.m_w_gate;
    auto m_W_up    = t.m_w_up;
    auto m_W_down  = t.m_w_down;
    auto m_FC_gate = t.m_fc_gate;
    auto m_FC_up   = t.m_fc_up;
    auto m_FC_down = t.m_fc_down;

    // Create memory objects
    auto O_proj_md  = t.m_x.get_desc();
    auto W_gate_md  = t.m_w_gate.get_desc();
    auto W_up_md    = t.m_w_up.get_desc();
    auto W_down_md  = t.m_w_down.get_desc();
    auto FC_gate_md = t.m_fc_gate.get_desc();
    auto FC_up_md   = t.m_fc_up.get_desc();
    auto FC_down_md = t.m_fc_down.get_desc();

    // quantization memory
    auto m_W_gate_quant  = t.m_w_gate_quantized;
    auto m_W_gate_scales = t.m_w_gate_scales;
    auto m_W_gate_zp     = t.m_w_gate_zp;
    auto m_W_up_quant    = t.m_w_up_quantized;
    auto m_W_up_scales   = t.m_w_up_scales;
    auto m_W_up_zp       = t.m_w_up_zp;

    auto m_W_gate_quant_md  = t.m_w_gate_quantized.get_desc();
    auto m_W_gate_scales_md = t.m_w_gate_scales.get_desc();
    auto m_W_gate_zp_md     = t.m_w_gate_zp.get_desc();
    auto m_W_up_quant_md    = t.m_w_up_quantized.get_desc();
    auto m_W_up_scales_md   = t.m_w_up_scales.get_desc();
    auto m_W_up_zp_md       = t.m_w_up_zp.get_desc();

    const memory::dims FC_gate_sz_t = {p.oc, p.mb};
    //const memory::dims FC_gate_sz_t = {p.mb, p.oc};
    auto FC_gate_md_t = memory::desc(FC_gate_sz_t, FC_gate_md.get_data_type(), tag::ab);
    auto m_FC_gate_t = memory(FC_gate_md_t, eng);

    if(verbose) {
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
    using namespace dnnl::experimental;
    primitive_attr gmlp_attr, gate_attr, up_attr, down_attr;
                    //DNNL_ARG_ATTR_SCALES | DNNL_ARG_WTS_GATE,
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            //wts_gate scale+zp
            gate_attr.set_scales(DNNL_ARG_SRC_0,
                                 (1<<0) + (1<<1), {1, p.gateup_group_size}, m_W_gate_scales_md.get_data_type());
            gate_attr.set_zero_points(DNNL_ARG_WEIGHTS, //TODO: wat??? why no arg_src0,1,2,3...
                                      (1<<0) + (1<<1), {1, p.gateup_group_size}, m_W_gate_zp_md.get_data_type());
            //wts_up scale+zp
            up_attr.set_scales(DNNL_ARG_SRC_1,
                               (1<<0) + (1<<1), {1, p.gateup_group_size}, m_W_up_scales_md.get_data_type());
            up_attr.set_zero_points(DNNL_ARG_WEIGHTS,
                                    (1<<0) + (1<<1), {1, p.gateup_group_size}, m_W_up_zp_md.get_data_type());

            break;
        case quantize_type::per_tensor:
            //wts_gate scale+zp
            gate_attr.set_scales(DNNL_ARG_SRC_0,
                                 0, {1, p.gateup_group_size}, m_W_gate_scales_md.get_data_type());
            gate_attr.set_zero_points(DNNL_ARG_WEIGHTS, //TODO: wat??? why no arg_src0,1,2,3...
                                      0, {1, p.gateup_group_size}, m_W_gate_zp_md.get_data_type());
            //wts_up scale+zp
            up_attr.set_scales(DNNL_ARG_SRC_1,
                               0, {1, p.gateup_group_size}, m_W_up_scales_md.get_data_type());
            up_attr.set_zero_points(DNNL_ARG_WEIGHTS,
                                    0, {1, p.gateup_group_size}, m_W_up_zp_md.get_data_type());

            break;

        case quantize_type::no_quantization: break;
        default: break;
    }

    auto gmlp_pd = [&](){
        if(p.do_quantize) {
            //TODO: W_up_md_quant
            return gmlp::primitive_desc(eng, O_proj_md, m_W_gate_quant_md, m_W_up_quant_md,
                W_down_md, FC_gate_md_t, gmlp_attr, gate_attr, up_attr);
        } else {
            return gmlp::primitive_desc(eng, O_proj_md, W_gate_md, W_up_md,
                W_down_md, FC_gate_md_t, gmlp_attr);
        }
    }();

    auto prim_fused_internal = gmlp(gmlp_pd);

    const auto loop = [&]() {
        if(p.do_quantize) {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC_0, m_O_proj},
                     {DNNL_ARG_SRC_1, m_W_gate_quant},
                     {DNNL_ARG_SRC_2, m_W_up_quant},
                     {DNNL_ARG_SRC_3, m_W_down},
                     //{DNNL_ARG_DST, m_FC_down}}); //CORRECT ARG
                     {DNNL_ARG_DST, m_FC_gate_t}
                     , {DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_SCALES, m_W_gate_scales}
                     , {DNNL_ARG_WTS_GATE | DNNL_ARG_ATTR_ZERO_POINTS, m_W_gate_zp}
                     , {DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_SCALES, m_W_up_scales}
                     , {DNNL_ARG_WTS_UP | DNNL_ARG_ATTR_ZERO_POINTS, m_W_up_zp}
                });   // TMP ARG for mm test
        } else {
            prim_fused_internal.execute(strm,
                    {{DNNL_ARG_SRC_0, m_O_proj},
                     {DNNL_ARG_SRC_1, m_W_gate},
                     {DNNL_ARG_SRC_2, m_W_up},
                     {DNNL_ARG_SRC_3, m_W_down},
                     //{DNNL_ARG_DST, m_FC_down}}); //CORRECT ARG
                     {DNNL_ARG_DST, m_FC_gate_t}
                });   // TMP ARG for mm test
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
    runs = 10;
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

    if (verbose && product(FC_down_md.get_dims()) < (64*64)+1) {
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
    for(int i=0; i<res.size(); ++i) {
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

void bench_gated_mlp_graph(engine::kind ekind, logical_tensor::data_type dt,
        const mlp_dims_t &p, double time_limit = 0.) {
    const bool quick_test = (time_limit == 0.);
    print_test_case(dt, p);

    allocator alloc = create_allocator(ekind);

    // Create execution dnnl::engine.
    dnnl::engine eng = make_engine_with_allocator(ekind, 0, alloc);
    // Create dnnl::stream.
    dnnl::stream strm(eng);

    // input shape
    const dims src_sz = {p.mb, p.ic};
    // weight0/weight1 shape: fc_gate and fc_up
    const dims wei0_sz = {p.ic, p.oc};
    // hidden shape
    const dims hd_sz = {p.mb, p.oc};
    // weight2 shape: fc_down
    const dims wei2_sz = {p.oc, p.ic};
    // output shape
    const dims out_sz = {p.mb, p.ic};

    // Incremental IDs used to create logical tensors and operations.
    size_t id = 0;

    // fc_gate
    auto src = logical_tensor(id++, dt, src_sz, layout_type::strided);
    auto wei0 = logical_tensor(id++, dt, wei0_sz, layout_type::strided);
    auto out0 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto fc_gate = op(id++, op::kind::MatMul, "fc_gate");
    fc_gate.add_inputs({src, wei0});
    fc_gate.add_outputs({out0});

    // fc_up
    auto wei1 = logical_tensor(id++, dt, wei0_sz, layout_type::strided);
    auto out1 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto fc_up = op(id++, op::kind::MatMul, "fc_up");
    fc_up.add_inputs({src, wei1});
    fc_up.add_outputs({out1});

    // activation: swish
    auto out2 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto act = op(id++, op::kind::HardSwish, "swish");
    act.add_inputs({out0});
    act.add_outputs({out2});

    // multiplication
    auto out3 = logical_tensor(id++, dt, hd_sz, layout_type::strided);
    auto mul = op(id++, op::kind::Multiply, "mul");
    mul.add_inputs({out2, out1});
    mul.add_outputs({out3});

    // fc_down
    auto wei2 = logical_tensor(id++, dt, wei2_sz, layout_type::strided);
    auto dst = logical_tensor(id++, dt, out_sz, layout_type::strided);
    auto fc_down = op(id++, op::kind::MatMul, "fc_down");
    fc_down.add_inputs({out3, wei2});
    fc_down.add_outputs({dst});

    // Construct a gated mlp graph with engine kind and operations.
    dnnl::graph::graph mlp(ekind);
    mlp.add_op(fc_gate);
    mlp.add_op(fc_up);
    mlp.add_op(act);
    mlp.add_op(mul);
    mlp.add_op(fc_down);
    mlp.finalize();

    // Get partitions from the mlp graph.
    std::vector<partition> partitions = mlp.get_partitions();
    // This is just for oneDNN testing purpose.
    if (partitions.size() != 1) {
        std::cout << "unsupported mlp" << std::endl;
        return;
    }

    // Compile the partition with inputs, outputs, and an engine.
    compiled_partition cp
            = partitions[0].compile({src, wei0, wei1, wei2}, {dst}, eng);

    // Create tensor objects
    auto ts_src = tensor(src, eng);
    auto ts_wei0 = tensor(wei0, eng);
    auto ts_wei1 = tensor(wei1, eng);
    auto ts_wei2 = tensor(wei2, eng);
    auto ts_dst = tensor(dst, eng);

    // Allocate user data.
    std::vector<float> src_data(product(src_sz));
    std::vector<float> wei0_data(product(wei0_sz));
    std::vector<float> wei1_data(product(wei0_sz));
    std::vector<float> wei2_data(product(wei2_sz));

    fill_random(src_data);
    fill_random(wei0_data);
    fill_random(wei1_data);
    fill_random(wei2_data);

    // Write data to tensor object's handle.
    write_to_dnnl_tensor(src_data.data(), ts_src);
    write_to_dnnl_tensor(wei0_data.data(), ts_wei0);
    write_to_dnnl_tensor(wei1_data.data(), ts_wei1);
    write_to_dnnl_tensor(wei2_data.data(), ts_wei2);

    // Warmup run.
    // Execute the compiled partition of mqa.
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});

    // Wait for the computation to finish.
    strm.wait();

    // First run.
    auto start_first = std::chrono::steady_clock::now();
    cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    strm.wait();
    auto end_first = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> dur_first
            = end_first - start_first;

    if (quick_test) return;

    // Timing runs.
    int runs = std::max(min_runs, int(time_limit / dur_first.count()));
    runs = 10;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i <= runs; i++) {
        cp.execute(strm, {ts_src, ts_wei0, ts_wei1, ts_wei2}, {ts_dst});
    }
    strm.wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Display the results.
    double avg_time = (duration.count() - dur_first.count()) / runs;
    std::cout << "graph runs: " << runs + 1 << "; ";
    std::cout << "avg_time: " << avg_time << " ms" << std::endl;

    /*
    if (product(out_sz) < 128) {
        std::vector<float> res(product(out_sz));
        read_from_dnnl_tensor(res.data(), ts_dst);

        for (int y = 0; y < p.mb; ++y) {
            for (int x = 0; x < p.ic; ++x) {
                printf("%f ", res[y * p.ic + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
}

template<typename T>
void generate_input_vectors(mlp_dims_t p, std::vector<T> &x_data, std::vector<T> &w_gate_data, std::vector<T> &w_up_data, std::vector<T> &w_down_data) {

    const memory::dims O_proj_sz = {p.mb, p.ic};
    const memory::dims W_gate_sz = {p.ic, p.oc};
    const memory::dims W_up_sz = {p.ic, p.oc};
    const memory::dims W_down_sz = {p.oc, p.ic};

    x_data.resize(product(O_proj_sz));
    w_gate_data.resize(product(W_gate_sz));
    w_up_data.resize(product(W_up_sz));
    w_down_data.resize(product(W_down_sz));

    fill_random(x_data);
    //fill_const(x_data, 0.01f);
    //fill_hceye(x_data, p.ic);

    fill_random(w_gate_data);
    //fill_hceye(w_gate_data, p.oc);
    //fill_hceye(w_gate_data);

    fill_const(w_up_data, 0.01f);
    //fill_random(w_down_data);
    fill_const(w_down_data, 0.1f);
    //fill_hceye(w_down_data);

    /*
    printf("\n----------\n\nGENDATA\n");
    for (int y = 0; y < p.mb; ++y) {
        for (int x = 0; x < p.ic; ++x) {
            if constexpr(std::is_same<half, T>::value) {
                printf("%5.1f ", half_cast<float>(x_data[y * p.ic + x]));
            } else {
                printf("%f ", x_data[y * p.ic + x]);
            }
        }
        printf("\n");
    }
    printf("inpB----------[%d %d]\n", p.ic, p.oc);
    for (int y = 0; y < p.ic; ++y) {
        for (int x = 0; x < p.oc; ++x) {
            if constexpr(std::is_same<half, T>::value) {
                printf("%5.1f ", half_cast<float>(w_gate_data[y * p.oc + x]));
            } else {
                printf("%f ", w_gate_data[y * p.oc + x]);
            }
        }
        printf("\n");
    }
    printf("----------\n");
    */

}

void bad_args() {
    std::cerr << "Usage: graph-gated-mlp-cpp [cpu|gpu]\n"
                 "       graph-gated-mlp-cpp [cpu|gpu] <mb> <ic> <oc>\n\n";
    throw std::invalid_argument("Incorrect input arguments.");
}

enum class api_kind {
    primitive,
    graph,
    internal_hack
};

template<typename T>
void bench(std::vector<T> &res, double &avg_time,
        gmlp_tensors &t,
        api_kind api,
        dnnl::engine &eng, dnnl::stream &strm,
        const mlp_dims_t &p, double time_limit = 0.) {

    try {
        if (api == api_kind::primitive) {
            bench_gated_mlp_primitives(res, avg_time,
                    t,
                    eng, strm, p, time_limit);
            strm.wait();
        } else if (api == api_kind::graph) {
            //bench_gated_mlp_graph(ekind, static_cast<logical_tensor::data_type>(dt),
                    //p, time_limit);
            //get_mem_pool().clear();
        } else {
            bench_gated_mlp_internal(res, avg_time,
                    t,
                    eng, strm, p, time_limit);
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

void mlp_perf(engine::kind ekind, int argc, char **argv) {
    // default testing parameters
    memory::data_type f16 = static_cast<memory::data_type>(dnnl_f16);
    mlp_dims_t params = {8, 4096, 14336, false, 128, 128, f16, f16, f16, f16, f16, f16, quantize_type::no_quantization};

    if (argc > 2) {
        if (argc == 5) { // 6? which ones? TODO: asktao
            params.mb = std::atoi(argv[2]);
            params.ic = std::atoi(argv[3]);
            params.oc = std::atoi(argv[4]);
        } else {
            bad_args();
        }

        if (params.mb <= 0 || params.ic <= 0 || params.oc <= 0) { bad_args(); }
    }

    //std::vector<float> x_data, w_gate_data, w_up_data, w_down_data;
    //generate_input_vectors<float>(params, x_data, w_gate_data, w_up_data, w_down_data);

#define TEST_QUANTIZE 0
#if TEST_QUANTIZE

    params.do_quantize = true;
    params.gateup_group_size = 8; //example, todo change
    params.down_group_size = 32;

    //params.wgu_wt    = static_cast<memory::data_type>(dnnl_u4);
    params.wgu_wt    = static_cast<memory::data_type>(dnnl_u8);
    params.wgu_s_dt  = static_cast<memory::data_type>(dnnl_f16);
    params.wgu_zp_dt = static_cast<memory::data_type>(dnnl_u8);

    //params.wd_wt    = static_cast<memory::data_type>(dnnl_u4);
    params.wd_wt    = static_cast<memory::data_type>(dnnl_u8);
    params.wd_s_dt  = static_cast<memory::data_type>(dnnl_f16);
    params.wd_zp_dt = static_cast<memory::data_type>(dnnl_u8);

    params.qtype = quantize_type::per_token_with_groups;
    //params.qtype = quantize_type::per_tensor;
    //TODO: test last possible mask?

#else

    params.qtype = quantize_type::no_quantization;
    params.do_quantize = false;
    params.gateup_group_size = 1;
    params.down_group_size = 1;

    params.wgu_wt    = static_cast<memory::data_type>(dnnl_f16);
    params.wgu_s_dt  = static_cast<memory::data_type>(dnnl_f16);
    params.wgu_zp_dt = static_cast<memory::data_type>(dnnl_f16);

    params.wd_wt    = static_cast<memory::data_type>(dnnl_f16);
    params.wd_s_dt  = static_cast<memory::data_type>(dnnl_f16);
    params.wd_zp_dt = static_cast<memory::data_type>(dnnl_f16);

    params.qtype = quantize_type::per_token; //remove? needs some additional logic in get_descriptors

#endif

    printf("eng?\n");
    // Create execution dnnl::engine.
    dnnl::engine eng(ekind, 0);
    dnnl::stream strm = dnnl::stream(eng);
    printf("eng?create\n");

    auto tensors = get_descriptors(eng, params);
    printf("get desc\n");
    //print_mem(tensors.m_w_gate_quantized, "gen_m_w_gate_quant");
    //print_mem(tensors.m_w_gate_scales, "gen_m_w_gate_scales");
    //print_mem(tensors.m_w_gate_zp, "gen_m_w_gate_zp");

    //bench(ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(ekind, dnnl_f16, params, 2000.0 /*ms*/);

    //TODO: merge w/existing graph PR
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_f16, params, 2000.0 /*ms*/);
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);

    //printf("GRAPH\n");
    //bench(api_kind::graph, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    std::vector<float> resp, resi;
    std::vector<half> resph, resih;
    double avg_time_int, avg_time_prim;

    printf("PRIMITIVE\n");
    //bench(resp, api_kind::primitive, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(resph, api_kind::primitive, ekind, dnnl_f16, dnnl_f16, params, 2000.0 /*ms*/);
    bench(resph, avg_time_prim,
          tensors,
          api_kind::primitive, eng, strm, params, 2000.0 /*ms*/);

    printf("INTERNAL\n");
    //bench(resi, api_kind::internal_hack, ekind, dnnl_f32, params, 2000.0 /*ms*/);
    //bench(resih, api_kind::internal_hack, ekind, dnnl_f16, dnnl_f16, params, 2000.0 /*ms*/);
    bench(resih, avg_time_int,
          tensors,
          api_kind::internal_hack, eng, strm, params, 2000.0 /*ms*/);

    /*
    if(resi.size() == 0) printf("[WARNING] Empty output! internal kernel fail X_X\n");
    for(int i=0; i<resi.size(); ++i) {
        if(std::abs((resi[i] - resp[i]) / resi[i]) > 1e-2) printf("mismatch @ %d, %f != %f\n", i, resi[i], resp[i]); //TODO: improve
    }
    */

    if(resih.size() == 0) printf("[WARNING] Empty output! internal kernel fail X_X\n");
    int n_mismatches=0, n_matches=0;
    printf("resih.size() %zu\n", resih.size());
    for(int i=0; i<resih.size(); ++i) {
        if(std::abs((resih[i] - resph[i]) / resih[i]) > 5e-3) {
            n_mismatches++;
            if(n_mismatches < 10)
                printf("mismatch @ %d, %f != %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
        } else {
            if(std::abs(half_cast<float>(resih[i])) > 5e-3) {
                n_matches++;
                if(n_matches < 10)
                    printf("vs @ %d, %f == %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
            }
        }
    }
    printf("total mismatches: %d \n", n_mismatches);
    printf("avg time internal: %f vs %f avg time primitive, w/speedup of %f\n", avg_time_int, avg_time_prim, avg_time_prim / avg_time_int);
    //bench(api_kind::primitive, ekind, dnnl_bf16, params, 2000.0 /*ms*/);
    //bench(api_kind::primitive, ekind, dnnl_f16, params, 2000.0 /*ms*/);
}

/*
int main(int argc, char **argv) {
    return handle_example_errors(
            mlp_perf, parse_engine_kind(argc, argv, 4), argc, argv);
}
*/






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
    std::vector<half> resph, resih;
    double avg_time_int, avg_time_prim;

    printf("PRIMITIVE\n");
    bench(resph, avg_time_prim,
          tensors,
          api_kind::primitive, eng, strm, params, 2000.0 /*ms*/);

    printf("INTERNAL\n");
    bench(resih, avg_time_int,
          tensors,
          api_kind::internal_hack, eng, strm, params, 2000.0 /*ms*/);

    if(resih.size() == 0) {
        printf("[WARNING] Empty output! internal kernel fail X_X\n");
        EXPECT_TRUE(false);
    }
    int n_mismatches=0, n_matches=0;
    printf("resih.size() %zu\n", resih.size());
    float max_diff = 0.0f, max_val, max_gold;
    for(int i=0; i<resih.size(); ++i) {
        float abs_diff = std::abs(resih[i] - resph[i]);
        float rel_diff = std::abs((resih[i] - resph[i]) / resih[i]);
        if(abs_diff > 1e-4 && rel_diff > 5e-3) {

            if (isfinite(rel_diff) && (abs_diff) > max_diff) {
                max_diff = abs_diff;
                max_val  = resih[i];
                max_gold = resph[i];
            }

            n_mismatches++;
            if(n_mismatches < 10)
                printf("mismatch @ %d, %f != %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
        } else {
            if(std::abs(half_cast<float>(resih[i])) > 5e-3) {
                n_matches++;
                if(n_matches < 10)
                    printf("vs @ %d, %f == %f\n", i, half_cast<float>(resih[i]), half_cast<float>(resph[i])); //TODO: improve
            }
        }
    }
    printf("total mismatches: %d \n", n_mismatches);
    printf("avg time internal: %f vs %f avg time primitive, w/speedup of %f\n", avg_time_int, avg_time_prim, avg_time_prim / avg_time_int);

    int total_size = resph.size();
    int threshold = total_size * 0.0006;

    std::cout << "max diff: " << max_diff <<":  " << max_val << " != " << max_gold << std::endl;
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

