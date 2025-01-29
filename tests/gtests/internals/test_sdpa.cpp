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

#include "sdpa_internal.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <memory>
#include <random>

using mdt = memory::data_type;

enum class quantize_type {
    no_quantization,
    per_tensor,
    per_tensor1,
    per_tensor3,
    per_token,
    per_token_with_groups
};

enum class mask_type { no_mask, oneD, twoD, causal };

struct sdpa_dims_t {
    memory::dim mb;
    memory::dim head_num;
    memory::dim kv_group_size;
    memory::dim seq_len;
    memory::dim query_num;
    memory::dim head_size;

    int kgroup_size;
    int vgroup_size;

    memory::data_type dt;

    memory::data_type qdt;

    memory::data_type kdt;
    memory::data_type ksdt;
    memory::data_type kzpdt;

    memory::data_type vdt;
    memory::data_type vsdt;
    memory::data_type vzpdt;

    memory::data_type mskdt;

    quantize_type qtype;
    bool with_key_transposed;
    mask_type mask;
};

struct sdpa_tensors_t {
    memory m_query, m_key, m_scale, m_mask, m_value, m_output;
    memory m_query_test;
    memory m_key_quantized, m_value_quantized, m_output_quantized;
    memory m_key_t, m_value_t;
    memory m_key_t_quantized, m_value_t_quantized;

    memory m_reorder_scale_attr, m_key_scales, m_key_scales_t, m_key_zp,
            m_value_scales, m_value_zp;
    dnnl::primitive_attr sdpa_attr, sdpa_attr_quantized, sdpa_kq_attr_quantized,
            sdpa_vs_attr_quantized;

    int kq_mask, vs_mask;
    memory::dims kq_groups, vs_groups;
};

inline memory::dim product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (memory::dim)1, std::multiplies<memory::dim>());
}

std::ostream &operator<<(std::ostream &ss, const quantize_type &qt) {
    switch (qt) {
        case quantize_type::no_quantization: ss << "no_quantization"; break;
        case quantize_type::per_tensor: ss << "per_tensor"; break;
        case quantize_type::per_tensor1: ss << "per_tensor1"; break;
        case quantize_type::per_tensor3: ss << "per_tensor3"; break;
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
        case mdt::bf16: ss << "bf16"; break;
        case mdt::s8: ss << "s8"; break;
        case mdt::u8: ss << "u8"; break;
        case mdt::s4: ss << "s4"; break;
        case mdt::u4: ss << "u4"; break;
        default: ss << "na"; break;
    }
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const sdpa_dims_t &p) {
    ss << "mb_" << p.mb;
    if (p.with_key_transposed)
        ss << "_T";
    else
        ss << "_";
    ss << "K_" << p.seq_len;
    ss << "_head_num_" << p.head_num;
    ss << "_D_" << p.head_size;
    ss << "_Q_" << p.query_num;
    ss << "_Qdt_" << p.qdt;
    ss << "_Kdt_" << p.kdt;
    if (p.kdt != mdt::f16 && p.kdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        ss << "_Ksdt_" << p.ksdt;
        ss << "_Kzpdt_" << p.kzpdt;
    }
    ss << "_Vdt_" << p.vdt;
    if (p.vdt != mdt::f16 && p.kdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        ss << "_Vsdt_" << p.vsdt;
        ss << "_Vzpdt_" << p.vzpdt;
    }
    switch (p.mask) {
        case mask_type::no_mask: ss << "_no_mask"; break;
        case mask_type::oneD: ss << "_mask1D"; break;
        case mask_type::twoD: ss << "_mask2D"; break;
        case mask_type::causal: ss << "_maskcausal"; break;
    }
    if (!(p.kdt == mdt::f16 || p.vdt == mdt::f16)
            && !(p.kdt == mdt::bf16 || p.vdt == mdt::f16)) {
        ss << "_qtype_" << p.qtype;
    }
    return ss;
}

std::string print_to_string(const ::testing::TestParamInfo<sdpa_dims_t> &info) {
    std::stringstream ss;
    ss << info.param;
    return ss.str();
}

std::string print_table_header(const sdpa_dims_t &p) {
    std::stringstream ss;
    ss << "| mb |    K | #Head |   D |    q | Kdt | Vdt |  time |";
    return ss.str();
}

std::string print_row(const sdpa_dims_t &p) {
    std::stringstream ss;

    ss << "|" << p.mb;
    ss << "|" << p.seq_len;
    ss << "|" << p.head_num;
    ss << "|" << p.head_size;
    ss << "|" << p.query_num;
    ss << "|" << p.kdt;
    if (p.kdt != mdt::f16 && p.vdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.ksdt;
        ss << "/" << p.kzpdt;
    }
    ss << "|" << p.vdt;
    if (p.vdt != mdt::f16 && p.vdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        ss << "/" << p.vsdt;
        ss << "/" << p.vzpdt;
    }
    ss << "|";
    switch (p.mask) {
        case mask_type::no_mask: ss << "no"; break;
        case mask_type::oneD: ss << "1D"; break;
        case mask_type::twoD: ss << "2D"; break;
        case mask_type::causal: ss << "causal"; break;
    }
    ss << "|" << p.qtype;
    return ss.str();
}

using dnnl::algorithm;
using dnnl::matmul;
using dnnl::memory;
using dnnl::primitive_attr;
using dnnl::softmax_forward;

#define COMPLAIN_DNNL_ERROR_AND_EXIT(what, status) \
    do { \
        printf("[%s:%d] `%s` returns oneDNN error: %s.\n", __FILE__, __LINE__, \
                what, dnnl_status2str(status)); \
        printf("Example failed.\n"); \
        exit(1); \
    } while (0)

#define COMPLAIN_EXAMPLE_ERROR_AND_EXIT(complain_fmt, ...) \
    do { \
        printf("[%s:%d] Error in the example: " complain_fmt ".\n", __FILE__, \
                __LINE__, __VA_ARGS__); \
        printf("Example failed.\n"); \
        exit(2); \
    } while (0)

#undef CHECK
#define CHECK(f) \
    do { \
        dnnl_status_t s_ = f; \
        if (s_ != dnnl_success) COMPLAIN_DNNL_ERROR_AND_EXIT(#f, s_); \
    } while (0)

/// Read from handle, write to memory
/// This function is similar to the function found in write_to_dnnl_memory but this
/// function has been expanded to perform an inline conversion from the source data
/// type(handle) to the destination memory object. Currently, it only supports
/// bf16, f16, f32, s32, u8/s8, u4/s4 data types for the dnnl::memory object
/// and unsigned int and float for the handle.
///
/// The function first transfers the data to the handle's data type then
/// uses reorder to perform the conversion to the destination data type.
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

std::random_device &get_random_device() {
    static std::random_device rd;
    return rd;
}

std::mt19937 &get_generator() {
    static std::mt19937 generator(get_random_device()());
    return generator;
}

// this is changed from the fill_random() function in matmul_perf.cpp.
void fill_random(std::vector<float> &out, const memory::desc &desc) {
    static std::vector<float> random_data_f;
    constexpr memory::dim nrand = 1037;

    if (random_data_f.empty()) {
        std::uniform_real_distribution<float> dist_f(-3.0f, 4.0f);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f)
            d = dist_f(get_generator());
    }

    auto elems = product(desc.get_dims());
    for (memory::dim i = 0; i < elems; i += nrand) {
        size_t chunk = std::min(nrand, elems - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

template <typename T>
void fill_random_quantized(std::vector<T> &out, const memory::desc &desc,
        bool is_unsigned = false) {
    static std::vector<T> random_data_f;
    static std::vector<T> random_data_u;
    constexpr memory::dim nrand = 2049;

    if (random_data_f.empty() || random_data_u.empty()) {
        std::uniform_int_distribution<int> dist_f(-4, 4);
        std::uniform_int_distribution<unsigned> dist_u(0, 6);

        random_data_u.resize(nrand);
        for (auto &d : random_data_u) {
            d = dist_u(get_generator());
        }
        random_data_f.resize(nrand);
        for (auto &d : random_data_f) {
            d = dist_f(get_generator());
        }
    }

    auto elems = product(desc.get_dims());
    if (std::is_same<unsigned, T>::value || is_unsigned) {
        for (memory::dim i = 0; i < elems; i += nrand) {
            size_t chunk = std::min(nrand, elems - i);
            std::memcpy(&out[i], random_data_u.data(), chunk * sizeof(T));
        }
    } else {
        for (memory::dim i = 0; i < elems; i += nrand) {
            size_t chunk = std::min(nrand, elems - i);
            std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(T));
        }
    }
}

void fill_random_scales(std::vector<float> &out, const memory::desc &desc) {
    static std::vector<float> random_data_f;
    constexpr memory::dim nrand = 1037;

    if (random_data_f.empty()) {
        std::uniform_int_distribution<int> dist_f(-16, 16);

        random_data_f.resize(nrand);
        for (auto &d : random_data_f) {
            auto value = dist_f(get_generator()) * 0.125f;
            if (value == 0.f) value = dist_f(get_generator()) * 0.125f;
            d = value;
        }
    }

    auto elems = product(desc.get_dims());
    for (memory::dim i = 0; i < elems; i += nrand) {
        size_t chunk = std::min(nrand, elems - i);
        std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(float));
    }
}

// initialize the mask with first 3/4 elements with 0s and the last 1/4 elements
// with -inf.
void fill_mask(std::vector<float> &mask, const memory::desc &desc) {
    size_t seq_len = desc.get_dims()[3];
    size_t query_num = desc.get_dims()[2];
    size_t batches = desc.get_dims()[1] * desc.get_dims()[0];
    for (size_t b = 0; b < batches; b++) {
        for (size_t q = 0; q < query_num; q++) {
            for (size_t i = 0; i < seq_len; i++) {
                if (i <= q) {
                    mask[b * query_num * seq_len + q * seq_len + i] = 0;
                    // = (float)i + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + i]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)i + (float)q / 100.f);
                }
            }
        }
    }
}

void fill_causal_mask(std::vector<float> &mask, const memory::desc &desc) {
    size_t seq_len = desc.get_dims()[3];
    size_t query_num = desc.get_dims()[2];
    size_t batches = desc.get_dims()[1] * desc.get_dims()[0];
    for (size_t b = 0; b < batches; b++) {
        for (size_t q = 0; q < query_num; q++) {
            for (size_t i = 0; i < seq_len; i++) {
                if (i <= q) {
                    mask[b * query_num * seq_len + q * seq_len + i] = 0;
                    // = (float)i + (float)q / 100.f;
                } else {
                    mask[b * query_num * seq_len + q * seq_len + i]
                            = -1 * std::numeric_limits<float>::infinity();
                    //= -((float)i + (float)q / 100.f);
                }
            }
        }
    }
}

void print_mem(const dnnl::memory &mem, const std::string &name = "") {
    auto eng = mem.get_engine();
    dnnl::stream s(eng);
    s.wait();
    auto desc = mem.get_desc();
    auto dims = desc.get_dims();
    auto strides = desc.get_strides();
    printf("%sbegin\n", name.c_str());
    printf("dims   : %6ld %6ld %6ld %6ld\n", (long)dims[0], (long)dims[1],
            (long)dims[2], (long)dims[3]);
    printf("strides: %6ld %6ld %6ld %6ld\n", (long)strides[0], (long)strides[1],
            (long)strides[2], (long)strides[3]);
    if (mem.get_desc().get_data_type() == dnnl_bf16) { printf("bf16\n"); }
    void *mapped_ptr_ = (void *)mem.map_data();
    printf("        i:    ");
    for (int i = 0; i < dims[3]; i++) {
        switch ((int)desc.get_data_type()) {
            case dnnl_u4:
            case dnnl_s4: printf("%4d", i); break;
            case dnnl_u8:
            case dnnl_s8: printf("%4d", i); break;
            case dnnl_f32:
            case dnnl_bf16:
            case dnnl_f16: printf("%9d", i); break;
        }
    }
    printf("\n");

    switch ((int)desc.get_data_type()) {
        case dnnl_u4:
        case dnnl_s4: {
            char *mapped_ptr = (char *)mapped_ptr_;
            for_(int l = 0; l < dims[0]; l++)
            for_(int k = 0; k < dims[1]; k++)
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
        } break;

        case dnnl_u8:
        case dnnl_s8: {
            char *mapped_ptr = (char *)mapped_ptr_;
            for_(int l = 0; l < dims[0]; l++)
            for_(int k = 0; k < dims[1]; k++)
            for (int j = 0; j < dims[2]; j++) {
                printf("(%2d, %2d, %3d): ", l, k, j);
                for (int i = 0; i < dims[3]; i++) {
                    printf("%4d",
                            (mapped_ptr[l * strides[0] + k * strides[1]
                                    + j * strides[2] + i * strides[3]]));
                }
                printf("\n");
            }
        } break;
        case dnnl_bf16: {
            using dnnl::impl::bfloat16_t;
            bfloat16_t *mapped_ptr = (bfloat16_t *)mapped_ptr_;

            for_(int l = 0; l < dims[0]; l++)
            for_(int k = 0; k < dims[1]; k++)
            for (int j = 0; j < dims[2]; j++) {
                printf("(%2d, %2d, %3d):", l, k, j);
                for (int i = 0; i < dims[3]; i++) {
                    printf("%+9.3f",
                            (float)(mapped_ptr[l * strides[0] + k * strides[1]
                                    + j * strides[2] + i * strides[3]]));
                }
                printf("\n");
            }
        } break;
        case dnnl_f16: {
            using dnnl::impl::float16_t;
            float16_t *mapped_ptr = (float16_t *)mapped_ptr_;

            for_(int l = 0; l < dims[0]; l++)
            for_(int k = 0; k < dims[1]; k++)
            for (int j = 0; j < dims[2]; j++) {
                printf("(%2d, %2d, %3d):", l, k, j);
                for (int i = 0; i < dims[3]; i++) {
                    printf("%+9.3f",
                            (mapped_ptr[l * strides[0] + k * strides[1]
                                    + j * strides[2] + i * strides[3]]
                                            .f()));
                }
                printf("\n");
            }
        } break;
        case dnnl_f32: {
            using dnnl::impl::float16_t;
            float *mapped_ptr = (float *)mapped_ptr_;

            for_(int l = 0; l < dims[0]; l++)
            for_(int k = 0; k < dims[1]; k++)
            for (int j = 0; j < dims[2]; j++) {
                printf("(%2d, %2d, %3d):", l, k, j);
                for (int i = 0; i < dims[3]; i++) {
                    printf("%+9.3f",
                            (mapped_ptr[l * strides[0] + k * strides[1]
                                    + j * strides[2] + i * strides[3]]));
                }
                printf("\n");
            }
        } break;
        default: throw std::runtime_error("Not supported");
    }
    mem.unmap_data(mapped_ptr_);
    printf("%send\n", name.c_str());
}

template <typename T>
std::vector<float> dequantize(const std::vector<float> &input,
        memory::desc &desc, memory::desc &scale_md,
        const std::vector<T> &zero_points, const std::vector<float> &scales,
        int group_size, quantize_type qtype, memory::dims groups,
        int token_dim = -1) {
    std::vector<float> out(input.size());
    if (qtype == quantize_type::per_tensor
            || qtype == quantize_type::per_tensor1
            || qtype == quantize_type::per_tensor3) {
        for (size_t i = 0; i < input.size(); i++)
            out[i] = (input[i] - zero_points[i / group_size])
                    * scales[i / group_size];
    } else {
        auto dims = desc.get_dims();
        auto strides = desc.get_strides();

        auto scales_dim = scale_md.get_dims();
        auto zp_dim = scale_md.get_dims();
        auto scales_strides = scale_md.get_strides();
        auto zp_strides = scale_md.get_strides();

        if (qtype == quantize_type::no_quantization) { groups = {1, 1}; }
        if (qtype == quantize_type::per_token) {
            if (token_dim == 0) {
                groups = {dims[2], 1};
            } else if (token_dim == 1) {
                groups = {1, dims[3]};
            }
        }

        int sg = 0;
        int zg = 0;
        int scale_offset = 0;
        int zp_offset = 0;

        for_(int l = 0; l < dims[0]; l++)
        for_(int k = 0; k < dims[1]; k++)
        for_(int j = 0; j < dims[2]; j++)
        for (int i = 0; i < dims[3]; i++) {
            if (groups[0] > 1) {
                sg = groups[0];
                scale_offset = l * scales_strides[0] + k * scales_strides[1]
                        + j / sg * scales_strides[2] + i * scales_strides[3];
            } else if (groups[1] > 1) {
                sg = groups[1];
                scale_offset = l * scales_strides[0] + k * scales_strides[1]
                        + j * scales_strides[2] + i / sg * scales_strides[3];
            }
            if (groups[1] > 1) {
                zg = groups[1];
                zp_offset = l * zp_strides[0] + k * zp_strides[1]
                        + j * zp_strides[2] + i / zg * zp_strides[3];
            } else if (groups[0] > 1) {
                zg = groups[0];
                zp_offset = l * zp_strides[0] + k * zp_strides[1]
                        + j / zg * zp_strides[2] + i;
            }
            int offset = l * strides[0] + k * strides[1] + j * strides[2]
                    + i * strides[3];

            out[offset] = (input[offset] - zero_points[zp_offset])
                    * scales[scale_offset];
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

void transpose(const dnnl::engine &eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    void *ptr2 = out.map_data();
    void *ptr1 = in.map_data();

    std::memcpy(ptr2, ptr1, in.get_desc().get_size());
    in.unmap_data(ptr1);
    out.unmap_data(ptr2);
}

void transpose_strides(const dnnl::engine &eng, memory &out, memory &in) {
    dnnl::stream s(eng);

    if (out.get_desc().get_data_type() == mdt::u4
            || out.get_desc().get_data_type() == mdt::s4) {
        auto desc = in.get_desc();
        auto dims = desc.get_dims();
        auto strides = desc.get_strides();
        auto strides_t = out.get_desc().get_strides();

        char *mapped_ptr = (char *)in.map_data();
        char *mapped_ptr_t = (char *)out.map_data();
        for_(int l = 0; l < dims[0]; l++)
        for_(int k = 0; k < dims[1]; k++)
        for_(int j = 0; j < dims[2]; j++)
        for (int i = 0; i < dims[3]; i++) {
            int is_odd = i % 2;
            int is_odd_t = j % 2;

            auto offset = l * strides[0] + k * strides[1] + j * strides[2]
                    + i * strides[3];
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
        in.unmap_data(mapped_ptr);
        out.unmap_data(mapped_ptr_t);
    } else {
        dnnl::reorder(in, out).execute(s, in, out);
    }
}

memory::dims double_mb(const memory::dims &dims) {
    memory::dims ret = dims;
    ret[0] *= 2;
    return ret;
}

memory double_and_resize(const memory::desc &desc, dnnl::engine &eng) {
    dnnl::stream s(eng);
    memory::dims dims2 = double_mb(desc.get_dims());
    auto desc2 = memory::desc(dims2, desc.get_data_type(), desc.get_strides());

    dnnl_memory_t mem2;
    CHECK(dnnl_memory_create(
            &mem2, desc2.get(), eng.get(), DNNL_MEMORY_ALLOCATE));

    void *mapped_ptr = nullptr;
    CHECK(dnnl_memory_map_data(mem2, &mapped_ptr));

    for (size_t i = 0; i < desc2.get_size(); i++) {
        ((uint8_t *)mapped_ptr)[i] = 0xFF;
    }
    CHECK(dnnl_memory_unmap_data(mem2, mapped_ptr));

    void *handle;
    CHECK(dnnl_memory_get_data_handle(mem2, &handle));
    return memory(desc, eng, handle);
}

sdpa_tensors_t get_descriptors(dnnl::engine &eng, const sdpa_dims_t &p) {
    sdpa_tensors_t out;

    // Prepare input and output shapes to construct the sdpa graph.
    const memory::dims q_sz = {p.mb, p.head_num, p.query_num, p.head_size};
    const memory::dims k_sz = {p.mb, p.head_num, p.head_size, p.seq_len};
    const memory::dims k_stride
            = {p.mb, p.head_num, p.head_size, p.seq_len * 2};
    const memory::dims k_t_stride
            = {p.mb, p.head_num, p.seq_len * 2, p.head_size};
    const memory::dims v_sz = {p.mb, p.head_num, p.seq_len, p.head_size};
    const memory::dims scale_sz = {1, 1, 1, 1};
    const memory::dims key_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        k_sz[0], k_sz[1], k_sz[2] / p.kgroup_size, k_sz[3]};
            case quantize_type::per_token:
                return memory::dims {k_sz[0], k_sz[1], 1, k_sz[3]};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {k_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {k_sz[0], k_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();
    const memory::dims val_scales_sz = [&] {
        switch (p.qtype) {
            case quantize_type::no_quantization:
                return memory::dims {1, 1, 1, 1};
            case quantize_type::per_token_with_groups:
                return memory::dims {
                        v_sz[0], v_sz[1], v_sz[2], v_sz[3] / p.vgroup_size};
            case quantize_type::per_token:
                return memory::dims {v_sz[0], v_sz[1], v_sz[2], 1};
            case quantize_type::per_tensor: return memory::dims {1, 1, 1, 1};
            case quantize_type::per_tensor1:
                return memory::dims {v_sz[0], 1, 1, 1};
            case quantize_type::per_tensor3:
                return memory::dims {v_sz[0], v_sz[1], 1, 1};
        }
        throw std::runtime_error("Quantization type not supported\n");
    }();

    memory::dims mask_sz;
    switch (p.mask) {
        case mask_type::no_mask: mask_sz = {};
        case mask_type::oneD: mask_sz = {1, 1, 1, p.seq_len}; break;
        case mask_type::causal:
        case mask_type::twoD: mask_sz = {1, 1, p.query_num, p.seq_len}; break;
    }

    auto ksdt = p.ksdt == mdt::undef ? p.kdt : p.ksdt;
    auto kzpdt = p.kzpdt == mdt::undef ? mdt::s8 : p.kzpdt;
    auto vsdt = p.vsdt == mdt::undef ? p.vdt : p.vsdt;
    auto vzpdt = p.vzpdt == mdt::undef ? mdt::s8 : p.vzpdt;

    memory::format_tag abcd = memory::format_tag::abcd;
    memory::format_tag abdc = memory::format_tag::abdc;
    // score = query x key.T
    // scaled_score = score / scale
    // masked_score = scaled_score + mask
    // All combined in a single matmul primitive.
    // clang-format off
    auto query_md            = memory::desc(q_sz,          p.qdt,   abcd);
    auto key_t_md            = memory::desc(k_sz,          p.dt,    abdc);
    auto key_md              = memory::desc(k_sz,          p.dt,    abcd);
    auto value_t_md          = memory::desc(v_sz,          p.dt,    abdc);
    auto value_md            = memory::desc(v_sz,          p.dt,    abcd);
    auto scale_md            = memory::desc(scale_sz,      p.qdt,    abcd);

    auto query_test_md       = memory::desc(q_sz,          p.qdt,   abcd);

    auto key_quantized_md    = memory::desc(k_sz,          p.kdt,   abcd);
    auto key_t_quantized_md  = memory::desc(k_sz,          p.kdt,   abdc);
    auto key_scales_md       = memory::desc(key_scales_sz, ksdt,    abcd);
    auto key_scales_t_md     = memory::desc(key_scales_sz, ksdt,    abdc);
    auto key_zp_md           = memory::desc(key_scales_sz, kzpdt,   abcd);

    auto val_quantized_md    = memory::desc(v_sz,          p.vdt,   abcd);
    auto val_t_quantized_md  = memory::desc(v_sz,          p.vdt,   abdc);
    auto val_scales_md       = memory::desc(val_scales_sz, vsdt,    abcd);
    auto val_zp_md           = memory::desc(val_scales_sz, vzpdt,   abcd);


    auto mask_md             = memory::desc(mask_sz,       p.mskdt, abcd);
    auto output_md           = memory::desc(q_sz,          p.qdt,   abcd);
    auto output_quantized_md = memory::desc(q_sz,          p.qdt,   abcd);
    // clang-format on

    // Create memory objects
    out.m_query = double_and_resize(query_md, eng);
    out.m_query_test = double_and_resize(query_test_md, eng);
    out.m_key = double_and_resize(key_md, eng);
    out.m_key_t = double_and_resize(key_t_md, eng);
    out.m_scale = double_and_resize(scale_md, eng);
    out.m_key_quantized = double_and_resize(key_quantized_md, eng);
    out.m_key_t_quantized = double_and_resize(key_t_quantized_md, eng);
    out.m_key_scales = double_and_resize(key_scales_md, eng);
    out.m_key_scales_t = double_and_resize(key_scales_t_md, eng);
    out.m_key_zp = double_and_resize(key_zp_md, eng);
    out.m_value_quantized = double_and_resize(val_quantized_md, eng);
    out.m_value_t_quantized = double_and_resize(val_t_quantized_md, eng);
    out.m_value_scales = double_and_resize(val_scales_md, eng);
    out.m_value_zp = double_and_resize(val_zp_md, eng);
    out.m_mask = double_and_resize(mask_md, eng);
    out.m_value = double_and_resize(value_md, eng);
    out.m_value_t = double_and_resize(value_t_md, eng);
    out.m_output = double_and_resize(output_md, eng);
    out.m_output_quantized = double_and_resize(output_quantized_md, eng);

    // Allocate user data.
    std::vector<float> query_data(product(q_sz), 0.f);
    std::vector<float> scale_data(product(scale_sz), std::sqrt(p.head_size));
    std::vector<float> key_quantized_data(product(k_sz), 0);
    std::vector<float> val_quantized_data(product(v_sz), 0);
    std::vector<float> key_scale_data(product(key_scales_sz), 1.f);
    std::vector<float> val_scale_data(product(val_scales_sz), 1.f);

    std::vector<int> key_zp_data_signed(product(key_scales_sz), INT_MAX);
    std::vector<int> val_zp_data_signed(product(val_scales_sz), INT_MAX);

    std::vector<unsigned> key_zp_data_unsigned(product(key_scales_sz), INT_MAX);
    std::vector<unsigned> val_zp_data_unsigned(product(val_scales_sz), INT_MAX);

    std::vector<float> mask_data(product(mask_sz), NAN);
    std::vector<float> output_data(product(q_sz), NAN);

    out.sdpa_attr.set_scratchpad_mode(dnnl::scratchpad_mode::library);
    out.sdpa_attr_quantized.set_scratchpad_mode(dnnl::scratchpad_mode::library);

    out.kq_mask = 0;
    out.vs_mask = 0;
    out.kq_groups = {};
    out.vs_groups = {};
    switch (p.qtype) {
        case quantize_type::per_token_with_groups:
            out.kq_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0;
            out.kq_groups = {p.kgroup_size, 1};
            out.vs_groups = {1, p.vgroup_size};
            break;
        case quantize_type::per_token:
            out.kq_mask = 1 << 3 | 1 << 1 | 1 << 0;
            out.vs_mask = 1 << 0 | 1 << 1 | 1 << 2;
            break;
        case quantize_type::per_tensor3:
            out.kq_mask = 3;
            out.vs_mask = 3;
            break;
        case quantize_type::per_tensor1:
            out.kq_mask = 1;
            out.vs_mask = 1;
            break;
        case quantize_type::per_tensor:
            out.kq_mask = 0;
            out.vs_mask = 0;
            break;
        case quantize_type::no_quantization: break;
    }

    if (p.qtype != quantize_type::no_quantization) {
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.ksdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.ksdt);
        }

        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vsdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_scales(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.vsdt);
        }

        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef) {
            out.sdpa_kq_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.kq_mask, out.kq_groups, p.kzpdt);
        }

        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef) {
            out.sdpa_vs_attr_quantized.set_zero_points(
                    DNNL_ARG_WEIGHTS, out.vs_mask, out.vs_groups, p.vzpdt);
        }
    }

    fill_random(query_data, query_md);
    fill_random_quantized(key_quantized_data, key_quantized_md,
            (p.kdt == mdt::u4 || p.kdt == mdt::u8));
    fill_random_quantized(val_quantized_data, val_quantized_md,
            (p.vdt == mdt::u4 || p.vdt == mdt::u8));
    if (p.qtype != quantize_type::no_quantization) {
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.ksdt != mdt::undef)
            fill_random_scales(key_scale_data, key_scales_md);
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vsdt != mdt::undef)
            fill_random_scales(val_scale_data, val_scales_md);
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef)
            fill_random_quantized(key_zp_data_signed, key_zp_md);
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef)
            fill_random_quantized(val_zp_data_signed, val_zp_md);
        if (p.kdt != mdt::f16 && p.kdt != mdt::bf16 && p.kzpdt != mdt::undef)
            fill_random_quantized(key_zp_data_unsigned, key_zp_md);
        if (p.vdt != mdt::f16 && p.vdt != mdt::bf16 && p.vzpdt != mdt::undef)
            fill_random_quantized(val_zp_data_unsigned, val_zp_md);
    }

    if (p.mask == mask_type::causal) {
        fill_causal_mask(mask_data, mask_md);
    } else {
        fill_mask(mask_data, mask_md);
    }

/// This section allows setting the values of the tensors using environment variables.
/// Syntax:
///    <Tensor Name>[<S for scales, Z for zero points>]<R for row C for column>
///
/// KR=3 KC=1 Set the value in the  Key tensor at (3, 1) to 1 and all other values should be zero
/// VSR=1 VSC=2  Set the scale for the Value tensor at (1, 2) to 1 and all other values to zero
#if 0
    auto &Q = query_data;
    auto &K = key_quantized_data;
    auto &V = val_quantized_data;
    auto &Ks = key_scale_data;
    auto &Vs = val_scale_data;
    auto &Kz = key_zp_data_signed;
    auto &Vz = val_zp_data_signed;
    auto d = p.head_size;
    auto k = p.seq_len;
    auto q = p.query_num;

    int kr = -1, kc = -1, qr = -1, qc = -1, vr = -1, vc = -1, mr = -1, mc = -1,
        xb = 0;
    int ksr = -1, ksc = -1, kzr = -1, kzc = -1, vsr = -1, vscales = -1,
        vzr = -1, vzc = -1;
    if (getenv("KR")) kr = atoi(getenv("KR"));
    if (getenv("KC")) kc = atoi(getenv("KC"));
    if (getenv("KSR")) ksr = atoi(getenv("KSR"));
    if (getenv("KSC")) ksc = atoi(getenv("KSC"));
    if (getenv("KZR")) kzr = atoi(getenv("KZR"));
    if (getenv("KZC")) kzc = atoi(getenv("KZC"));
    if (getenv("QR")) qr = atoi(getenv("QR"));
    if (getenv("QC")) qc = atoi(getenv("QC"));
    if (getenv("VR")) vr = atoi(getenv("VR"));
    if (getenv("VC")) vc = atoi(getenv("VC"));
    if (getenv("VSR")) vsr = atoi(getenv("VSR"));
    if (getenv("VScaleC")) vscales = atoi(getenv("VScaleC"));
    if (getenv("VZR")) vzr = atoi(getenv("VZR"));
    if (getenv("VZC")) vzc = atoi(getenv("VZC"));
    if (getenv("XB")) xb = atoi(getenv("XB"));

    if (getenv("MR")) mr = atoi(getenv("MR"));
    if (getenv("MC")) mc = atoi(getenv("MC"));

    if (mr >= 0 || mc >= 0) {
        mr = std::max(mr, 0);
        mc = std::max(mc, 0);
        for (auto &m : mask_data)
            m = 0;
        mask_data[mr * p.seq_len + mc] = -999;
    }
    if (kr >= 0 || kc >= 0) {
        kr = std::max(kr, 0);
        kc = std::max(kc, 0);
        if (getenv("KX")) {
            for (int kr_ = 0; kr_ < d; kr_++)
                for (int kc_ = 0; kc_ < k; kc_++)
                    if (kr_ >= kr || kc_ >= kc) K[kr_ * k + kc_] = 0;
        } else {
            for (auto &k : K)
                k = 0;
            K[xb * d * k + kr * k + kc] = 1;
        }
    }
    if (ksr >= 0 || ksc >= 0) {
        ksr = std::max(ksr, 0);
        ksc = std::max(ksc, 0);
        for (auto &ks : Ks)
            ks = 0;
        Ks[(xb * d / p.kgroup_size * k + ksr * k) + ksc] = 1;
    }
    if (kzr >= 0 || kzc >= 0) {
        kzr = std::max(kzr, 0);
        kzc = std::max(kzc, 0);
        for (auto &kz : Kz)
            kz = 0;
        Kz[(xb * d * k + kzr * d) / p.kgroup_size + kzc] = 2;
    }
    if (qr >= 0 || qc >= 0) {
        qr = std::max(qr, 0);
        qc = std::max(qc, 0);
        if (getenv("QX")) {
            for (int qr_ = 0; qr_ < d; qr_++)
                for (int qc_ = 0; qc_ < q; qc_++)
                    if (qr_ >= qr || qc_ >= qc) Q[qr_ * d + qc_] = 0;
        } else {
            for (auto &q : Q)
                q = 0;
            Q[xb * d * q + qr * d + qc] = 1;
        }
    }
    if (vr >= 0 || vc >= 0) {
        vr = std::max(vr, 0);
        vc = std::max(vc, 0);
        if (getenv("VX")) {
            for (int vr_ = 0; vr_ < k; vr_++)
                for (int vc_ = 0; vc_ < d; vc_++)
                    if (vr_ >= vr || vc_ >= vc) V[vr_ * d + vc_] = 0;
        } else {
            for (auto &v : V)
                v = 0;
            V[xb * d * k + vr * d + vc] = 1;
        }
    }
    if (vsr >= 0 || vscales >= 0) {
        vsr = std::max(vsr, 0);
        vscales = std::max(vscales, 0);
        for (auto &vs : Vs)
            vs = 0;
        Vs[(xb * d * k + vscales * d) / p.vgroup_size + vsr] = 1;
    }
    if (vzr >= 0 || vzc >= 0) {
        vzr = std::max(vzr, 0);
        vzc = std::max(vzc, 0);
        for (auto &vz : Vz)
            vz = 0;
        Vz[(xb * d * k + vzc * d) / p.vgroup_size + vzr] = 1;
    }
#endif

    int group_size = p.kgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = k_sz[0] * k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = k_sz[1] * k_sz[2] * k_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = k_sz[2] * k_sz[3];
    }

    std::vector<float> key_data;
    if (p.kzpdt == mdt::s4 || p.kzpdt == mdt::s8) {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_signed, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    } else {
        key_data = dequantize(key_quantized_data, key_md, key_scales_md,
                key_zp_data_unsigned, key_scale_data, group_size, p.qtype,
                out.kq_groups, 0);
    }
    group_size = p.vgroup_size;
    if (p.qtype == quantize_type::per_tensor) {
        group_size = v_sz[0] * v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor1) {
        group_size = v_sz[1] * v_sz[2] * v_sz[3];
    } else if (p.qtype == quantize_type::per_tensor3) {
        group_size = v_sz[2] * v_sz[3];
    }
    std::vector<float> value_data;
    if (p.vzpdt == mdt::s4 || p.vzpdt == mdt::s8) {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_signed, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    } else {
        value_data = dequantize(val_quantized_data, value_md, val_scales_md,
                val_zp_data_unsigned, val_scale_data, group_size, p.qtype,
                out.vs_groups, 1);
    }

    write_to_dnnl_memory(mask_data.data(), out.m_mask);
    write_to_dnnl_memory(scale_data.data(), out.m_scale);

    // Write data to tensor object's handle.
    write_to_dnnl_memory(key_data.data(), out.m_key);
    write_to_dnnl_memory(value_data.data(), out.m_value);
    write_to_dnnl_memory(query_data.data(), out.m_query);
    write_to_dnnl_memory(query_data.data(), out.m_query_test);

    write_to_dnnl_memory(key_quantized_data.data(), out.m_key_quantized);

    write_to_dnnl_memory(val_quantized_data.data(), out.m_value_quantized);
    if (p.kzpdt == mdt::s4 || p.kzpdt == mdt::s8) {
        write_to_dnnl_memory(key_zp_data_signed.data(), out.m_key_zp);
    } else {
        write_to_dnnl_memory(key_zp_data_unsigned.data(), out.m_key_zp);
    }
    if (p.vzpdt == mdt::s4 || p.vzpdt == mdt::s8) {
        write_to_dnnl_memory(val_zp_data_signed.data(), out.m_value_zp);
    } else {
        write_to_dnnl_memory(val_zp_data_unsigned.data(), out.m_value_zp);
    }
    write_to_dnnl_memory(key_scale_data.data(), out.m_key_scales);
    write_to_dnnl_memory(val_scale_data.data(), out.m_value_scales);
    write_to_dnnl_memory(output_data.data(), out.m_output);
    write_to_dnnl_memory(output_data.data(), out.m_output_quantized);

    transpose_strides(eng, out.m_key_scales_t, out.m_key_scales);
    transpose_strides(eng, out.m_key_t, out.m_key);
    transpose_strides(eng, out.m_key_t_quantized, out.m_key_quantized);
    transpose_strides(eng, out.m_value_t, out.m_value);
    transpose_strides(eng, out.m_value_t_quantized, out.m_value_quantized);

    return out;
}
class sdpa_test_t : public ::testing::TestWithParam<sdpa_dims_t> {
public:
    void SetUp() override {
        p = GetParam();
        SKIP_IF_CUDA(true, "SDPA primitive tests do not support CUDA");
        SKIP_IF_HIP(true, "SDPA primitive tests do not support HIP");
#ifdef DNNL_TEST_WITH_ENGINE_PARAM
        SKIP_IF(get_test_engine_kind() != dnnl::engine::kind::gpu,
                "This test requires GPU engine");
        eng = get_test_engine();
#else
        SKIP_IF(engine::get_count(engine::kind::gpu) == 0,
                "SDPA tests require gpus.");
        eng = dnnl::engine(engine::kind::gpu, 0);
#endif
        strm = dnnl::stream(eng);
        t = get_descriptors(eng, p);
    }

protected:
    sdpa_dims_t p;
    dnnl::engine eng;
    dnnl::stream strm;
    sdpa_tensors_t t;
};

bool with_key_transposed = true;
bool no_key_transposed = false;

// clang-format off
INSTANTIATE_TEST_SUITE_P(Graph,
    sdpa_test_t,
                               //  mb,  hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,       qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,  vzpdt,   mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,       2,        32,     32,      32,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,     33,       1,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    128,      64,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    129,       1,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    259,       1,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    384,     384,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    384,     384,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_tensor,             no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    384,     384,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_tensor1,            no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    384,     384,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_tensor3,            no_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,       2,        32,    384,     384,       96,   96,      96, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8, mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token,              no_key_transposed, mask_type::causal }
    ), &print_to_string);


//llama-2-7b-chat shape: Q [1x32xSEQ_LENx128] KV [1x32xSEQ_LENx128]
//llama-3-8b shape: Q [1x32xSEQ_LENx128] KV [1x8xSEQ_LENx128]
//minicpm-1b-sft shape:  Q [1x24xSEQ_LENx64]  KV [1x8xSEQ_LENx64]
//qwen2-7b shape: Q [1x28xSEQ_LENx128] KV [1x4xSEQ_LENx128]
//phi3-mini-4k-instruct shape: Q [1x32xSEQ_LENx96] KV [1x32xSEQ_LENx96]


INSTANTIATE_TEST_SUITE_P(llama_2_7b_chat,
    sdpa_test_t,
                               // mb,  hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,       qdt,       kdt,        ksdt,      kzpdt,        vdt,       vsdt,      vzpdt,     mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,      32,        32,    384,     384,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    385,       1,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    512,     512,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    513,       1,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   1024,    1024,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   1025,       1,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   2048,    2048,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   2049,       1,     128,   128,     128, mdt::f16,  mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,    mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed,  mask_type::causal }
    ), &print_to_string);



INSTANTIATE_TEST_SUITE_P(llama_3_8b,
    sdpa_test_t,
                               // mb, hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,      qdt,       kdt,        ksdt,      kzpdt,       vdt,       vsdt,      vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,  mdt::f16,    mdt::f16,    mdt::s8,  mdt::f16,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::no_quantization,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,     32,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16,    mdt::s8,   mdt::s8, mdt::undef, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,  mdt::undef, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },

                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16, mdt::undef, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },


                    sdpa_dims_t{   1,     2,          4,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          4,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8, mdt::undef,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },


                    sdpa_dims_t{   1,     2,          8,    384,     384,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,    385,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,    512,     512,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,    513,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,   1024,    1024,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,   1025,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,   2048,    2048,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,     2,          8,   2049,       1,     128,   128,     128, mdt::f16, mdt::f16,   mdt::s8,    mdt::f16, mdt::undef,   mdt::s8,   mdt::f16,    mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal }

    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(minicpm_1b_sft,
    sdpa_test_t,
                               // mb,  hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,       qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,      24,         8,    384,     384,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,    385,       1,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,    512,     512,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,    513,       1,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,   1024,    1024,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,   1025,       1,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,   2048,    2048,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      24,         8,   2049,       1,      64,    64,      64, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal }
    ), &print_to_string);


INSTANTIATE_TEST_SUITE_P(qwen2_7b,
    sdpa_test_t,
                               // mb,  hd_num, kv_grp_sz,seq_len,  qry_num, hd_size, kg_sz, vgrp_sz,       dt,        qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,  vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,      28,         4,    384,      384,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,    385,        1,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,    512,      512,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,    513,        1,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,   1024,     1024,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,   1025,        1,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,   2048,     2048,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      28,         4,   2049,        1,     128,   128,     128, mdt::f16,  mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal }
    ), &print_to_string);



INSTANTIATE_TEST_SUITE_P(phi3_mini_4k_instruct,
    sdpa_test_t,
                               // mb,  hd_num, kv_grp_sz,seq_len, qry_num, hd_size, kg_sz, vgrp_sz,       dt,        qdt,     kdt,      ksdt,   kzpdt,      vdt,     vsdt,   vzpdt,    mskdt, qtype
    testing::Values(
                    sdpa_dims_t{   1,      32,        32,    384,     384,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    385,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    512,     512,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,    513,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   1024,    1024,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   1025,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   2048,    2048,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal },
                    sdpa_dims_t{   1,      32,        32,   2049,       1,     96,     96,      96, mdt::f16,   mdt::f16, mdt::s8,  mdt::f16, mdt::s8,  mdt::s8, mdt::f16, mdt::s8, mdt::f16, quantize_type::per_token_with_groups,  with_key_transposed, mask_type::causal }
    ), &print_to_string);

// clang-format on

memory as(dnnl::stream &strm, memory &mem, memory::data_type dt) {
    const memory::dims sz = mem.get_desc().get_dims();

    auto md = memory::desc(sz, dt, mem.get_desc().get_strides());
    auto out = memory(md, mem.get_engine());
    dnnl::reorder(mem, out).execute(strm, mem, out);
    strm.wait();
    return out;
}

std::pair<dnnl::reorder, memory> dequantize_prim(const engine &eng, mdt dt,
        const memory::desc &desc, int mask, const memory::dims &groups, mdt sdt,
        mdt zpdt, dnnl::memory::format_tag tag = memory::format_tag::abcd) {
    auto dequantized_md = memory::desc(desc.get_dims(), dt, tag);
    primitive_attr dequantized_attr;

    if (sdt != mdt::undef) {
        dequantized_attr.set_scales(DNNL_ARG_FROM, mask, groups, sdt);
    }
    if (zpdt != mdt::undef) {
        dequantized_attr.set_zero_points(DNNL_ARG_SRC, mask, groups, zpdt);
    }

    auto dequantize_pd = dnnl::reorder::primitive_desc(
            eng, desc, eng, dequantized_md, dequantized_attr, false);

    memory dequantized_mem
            = memory({desc.get_dims(), dt, memory::format_tag::abcd}, eng);
    return std::make_pair(dnnl::reorder(dequantize_pd), dequantized_mem);
}

void prim_sdpa_quant(const sdpa_dims_t &p, const sdpa_tensors_t &t,
        dnnl::engine &eng, dnnl::stream &strm, dnnl::memory &query,
        dnnl::memory &key, dnnl::memory &key_scales, dnnl::memory &key_zp,
        dnnl::memory::data_type scale_dt, dnnl::memory &scale,
        dnnl::memory &mask, dnnl::memory &value, dnnl::memory &value_scales,
        dnnl::memory &value_zp, dnnl::memory &output, bool invert_scale) {

    const memory::dims score_sz = {p.mb, p.head_num, p.query_num, p.seq_len};
    auto score_md = memory::desc(score_sz, mdt::f32, memory::format_tag::abcd);

    using namespace dnnl;
    primitive_attr bmm1_attr;
    bmm1_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    post_ops bmm1_po;
    auto scale_f32 = as(strm, scale, mdt::f32);
    auto mask_f32 = as(strm, mask, mdt::f32);
    if (scale_dt != mdt::undef) {
        if (invert_scale)
            bmm1_po.append_binary(algorithm::binary_div, scale_f32.get_desc());
        else
            bmm1_po.append_binary(algorithm::binary_mul, scale_f32.get_desc());
    }
    if (p.mask != mask_type::no_mask) {
        bmm1_po.append_binary(algorithm::binary_add, mask_f32.get_desc());
    }

    bmm1_attr.set_post_ops(bmm1_po);

    memory key_dequantized;
    if ((key.get_desc().get_data_type() != mdt::f16
                && key.get_desc().get_data_type() != mdt::bf16)
            && p.qtype != quantize_type::no_quantization) {

        dnnl::reorder key_dequantize_prim;
        std::tie(key_dequantize_prim, key_dequantized)
                = dequantize_prim(eng, mdt::f16, key.get_desc(), t.kq_mask,
                        t.kq_groups, p.ksdt, p.kzpdt);

        key_dequantize_prim.execute(strm,
                {{DNNL_ARG_FROM, key}, {DNNL_ARG_TO, key_dequantized},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM, key_scales},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM, key_zp}});

        strm.wait();
    } else {
        key_dequantized = key;
        strm.wait();
    }

    memory value_dequantized;
    if (value.get_desc().get_data_type() != mdt::f16
            && value.get_desc().get_data_type() != mdt::bf16
            && p.qtype != quantize_type::no_quantization) {
        dnnl::reorder value_dequantize_prim;
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f32, value.get_desc(), t.vs_mask,
                        t.vs_groups, p.vsdt, p.vzpdt);

        value_dequantize_prim.execute(strm,
                {{DNNL_ARG_FROM, value}, {DNNL_ARG_TO, value_dequantized},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM, value_scales},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM, value_zp}});
        strm.wait();
    } else {
        value_dequantized = as(strm, value, mdt::f32);
        strm.wait();
    }

    auto score = memory(score_md, eng);
    auto score2 = memory(score_md, eng);
    auto bmm1_pd = matmul::primitive_desc(eng, query.get_desc(),
            key_dequantized.get_desc(), score.get_desc(), bmm1_attr);
    auto bmm1_prim = matmul(bmm1_pd);

    primitive_attr softmax_attr;
    softmax_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto softmax_pd = softmax_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::softmax_accurate,
            score.get_desc(), score.get_desc(), 3, softmax_attr);
    auto softmax_prim = softmax_forward(softmax_pd);

    // attention_output = attention_probs x value
    primitive_attr bmm2_attr;

    bmm2_attr.set_scratchpad_mode(scratchpad_mode::user);
    auto bmm2_pd = matmul::primitive_desc(eng, score.get_desc(),
            value_dequantized.get_desc(), output.get_desc(), bmm2_attr);
    auto bmm2_prim = matmul(bmm2_pd);

    size_t max_scratchpad_size = 0;
    auto bmm1_scratchpad = bmm1_pd.scratchpad_desc().get_size();
    auto softmax_scratchpad = softmax_pd.scratchpad_desc().get_size();
    auto bmm2_scratchpad = bmm2_pd.scratchpad_desc().get_size();
    for (auto &sz : {bmm1_scratchpad, softmax_scratchpad, bmm2_scratchpad}) {
        if (max_scratchpad_size < sz) max_scratchpad_size = sz;
    }
    auto scratchpad_md
            = memory::desc({static_cast<memory::dim>(max_scratchpad_size)},
                    memory::data_type::u8, memory::format_tag::a);

    // allocate intermediate memory
    auto m_scratchpad = memory(scratchpad_md, eng);

    std::unordered_map<int, memory> bmm1_args
            = {{DNNL_ARG_SRC, query}, {DNNL_ARG_WEIGHTS, key_dequantized},
                    {DNNL_ARG_DST, score}, {DNNL_ARG_SCRATCHPAD, m_scratchpad}};

    if (scale_dt != mdt::undef) {
        bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                = scale_f32;
        if (p.mask != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1]
                    = mask_f32;
        }
    } else {
        if (p.mask != mask_type::no_mask) {
            bmm1_args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1]
                    = mask_f32;
        }
    }

    const auto loop = [&]() {
        // each primitive will use all threads
        bmm1_prim.execute(strm, bmm1_args);

        //strm.wait();
        //print_mem(score, "score");

        //strm.wait();
        softmax_prim.execute(strm,
                {{DNNL_ARG_SRC, score}, {DNNL_ARG_DST, score2},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
        //strm.wait();
        //print_mem(score2, "score2");

        bmm2_prim.execute(strm,
                {{DNNL_ARG_SRC, score2}, {DNNL_ARG_WEIGHTS, value_dequantized},
                        {DNNL_ARG_DST, output},
                        {DNNL_ARG_SCRATCHPAD, m_scratchpad}});
    };

    // Warmup run.
    // Execute primitives of sdpa.
    loop();
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

GPU_TEST_P(sdpa_test_t, compare) {
    memory::data_type scale_dt = t.m_query_test.get_desc().get_data_type();
    //memory::data_type scale_dt = memory::data_type::undef;
    bool invert_scale = true;

    using namespace dnnl::impl;
    auto mask = t.m_mask.get_desc();

    memory::desc *mask_ptr = nullptr;

    switch (p.mask) {
        case mask_type::no_mask:
        case mask_type::causal: mask_ptr = nullptr; break;
        case mask_type::oneD:
        case mask_type::twoD: mask_ptr = &mask; break;
    }

    auto sdpa_quantized_pd
            = sdpa::primitive_desc(eng, t.m_query_test.get_desc(),
                    p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                          : t.m_key_quantized.get_desc(),
                    t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
                    t.m_output_quantized.get_desc(), invert_scale, p.head_num,
                    p.mask == mask_type::causal, t.sdpa_attr_quantized,
                    t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);
    auto sdpa_quantized_p = sdpa(sdpa_quantized_pd);

    std::unordered_map<int, memory> s8_args = {{{DNNL_ARG_QUERIES, t.m_query},
            {DNNL_ARG_VALUES, t.m_value_quantized},
            {DNNL_ARG_DST, t.m_output_quantized}}};

    if (p.with_key_transposed) {
        s8_args[DNNL_ARG_KEYS] = t.m_key_t_quantized;
    } else {
        s8_args[DNNL_ARG_KEYS] = t.m_key_quantized;
    }
    if (scale_dt != mdt::undef) { s8_args[DNNL_ARG_SCALE] = t.m_scale; }

    bool k_is_16_bit_float = ((p.kdt == mdt::f16) || (p.kdt == mdt::bf16));
    bool v_is_16_bit_float = ((p.vdt == mdt::f16) || (p.vdt == mdt::bf16));
    if (!k_is_16_bit_float && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
    }
    if (!v_is_16_bit_float && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = t.m_value_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES] = t.m_value_zp;
    }
    if (mask_ptr) { s8_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    std::unordered_map<int, memory> f16_args
            = {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, t.m_key},
                    {DNNL_ARG_VALUES, t.m_value}, {DNNL_ARG_DST, t.m_output}};
    if (scale_dt != mdt::undef) { f16_args[DNNL_ARG_SCALE] = t.m_scale; }
    if (mask_ptr) { f16_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    auto loop_quantized = [&] { sdpa_quantized_p.execute(strm, s8_args); };
    loop_quantized();
    strm.wait();
    prim_sdpa_quant(p, t, eng, strm, t.m_query,
            p.with_key_transposed ? t.m_key_t_quantized : t.m_key_quantized,
            t.m_key_scales, t.m_key_zp, scale_dt, t.m_scale, t.m_mask,
            t.m_value_quantized, t.m_value_scales, t.m_value_zp, t.m_output,
            invert_scale);
    strm.wait();

    strm.wait();

#if 0
    if (::getenv("SKIP_CHECK")) return;
#endif
    if (t.m_output.get_desc().get_data_type() == mdt::f16)
        check_memory<float16_t>(t.m_output, t.m_output_quantized);
    else if (t.m_output.get_desc().get_data_type() == mdt::bf16)
        check_memory<bfloat16_t>(t.m_output, t.m_output_quantized);

#if 0
    for (auto &kv : hist) {
        for (auto &kv2 : kv.second) {
            printf("hist[%d][%d] = %d\n", kv.first, kv2.first, kv2.second);
        }
    }
#endif
}
std::vector<std::chrono::microseconds> timeit(
        const std::function<void()> &func, dnnl::stream &str, int iterations) {
    using namespace std::chrono;
    func();
    func();
    std::vector<std::chrono::microseconds> times;
    for (int j = 0; j < 5; j++) {
        auto e = steady_clock::now();
        str.wait();
        auto s = steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            func();
        }
        str.wait();
        e = steady_clock::now();
        times.push_back(std::chrono::duration_cast<microseconds>(e - s));
    }
    return times;
}

GPU_TEST_P(sdpa_test_t, perf) {
    memory::data_type scale_dt = memory::data_type::f16;
    //memory::data_type scale_dt = memory::data_type::undef;
    bool invert_scale = true;

    using namespace dnnl::impl;
    auto mask = t.m_mask.get_desc();

    memory::desc *mask_ptr = nullptr;

    switch (p.mask) {
        case mask_type::no_mask:
        case mask_type::causal: mask_ptr = nullptr; break;
        case mask_type::oneD:
        case mask_type::twoD: mask_ptr = &mask; break;
    }

    auto sdpaf16_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
            p.with_key_transposed ? t.m_key_t.get_desc() : t.m_key.get_desc(),
            t.m_value.get_desc(), mask_ptr, scale_dt, t.m_output.get_desc(),
            invert_scale, p.head_num, p.mask == mask_type::causal, t.sdpa_attr);
    auto sdpaf16_p = sdpa(sdpaf16_pd);

    auto sdpa_quantized_pd = sdpa::primitive_desc(eng, t.m_query.get_desc(),
            p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                  : t.m_key_quantized.get_desc(),
            t.m_value_quantized.get_desc(), mask_ptr, scale_dt,
            t.m_output_quantized.get_desc(), invert_scale, p.head_num,
            p.mask == mask_type::causal, t.sdpa_attr_quantized,
            t.sdpa_kq_attr_quantized, t.sdpa_vs_attr_quantized);
    auto sdpa_quantized_p = sdpa(sdpa_quantized_pd);

    std::unordered_map<int, memory> s8_args = {{{DNNL_ARG_QUERIES, t.m_query},
            {DNNL_ARG_VALUES, t.m_value_quantized},
            {DNNL_ARG_DST, t.m_output_quantized}}};

    if (p.with_key_transposed) {
        s8_args[DNNL_ARG_KEYS] = t.m_key_t_quantized;
    } else {
        s8_args[DNNL_ARG_KEYS] = t.m_key_quantized;
    }
    if (scale_dt != mdt::undef) { s8_args[DNNL_ARG_SCALE] = t.m_scale; }

    if (p.kdt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_KEYS] = t.m_key_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_KEYS] = t.m_key_zp;
    }
    if (p.vdt != mdt::f16 && p.qtype != quantize_type::no_quantization) {
        s8_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_VALUES] = t.m_value_scales;
        s8_args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_VALUES] = t.m_value_zp;
    }
    if (mask_ptr) { s8_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    auto loop_quantized = [&] { sdpa_quantized_p.execute(strm, s8_args); };

    /// Dequantize reorder for key
    memory key_dequantized;
    dnnl::reorder key_dequantize_prim;
    bool dequantize_k = p.kdt != mdt::f16 && p.kdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization;
    if (dequantize_k) {
        std::tie(key_dequantize_prim, key_dequantized)
                = dequantize_prim(eng, mdt::f16,
                        p.with_key_transposed ? t.m_key_t_quantized.get_desc()
                                              : t.m_key_quantized.get_desc(),
                        t.kq_mask, t.kq_groups, p.ksdt, p.kzpdt,
                        (p.with_key_transposed ? memory::format_tag::abdc
                                               : memory::format_tag::abcd));
    } else {
        key_dequantized = p.with_key_transposed ? t.m_key_t_quantized
                                                : t.m_key_quantized;
    }

    /// Dequantize reorder for value
    memory value_dequantized;
    dnnl::reorder value_dequantize_prim;
    bool dequantize_v = p.vdt != mdt::f16 && p.vdt != mdt::bf16
            && p.qtype != quantize_type::no_quantization;
    if (dequantize_v) {
        std::tie(value_dequantize_prim, value_dequantized)
                = dequantize_prim(eng, mdt::f16, t.m_value_quantized.get_desc(),
                        t.vs_mask, t.vs_groups, p.vsdt, p.vzpdt);
    } else {
        value_dequantized = t.m_value_quantized;
    }

    std::unordered_map<int, memory> f16_args
            = {{DNNL_ARG_QUERIES, t.m_query}, {DNNL_ARG_KEYS, key_dequantized},
                    {DNNL_ARG_VALUES, value_dequantized},
                    {DNNL_ARG_DST, t.m_output_quantized}};
    if (scale_dt != mdt::undef) { f16_args[DNNL_ARG_SCALE] = t.m_scale; }
    if (mask_ptr) { f16_args[DNNL_ARG_ATTN_MASK] = t.m_mask; }

    auto loop_sdpa_f16 = [&] { sdpaf16_p.execute(strm, f16_args); };
    auto loop_reorder_sdpa = [&] {
        if (dequantize_k) {
            key_dequantize_prim.execute(strm,
                    {{DNNL_ARG_FROM, t.m_key}, {DNNL_ARG_TO, key_dequantized},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM,
                                    t.m_key_scales},
                            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM,
                                    t.m_key_zp}});
        }

        if (dequantize_v) {
            value_dequantize_prim.execute(strm,
                    {{DNNL_ARG_FROM, t.m_value},
                            {DNNL_ARG_TO, value_dequantized},
                            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM,
                                    t.m_value_scales},
                            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM,
                                    t.m_value_zp}});
        }

        sdpaf16_p.execute(strm, f16_args);
    };

    int iterations = 100;
    auto quantized_time = timeit(loop_quantized, strm, iterations);
    auto reorder_time = timeit(loop_reorder_sdpa, strm, iterations);
    auto sdpa_f16_time = timeit(loop_sdpa_f16, strm, iterations);

    auto min_time = [](const std::vector<std::chrono::microseconds> &a) {
        return *std::min_element(a.begin(), a.end());
    };

    std::cout << print_row(p) << "|"
              << min_time(quantized_time).count() / float(iterations) << "|"
              << min_time(reorder_time).count() / float(iterations) << "|"
              << min_time(sdpa_f16_time).count() / float(iterations) << "|"
              << std::endl;
    strm.wait();
}
