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

#ifndef DNNL_TEST_INTERNAL_TEST_UTILS_HPP
#define DNNL_TEST_INTERNAL_TEST_UTILS_HPP

#include <dnnl_test_common.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include <random>

enum class quantize_type {
    no_quantization,
    per_tensor,
    per_tensor1,
    per_tensor3,
    per_token,
    per_token_with_groups
};

dnnl::memory::dim product(const std::vector<int64_t> &dims);

std::random_device &get_random_device();

std::mt19937 &get_generator();

void fill_random(std::vector<float> &out, const dnnl::memory::desc &desc,
        float minval = -3.f, float maxval = 4.f);

void fill_random_scales(
        std::vector<float> &out, const dnnl::memory::desc &desc);

void print_mem(const dnnl::memory &mem, const std::string &name = "");

void transpose(const dnnl::engine &eng, dnnl::memory &out, dnnl::memory &in);

void transpose_strides(
        const dnnl::engine &eng, dnnl::memory &out, dnnl::memory &in);

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
void write_to_dnnl_memory(const T *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

    dnnl::stream s(eng);
    if (eng.get_kind() == dnnl::engine::kind::gpu) {
        if (mem.get_desc().get_data_type() != dnnl_f32
                && std::is_same<T, float>::value) {
            dnnl::memory mem_f32_mem(
                    {mem.get_desc().get_dims(), dnnl::memory::data_type::f32,
                            mem.get_desc().get_strides()},
                    eng);
            write_to_dnnl_memory<float>((const float *)handle, mem_f32_mem);
            dnnl::reorder(mem_f32_mem, mem).execute(s, mem_f32_mem, mem);
            s.wait();
        } else if (mem.get_desc().get_data_type() != dnnl_s32
                && std::is_same<T, int>::value) {
            dnnl::memory mem_s32_mem(
                    {mem.get_desc().get_dims(), dnnl::memory::data_type::s32,
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
            dnnl::memory mem_u32_mem(
                    {mem.get_desc().get_dims(), dnnl::memory::data_type::s32,
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

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)handle)[i];
        return;
    }

    assert(!"not expected");
}

template <typename T>
void fill_random_quantized(std::vector<T> &out, const dnnl::memory::desc &desc,
        bool is_unsigned = false) {
    static std::vector<T> random_data_f;
    static std::vector<T> random_data_u;
    constexpr dnnl::memory::dim nrand = 2049;

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
        for (dnnl::memory::dim i = 0; i < elems; i += nrand) {
            size_t chunk = std::min(nrand, elems - i);
            std::memcpy(&out[i], random_data_u.data(), chunk * sizeof(T));
        }
    } else {
        for (dnnl::memory::dim i = 0; i < elems; i += nrand) {
            size_t chunk = std::min(nrand, elems - i);
            std::memcpy(&out[i], random_data_f.data(), chunk * sizeof(T));
        }
    }
}

template <typename T>
std::vector<float> dequantize(const std::vector<float> &input,
        dnnl::memory::desc &desc, dnnl::memory::desc &scale_md,
        const std::vector<T> &zero_points, const std::vector<float> &scales,
        int group_size, quantize_type qtype, dnnl::memory::dims groups,
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

        size_t ndims = dims.size();
        assert(ndims > 1); // TODO: will fail w/ndim == 1
        size_t lastdim = ndims - 1;
        size_t n2lastdim = lastdim - 1;

        if (qtype == quantize_type::no_quantization) { groups = {1, 1}; }
        if (qtype == quantize_type::per_token) {
            if (token_dim == 0) {
                groups = {dims[n2lastdim], 1};
            } else if (token_dim == 1) {
                groups = {1, dims[lastdim]};
            }
        }

        std::vector<size_t> idxs(ndims, 0);
        while (true) {
            size_t offset = 0;
            size_t scale_offset = 0;
            size_t zp_offset = 0;

            int group = 0;

            for (size_t i = 0; i < ndims; ++i) {
                if (groups[0] > 1) {
                    group = (i == n2lastdim) ? groups[0] : 1;
                    scale_offset += idxs[i] / group * scales_strides[i];

                    // set final stride = 1
                    auto zp_stride = (i == lastdim) ? 1 : zp_strides[i];
                    zp_offset += idxs[i] / group * zp_stride;
                } else if (groups[1] > 1) {
                    group = (i == lastdim) ? groups[1] : 1;
                    scale_offset += idxs[i] / group * scales_strides[i];
                    zp_offset += idxs[i] / group * zp_strides[i];
                }
                offset += idxs[i] * strides[i];
            }

            out[offset] = (input[offset] - zero_points[zp_offset])
                    * scales[scale_offset];

            int d = lastdim;
            while (d >= 0) {
                if (++idxs[d] < dims[d]) {
                    break;
                } else {
                    idxs[d--] = 0;
                }
            }
            if (d < 0) { break; }
        }
    }
    return out;
}

std::ostream &operator<<(std::ostream &ss, const quantize_type &qt);

std::ostream &operator<<(std::ostream &ss, const memory::data_type &dt);

#endif
