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
#include "test_utils.hpp"

using dnnl::memory;
using mdt = memory::data_type;

memory::dim product(const std::vector<int64_t> &dims) {
    return dims.empty() ? 0
                        : std::accumulate(dims.begin(), dims.end(),
                                (memory::dim)1, std::multiplies<memory::dim>());
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

void print_mem(const dnnl::memory &mem, const std::string &name) {
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
