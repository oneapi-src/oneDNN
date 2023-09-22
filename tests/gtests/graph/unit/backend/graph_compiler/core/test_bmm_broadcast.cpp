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
 ******************************************************************************/

#include <fstream>
#include <iostream>
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/graph_op.hpp"
#include "compiler/ir/graph/lowering.hpp"
#include "compiler/ir/graph/pass/pass.hpp"
#include "compiler/ir/graph/transform/transform.hpp"
#include "compiler/jit/jit.hpp"
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

void run_single_bmm(const sc_dims &A_dims, const sc_dims &B_dims,
        const sc_dims &out_dims, test_buffer<float> &A_data,
        test_buffer<float> &B_data, test_buffer<float> &out_data) {
    auto ctx = std::make_shared<context_t>(*get_test_ctx());

    auto make_tensor = [](const sc_dims &shape) {
        return std::make_shared<graph_tensor>(nullptr, sc_data_format_t(),
                shape, sc_data_type_t(sc_data_etype::F32, 1));
    };

    sc_graph_t graph;
    auto ins0 = make_tensor(A_dims);
    auto ins1 = make_tensor(B_dims);
    auto outs0 = make_tensor(out_dims);
    any_map_t attrs({{"transpose_a", false}, {"transpose_b", false},
            {"output2d", false}, {"use_mmm", false}});
    auto in = graph.make_input({ins0, ins1});
    auto matmul = graph.make("matmul_core", in->get_outputs(), {outs0}, attrs);
    auto output = graph.make_output(matmul->get_outputs());

    graph_driver(graph, ctx);
    auto mod = lower_graph(ctx, graph, {output, in});
    auto fptr = jit_engine_t::make(ctx)->get_entry_func(mod, true);

    std::vector<float *> sc_args = {
            &out_data[0],
            &A_data[0],
            &B_data[0],
    };
    std::vector<generic_val> generic_args;
    for (unsigned i = 0; i < sc_args.size(); i++)
        generic_args.emplace_back(sc_args.at(i));
    fptr->call_generic_default(generic_args.data());
}

TEST(GCCore_CPU_test_bmm_broadcast, Case1) {
    REQUIRE_AVX2();
    sc_dims A_dims {1, 16, 16, 128}, B_dims {1, 1, 128, 16},
            out_dims {1, 16, 16, 16};
    sc_dims B_bc_dims {1, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < B_size; ++j) {
            B_bc_data[i * B_size + j] = B_data[j];
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_dims, B_bc_dims, out_dims, A_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case2) {
    REQUIRE_AVX2();
    sc_dims A_dims {8, 16, 16, 128}, B_dims {8, 1, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims B_bc_dims {8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            for (size_t k = 0; k < 128 * 16; ++k) {
                B_bc_data[i * 16 * 128 * 16 + j * 128 * 16 + k]
                        = B_data[i * 128 * 16 + k];
            }
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_dims, B_bc_dims, out_dims, A_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case3) {
    REQUIRE_AVX2();
    sc_dims A_dims {8, 16, 16, 128}, B_dims {1, 16, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims B_bc_dims {8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < B_size; ++j) {
            B_bc_data[i * B_size + j] = B_data[j];
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_dims, B_bc_dims, out_dims, A_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case4) {
    REQUIRE_AVX2();
    sc_dims A_dims {8, 16, 16, 128}, B_dims {1, 1, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims B_bc_dims {8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            for (size_t k = 0; k < B_size; ++k) {
                B_bc_data[i * 16 * B_size + j * B_size + k] = B_data[k];
            }
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_dims, B_bc_dims, out_dims, A_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case5) {
    REQUIRE_AVX2();
    sc_dims A_dims {1, 16, 16, 128}, B_dims {8, 1, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims A_bc_dims {8, 16, 16, 128}, B_bc_dims {8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t A_bc_size = test_utils::product(A_bc_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto A_bc_data = alloc_array<float>(A_bc_size);
    // broadcast A_data to A_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < A_size; ++j) {
            A_bc_data[i * A_size + j] = A_data[j];
        }
    }
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            for (size_t k = 0; k < 128 * 16; ++k) {
                B_bc_data[i * 16 * 128 * 16 + j * 128 * 16 + k]
                        = B_data[i * 128 * 16 + k];
            }
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_bc_dims, B_bc_dims, out_dims, A_bc_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case6) {
    REQUIRE_AVX2();
    sc_dims A_dims {8, 1, 16, 128}, B_dims {1, 16, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims A_bc_dims {8, 16, 16, 128}, B_bc_dims {8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t A_bc_size = test_utils::product(A_bc_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto A_bc_data = alloc_array<float>(A_bc_size);
    // broadcast A_data to A_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            for (size_t k = 0; k < 16 * 128; ++k) {
                A_bc_data[i * 16 * 16 * 128 + j * 16 * 128 + k]
                        = A_data[i * 16 * 128 + k];
            }
        }
    }
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < B_size; ++j) {
            B_bc_data[i * B_size + j] = B_data[j];
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_bc_dims, B_bc_dims, out_dims, A_bc_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case7) {
    REQUIRE_AVX2();
    sc_dims A_dims {1, 16, 16, 128}, B_dims {32, 8, 1, 128, 16},
            out_dims {32, 8, 16, 16, 16};
    sc_dims A_bc_dims {32, 8, 16, 16, 128}, B_bc_dims {32, 8, 16, 128, 16};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t A_bc_size = test_utils::product(A_bc_dims);
    const size_t B_bc_size = test_utils::product(B_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto A_bc_data = alloc_array<float>(A_bc_size);
    // broadcast A_data to A_bc_data manually
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            for (size_t k = 0; k < A_size; ++k) {
                A_bc_data[i * 8 * A_size + j * A_size + k] = A_data[k];
            }
        }
    }
    auto B_bc_data = alloc_array<float>(B_bc_size);
    // broadcast B_data to B_bc_data manually
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            for (size_t k = 0; k < 16; ++k) {
                // replicate 32 * 8 * 16 times.
                for (size_t l = 0; l < 128 * 16; ++l) {
                    B_bc_data[i * 8 * 16 * 128 * 16 + j * 16 * 128 * 16
                            + k * 128 * 16 + l]
                            = B_data[i * 8 * 128 * 16 + j * 128 * 16 + l];
                }
            }
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_bc_dims, B_bc_dims, out_dims, A_bc_data, B_bc_data, ref_out_data);
    test_utils::compare_data(out_data, ref_out_data, 1e-3f, 1e-4f);
}

TEST(GCCore_CPU_test_bmm_broadcast, Case8) {
    REQUIRE_AVX2();
    sc_dims A_dims {1, 16, 128}, B_dims {8, 16, 128, 16},
            out_dims {8, 16, 16, 16};
    sc_dims A_bc_dims {8, 16, 16, 128};
    const size_t A_size = test_utils::product(A_dims);
    const size_t B_size = test_utils::product(B_dims);
    const size_t A_bc_size = test_utils::product(A_bc_dims);
    const size_t out_size = test_utils::product(out_dims);
    auto A_data = alloc_array<float>(A_size);
    auto B_data = alloc_array<float>(B_size);
    auto A_bc_data = alloc_array<float>(A_bc_size);
    // broadcast A_data to A_bc_data manually
    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            for (size_t k = 0; k < A_size; ++k) {
                A_bc_data[i * 16 * A_size + j * A_size + k] = A_data[k];
            }
        }
    }
    auto out_data = alloc_array<float>(out_size);
    auto ref1_out_data = alloc_array<float>(out_size);
    auto ref2_out_data = alloc_array<float>(out_size);

    run_single_bmm(A_dims, B_dims, out_dims, A_data, B_data, out_data);
    run_single_bmm(
            A_bc_dims, B_dims, out_dims, A_bc_data, B_data, ref1_out_data);
    test_utils::compare_data(out_data, ref1_out_data, 1e-3f, 1e-4f);

    sc_dims A_dims_squeezed {16, 128}; // original: {1, 16, 128}
    run_single_bmm(
            A_dims_squeezed, B_dims, out_dims, A_data, B_data, ref2_out_data);
    test_utils::compare_data(out_data, ref2_out_data, 1e-3f, 1e-4f);
}
