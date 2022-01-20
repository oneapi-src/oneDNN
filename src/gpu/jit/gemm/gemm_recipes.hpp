/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
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

#ifndef GPU_JIT_GEMM_RECIPES_HPP
#define GPU_JIT_GEMM_RECIPES_HPP

#include "gpu/jit/ngen/ngen_core.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {
namespace {

struct gemm_recipe_t {
    ngen::HW hw;
    char precisions[4];
    char layouts[4];
    struct extra_t {
        int aCP, bCP;
        int aAlign, bAlign;

        extra_t() : aCP(1), bCP(1), aAlign(1), bAlign(1) {}
    } extra;
    int unrollM, unrollN;
    const char *strategyString;
    struct {
        int a, b, c;
    } unused;
    char tag;
};

inline namespace {
static gemm_recipe_t::extra_t make_crosspack(int a, int b) {
    gemm_recipe_t::extra_t result;
    result.aCP = a;
    result.bCP = b;
    return result;
}
static gemm_recipe_t::extra_t make_align(int a, int b) {
    gemm_recipe_t::extra_t result;
    result.aAlign = a;
    result.bAlign = b;
    return result;
}
} // namespace

// clang-format off
const gemm_recipe_t gemm_recipes[] = {
    {ngen::HW::Gen9, "SSS", "NNN", {}, 8,  4,  "ab8x2 ab16x2 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 8,  8,  "ab8x2 ab16x2 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 16, 8,  "ab4x2 ab16x2 ab acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 16, 16, "ab8 ab16 ab acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 16, 32, "ab2 as4 ab acb", {1024, 8192, 1024}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 8,  "ab4x2 ab16x2 ab acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 12, "ab1x2 ab16 ab acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 16, "ab1x2 ab16 ab acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 32, 16, "ab2x2 as4x2 as acb", {8192, 1024, 1024}, {}},
    {ngen::HW::Gen9, "SSS", "NNN", {}, 64, 8,  "ab1x2 ab16 ab acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 8,  4,  "as8x2 ab16x2 as cb1 wg 8x1 acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 8,  8,  "ab2x2 as8x2 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 8,  "ab4x2 ab4x2 ab cb1 wg 8x1 acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 16, "ab4x2 ab16 ab acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 16, 32, "ab1x2 ab8 ab acb", {1024, 8192, 1024}, {}},
    {ngen::HW::Gen9, "SSS", "NTN", {}, 32, 16, "ab4 ab4x2 ab acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "NTT", {}, 16, 32, "ab2x2 ab2x2 as k8 acb ns64", {}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 8,  4,  "as8x2 ab32 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 8,  8,  "as8x2 ab32 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 8,  "ab16 as4x2 as cb1 wg 8x1 acb nmk", {}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 16, "as1x2 ab16 ab ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 16, "ab16 as4 as acb nmk", {8192, 1024, 1024}, {}},
    {ngen::HW::Gen9, "SSS", "TNN", {}, 16, 32, "as4 su4 ab k8 da cs", {}, {}},
    {ngen::HW::Gen9, "SSS", "TTN", {}, 12, 32, "ab16/8 ab4x2 su k32 acb", {}, {}},
    {ngen::HW::Gen9, "SSS", "TTN", {}, 16, 32, "as4 ab4 su k8 da cs", {}, {}},
    {ngen::HW::Gen9, "SSS", "TTT", {}, 16, 32, "as4 ab4 ab k8 da cs", {}, {}},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 8,  8,  "ab16x2 as16x2 ab l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 16, 16, "ab8x2 ab32x2 ab l4 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 32, 16, "ab8x2 ab32/8 ab l4 acb nmk", {}, {}},
    {ngen::HW::Gen9, "HHH", "NNN", {}, 32, 32, "ab4x2 as8 ab k16 l4 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 l4 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "NTN", {}, 32, 32, "ab2x2 ab2x2 ab k4 l4 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 8,  8,  "as32x2 as16x2 ab l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 16, 16, "as2x2 ab32 ab l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 32, 16, "as8 ab16 ab l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "TNN", {}, 32, 32, "as2 as8 ab l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "HHH", "TTN", {}, 32, 16, "as8 ab4 ab k8 ra8 l4 cs", {}, {}},
    {ngen::HW::Gen9, "HHH", "TTN", {}, 32, 32, "as8 ab2 ab k8 ra8 l4 cs", {}, {}},
    {ngen::HW::Gen9, "OOI", "NNN", {}, 32, 16, "ab4/2x2 as2x2 as l4 cb1 wg 8x1 acb nmk", {}, {}},
    {ngen::HW::Gen9, "OOI", "NTN", {}, 32, 16, "ab2 ab1x2 as l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen9, "OOI", "TNN", {}, 16, 16, "as8 as8 as l4 cab1 k32 wg 2x4 acb", {}, {}},
    {ngen::HW::Gen9, "OOI", "TTN", {}, 16, 32, "as2x2 ab8/2x2 as l4 ca1 wg 1x8 acb", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 8,  4,  "ab16 ab32/16x2 ab ca1 wg 2x8 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 8,  8,  "ab32 ab32 ab ca1 wg 2x8 vnc", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 16, 4,  "ab4x2 as4x2 ab int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 16, 8,  "ab2 ab32 ab ca1 wg 2x8 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 16, 16, "ab8 ab8 ab cab1 wg 4x4 vnc", {8192, 1024, 1024}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 8,  "ab8 ab8 ab int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 8,  "ab2 ab32 ab ca1 wg 2x8 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 12, "ab4x2 ab16/8 ab k32 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 16, "ab4 ab8 ab int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 16, "ab4 ab8 ab cb1 wg 8x2 int nmk", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NNN", {}, 32, 16, "ab2x2 ab8 as cb1 wg 8x2 int nmk", {1024, 8192, 1024}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 8,  4,  "ab4 ab16 ab cab1 wg 4x4 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 8,  8,  "ab4 ab16 ab cab1 wg 4x4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 8,  "ab4x2 ab8 ab cb1 wg 8x2 int nmk", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 16, "ab4 ab4x2 ab vnc nmk", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 16, 32, "ab4x2 ab2x2 ab k8 int ns64", {8192, 1024, 1024}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 int ns64", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "NTT", {}, 16, 32, "ab4x2 ab2x2 ab k8 int ns64", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 8,  4,  "ab16 ab32 ab ca1 wg 2x8 int ek", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 8,  8,  "ab32 ab8 as cab1 wg 4x4 vnc ek", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 16, 8,  "ab8x2 as8x2 as cab1 wg 4x4 int bk1024", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 16, 16, "ab8 ab8 ab k16 cab1 wg 4x4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 16, 16, "ab16/8 as4 as cab1 wg 4x4 vnc", {1024, 8192, 1024}, {}},
    {ngen::HW::Gen12LP, "SSS", "TNN", {}, 32,  8, "ab8 as8 ab cab1 wg 4x4 int bk1024", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TTN", {}, 12, 32, "ab16/8 ab4x2 su k32 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TTN", {}, 16, 32, "as4 ab4 su k8 int", {}, {}},
    {ngen::HW::Gen12LP, "SSS", "TTT", {}, 16, 32, "as4 ab4 ab k8 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32,  4, "ab4x2 as8x2 ab l4 int nmk", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32,  8, "ab2x2 as16 ab l4 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32, 16, "ab4x2 ab32/8 ab k64 l4 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 32, 32, "ab2x2 as8x2 ab k16 l4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NNN", {}, 64, 16, "sb2/1 su8x2 ab l4 ca1 wg 2x8 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NTN", {}, 32, 16, "ab2x2 ab4x2 ab k8 l4 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "NTN", {}, 32, 32, "ab2x2 ab2x2 ab k4 l4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "TNN", {}, 16, 16, "as8x2 ab32 ab l4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "TNN", {}, 32, 16, "ab4 ab4 ab k8 vnc cab1 wg 4x4", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "TNN", {}, 32, 32, "as4 as8 ab k8 ra4 l4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "TTN", {}, 32, 16, "as8 ab4x2 ab k16 ra8 l4 int", {}, {}},
    {ngen::HW::Gen12LP, "HHH", "TTN", {}, 32, 32, "as8 ab2x2 ab k16 ra8 l4 vnc", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "NNN", {}, 32,  4, "sb8x2 su16x2 sb l4 ca1 wg 2x8 int", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "NNN", {}, 32,  8, "ab4 as32 ab l4 cab1 wg 4x4 int", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "NNN", {}, 64,  8, "sb4x2 su16 sb l4 int", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "NNN", {}, 32, 16, "sb4 sb8 sb l4 int k32 cab1 wg 4x4", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "NTN", {}, 16, 32, "sb8 sb4 sb l4 int k16 cab1 wg 4x4", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "TNN", {}, 16, 16, "sb8x2 sb8x2 sb l4 vnc k32 cab1 wg 4x4", {}, {}},
    {ngen::HW::Gen12LP, "OOI", "TTN", {}, 16, 32, "sb8 sb4 sb l4 int k32 cab1 wg 4x4 fn nmk", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 64, 8,  "sb1x2 su4/2x2 sb ca1 wg 2x8 cs di hi bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 16, "ab4/2 ab8 ab cb1 wg 8x2 cs di nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 16, "sb4/2x2 su1 su cb1 wg 16x2 cs di nmk", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 8,  "sb4 sb8 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 32, 4,  "sb8 su8x2 sb ca1 wg 2x8 cs di bm4096 bn4096 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 16, "sb8 sb16/8 sb cs di", {16384, 2048, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 8,  "sb16 sb16 sb cab1 wg 4x4 cs di bm2048 bn2048 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 16, 4,  "sb8 su32 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di bm2048 bn2048 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NNN", {}, 8,  4,  "sb16 sb32 sb cab1 wg 2x8 cs di bm8192 bn8192 bk2048 ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 32, 16, "sb4/2x2 sb4x2 sb cs di hi bk2048", {}, {}},   // faster for large sizes: sb4 su8/4 sb cab1 wg 4x4 cs di
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 32, "sb2x2 sb4/2x2 sb cs di nmk", {16384, 4096, 4096}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 16, "sb4x3 sb4x3 sb cs di cab1 wg 4x4 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 16, "ab4x2 ab4x2 ab cs di cab1 wg 4x4 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 16, 8,  "sb16 sb16 sb cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "SSS", "NTN", {}, 8,  8,  "sb2x2 sb2x2 ab wg 2x2x16 kb kr cs di ar bk64", {}, 'K'},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 16, 16, "sb8 sb8 su cab1 wg 4x4 cs di hi bk2048 ek", {2048, 16384, 2048}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  16, "sb16 sb16 sb cab1 wg 4x4 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  8,  "sb16 sb16 sb cab1 wg 4x4 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TNN", {}, 8,  4,  "sb16 sb32 sb cab1 wg 2x8 cs di ek", {}, {}},
    {ngen::HW::XeHP, "SSS", "TTN", {}, 8,  32, "sb8 sb4 sb cab1 wg 4x4 cs di bm4096 bn4096 bk2048", {}, {}},
    {ngen::HW::XeHP, "SSS", "TTN", {}, 8,  64, "su4/2x2 sb1x2 su cb1 wg 8x2 cs di fn hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 32, "ab16x3 ab16x3 ab fs sc bo bk2048", {}, {}},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 4x4 bo bk4096", {}, 'A'},
    {ngen::HW::XeHP, "HHS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 8x4 bo bk4096", {}, 'B'},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 32, "ab4/1x2 as4/1x2 ab l4 cb1 wg 8x2 cs nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "sb16 su4 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32,  4, "sb8 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 16,  4, "sb4 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NTN", {}, 32, 32, "ab4/1x2 ab2/1x2 ab l4 cs hi bk2048", {}, {}},  // slightly faster: ab8/2 ab4/1x2 ab l4 ca1 wg 2x8 cs np
    {ngen::HW::XeHP, "HHH", "NTN", {}, 32, 16, "sb8 sb16 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16, 16, "su8 su32 sb l4 cab1 wg 4x4 cs di hi bk2048", {}, {}},
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16,  8, "su32 su16 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "TNN", {}, 32,  1, "su8 sb32x2 sb wg 2x1x16 kr l4 cs di", {}, 'k'},
    {ngen::HW::XeHP, "HHH", "TTN", {}, 32, 32, "as4/1x2 ab4/1x2 ab l4 ca1 wg 2x8 cs nmk hi bk2048", {}, {}},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 32, "ab16x3 ab16x3 ab fs sc bo bk4096", {}, {}},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "ab16 ab16 ab fs wg 4x4 bo bk8192", {}, 'A'},
    {ngen::HW::XeHP, "OOI", "ABN", make_crosspack(4, 32), 32, 48, "ab16 ab16 ab fs wg 8x4 bo bk8192", {}, 'B'},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 32, 16, "sb8 sb32 sb l4 cab1 wg 4x4 cs di pab", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 32,  4, "sb8x2 su16x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NNN", {}, 16,  4, "sb16 su32x2 sb l4 ca1 wg 2x8 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "NTN", {}, 32, 16, "sb8x2 sb8x2 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "TNN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "OOI", "TNN", {}, 32,  1, "su16 sb64x2 sb wg 2x1x16 l4 cs kr", {}, 'k'},
    {ngen::HW::XeHP, "OOI", "TTN", {}, 16, 16, "sb32 sb32 sb l4 cab1 wg 4x4 cs di", {}, {}},
    {ngen::HW::XeHP, "HHH", "NNN", {}, 16,  4, "ab2x2 as8 ab l4 cs di", {}, {}},                        // DLRM
    {ngen::HW::XeHP, "HHH", "TNN", {}, 16,  8, "sb16 sb16 ab cab2 wg 2x4 cs pab", {}, {}},              // DLRM
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 32, "ab16x3 ab16x3 ab fs sc bo bk2048", {}, {}},
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 4x4 bo bk4096", {}, 'A'},
    {ngen::HW::XeHP, "BBS", "ABN", make_crosspack(2, 16), 32, 48, "ab16 ab16 ab fs wg 8x4 bo bk4096", {}, 'B'},
    {ngen::HW::XeHP, "BBS", "NNN", {}, 32,  8, "sb16 sb16 ab cab1 wg 4x4 fn nmk cs pab", {}, {}},       // DLRM
    {ngen::HW::XeHP, "BBS", "NTN", {}, 16, 16, "sb1x4 sb1x4 sb l4 cs di nmk fn pab", {}, {}},           // DLRM
    {ngen::HW::XeHP, "BBS", "NTN", {},  8,  8, "sb4x2 sb4x2 ab cs di wg 2x2x16 kr kb bk64", {}, 'K'},   // DCRNN
    {ngen::HW::XeHP, "BBS", "TNN", {}, 32,  8, "sb16 sb16 ab ca2 wg 1x4 fn nmk cs pab", {}, {}},        // DLRM
    {ngen::HW::XeHP, "BBS", "TNN", {}, 32,  4, "sb16 su16 sb l4 cab1 wg 2x8 cs di", {}, {}},            // DCRNN
    {ngen::HW::XeHP, "BBS", "TTN", {}, 8,  32, "sb16 sb16 as cab1 wg 4x4 cs pab", {}, {}},              // DLRM
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "ab16/8 as16 ab l4 cab1 wg 4x4 cs di", {}, 'A'},         // BERT
    {ngen::HW::XeHP, "HHH", "NNN", {}, 32, 16, "ab8 ab16 ab l4 ca1 wg 2x8 cs di", {}, 'B'},             // BERT
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 64, 32, "aB8/4+B8@2/0 aS8+S1,16@2/0 aB nse di wg 4x8 bo pt sr kc8 sn sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 16, 8,  "aB8x2+B8@2 aS8+S1,16@2 aB nse di wg 4x8 bo pt sr kc8 sn sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NNN", {}, 64, 16, "ab2x2 ab16+s1,32@3 ab cs di wg 2x16 hi sn bk4096", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NTN", {}, 64, 32, "aB8/4x2+B8@2 aB8/4x2+B8@2 aB nse di wg 4x8 bo pt sr sb256 kc8 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "NTN", {}, 16, 16, "aB8x2+B8@2 aB8x2+B8@2 aB nse di wg 4x8 bo pt sr kc8 sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 64, 32, "aS8+S1,16@2/0 aS16+S1,16@2/0 aB nse di wg 4x8 bo pt sr kc8 sm sn sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 16, 16, "aS16x2+S1,16@3 aS16x2+S1,16@3 aB nse di wg 4x8 bo pt sr kc8 sm sn sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TNN", {}, 16, 8,  "aS16+S1,16@2 aS16+S1,16@2 aB nse di wg 4x8 bo pt sr kc8 sm sn sb256 bk0", {}, {}},
    {ngen::HW::XeHPC, "SSS", "TTN", {}, 32, 64, "aS8+S1,16@2/0 aB8/4+B8@2/0 aU nse di wg 8x4 bo pt sr kc8 sm sb256 grf256 bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "ABN", make_crosspack(2, 16), 64, 32, "aB16+B16@3 aB16+B16@3 aB nse di sys grf256 af hi pt wg 4x8 sr sb256", {}, {}},
    {ngen::HW::XeHPC, "HHS", "ABN", make_crosspack(2, 16), 16, 16, "aB16x2+B16@3 aB16x2+B16@3 aB nse sb256 wg 4x8 sr di bo pt sys", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NNN", make_align(128, 128),  64, 32, "av32+B32@2 am32+S1,32@2 aB cs di sys grf256 af wg 4x8 bo sb256 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NNN", make_align(128, 128),  16, 16, "av32+B32@1 am64+S1,32@1 ab cs sb256 wg 8x4 sr di sn bo pt sys", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NTN", make_align(128, 128),  64, 32, "av16x2+B16@1/0 av16+B16@1 aB sys af wg 4x8 k64 cb1 grf256 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NTN", make_align(128, 128),  16, 16, "av16x2+B16@1/0 av16+B16@1 aB sys af wg 4x8 k64 cb1 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TNN", make_align(4,   128),  64, 32, "aS16+S1,32@1 am16+S1,32@1 aB cs di sys grf256 af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TNN", make_align(4,   128),  16, 16, "aS16+S1,32@1 am16+S1,32@1 aB cs di sys af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TTN", make_align(128, 128),  32, 64, "am32+S1,32@2 av32+B32@2 aS cs di sys grf256 af wg 8x4 bo sb256 sm bk0", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NNN", {}, 48, 32, "ab16 ab16 ab sys grf256 cab2 wg 4x4 sn l4", {}, {}},
    {ngen::HW::XeHPC, "HHS", "NTN", {}, 48, 32, "ab16 as16 ab sys grf256 cab2 wg 4x4 l4", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TNN", {}, 48, 32, "as16 ab16 ab sys grf256 cab2 wg 4x4 sn l4", {}, {}},
    {ngen::HW::XeHPC, "HHS", "TTN", {}, 48, 32, "as16 as16 ab sys grf256 cab2 wg 4x4 l4", {}, {}},
    {ngen::HW::XeHPC, "BBS", "ABN", make_crosspack(2, 16), 64, 32, "aB16+B16@3 aB16+B16@3 aB nse di sys grf256 af hi pt wg 4x8 sr sb256", {}, {}},
    {ngen::HW::XeHPC, "BBS", "ABN", make_crosspack(2, 16), 16, 16, "aB16x2+B16@3 aB16x2+B16@3 aB nse sb256 wg 4x8 sr di bo pt sys", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NNN", make_align(128, 128),  64, 32, "av32+B32@2 am32+S1,32@2 aB cs di sys grf256 af wg 4x8 bo sb256 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NNN", make_align(128, 128),  16, 16, "av32+B32@1 am64+S1,32@1 ab cs sb256 wg 8x4 sr di sn bo pt sys", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NTN", make_align(128, 128),  64, 32, "av16x2+B16@1/0 av16+B16@1 aB sys af wg 4x8 k64 cb1 grf256 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NTN", make_align(128, 128),  16, 16, "av16x2+B16@1/0 av16+B16@1 aB sys af wg 4x8 k64 cb1 bo sb256 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TNN", make_align(4,   128),  64, 32, "aS16+S1,32@1 am16+S1,32@1 aB cs di sys grf256 af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TNN", make_align(4,   128),  16, 16, "aS16+S1,32@1 am16+S1,32@1 aB cs di sys af wg 4x8 bo sb256 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TTN", make_align(128, 128),  32, 64, "am32+S1,32@2 av32+B32@2 aS cs di sys grf256 af wg 8x4 bo sb256 sm bk0", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NNN", {}, 48, 32, "ab16 ab16 ab sys grf256 cab2 wg 4x4 sn l4", {}, {}},
    {ngen::HW::XeHPC, "BBS", "NTN", {}, 48, 32, "ab16 as16 ab sys grf256 cab2 wg 4x4 l4", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TNN", {}, 48, 32, "as16 ab16 ab sys grf256 cab2 wg 4x4 sn l4", {}, {}},
    {ngen::HW::XeHPC, "BBS", "TTN", {}, 48, 32, "as16 as16 ab sys grf256 cab2 wg 4x4 l4", {}, {}},
    {ngen::HW::XeHPC, "OOI", "ABN", make_crosspack(4, 32), 64, 32, "aB32+B32@3 aB32+B32@3 aB nse di sys grf256 af hi pt wg 4x8 sr sb512", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NNN", make_align(128, 128),  64, 32, "av64+B64@2 am64+S1,64@2/0 aB cs di sys grf256 af wg 4x8 bo sb512 sn bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NTN", make_align(128, 128),  64, 32, "av32+B32@1 av32+B32@1 aB sys af wg 4x8 k128 cb1 grf256 bo sb512 pt bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TNN", make_align(4,   128),  64, 32, "aS32+S1,64@1 am32+S1,64@1 aB cs di sys grf256 af wg 4x8 bo sb512 sm sn bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TTN", make_align(128, 128),  32, 64, "am64+S1,64@2/0 av64+B64@2 aS cs di sys grf256 af wg 8x4 bo sb512 sm bk0", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NNN", {}, 48, 16, "ab32 ab32 ab sys grf256 cab2 wg 4x4 sn ek l4", {}, {}},
    {ngen::HW::XeHPC, "OOI", "NTN", {}, 48, 16, "ab32 as32 ab sys grf256 cab2 wg 4x4 ek l4", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TNN", {}, 48, 32, "as32 ab32 ab sys grf256 cab2 wg 4x4 sn ek l4", {}, {}},
    {ngen::HW::XeHPC, "OOI", "TTN", {}, 48, 32, "as32 as32 ab sys grf256 cab2 wg 4x4 ek l4", {}, {}},
};
// clang-format on

const int gemm_recipe_count = sizeof(gemm_recipes) / sizeof(gemm_recipes[0]);

} // anonymous namespace
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */
