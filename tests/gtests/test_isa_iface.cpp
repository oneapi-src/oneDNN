/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "gtest/gtest.h"

#include "dnnl.hpp"
#if 0 // original, for reference...
#include "src/cpu/cpu_isa_traits.hpp"

namespace dnnl {

class isa_set_once_test : public ::testing::Test {};
TEST(isa_set_once_test, TestISASetOnce) {
    auto st = set_max_cpu_isa(cpu_isa::sse41);
    ASSERT_TRUE(st == status::success || st == status::unimplemented);
    ASSERT_TRUE(impl::cpu::mayiuse(impl::cpu::sse41));
    st = set_max_cpu_isa(cpu_isa::sse41);
    ASSERT_TRUE(st == status::invalid_arguments || st == status::unimplemented);
};

} // namespace dnnl
#endif

#include "dnnl_debug.h"
#include "src/cpu/cpu_isa_traits.hpp"

//#include <iomanip>
//#include <iostream>
//using std::cout;
//using std::dec;
//using std::endl;
//using std::hex;

namespace dnnl {

//
// choose a dnnl_cpu_isa and cpu_isa type good for the build target
//
#if TARGET_X86_JIT && CPU_ISA >= CPU_ISA_SSE41
/** dnnl original value, include/dnnl.hpp and include/dnnl_types.h */
#define PUBLIC_ISA sse41
/** internal flags value, src/cpu/cpu_isa_traits.hpp .
 * Note that these are also determinable by searching for */
#define ISA_MASK dnnl::impl::cpu::sse41
//#warning "isa_set_once_test using dnnl_cpu_isa_sse41"

#elif TARGET_X86_JIT
#define PUBLIC_ISA any /*jit, but not even sse41 - no jit impls for this yet*/
#define ISA_MASK dnnl::impl::cpu::x86_any
#warning "isa_set_once_test using dnnl_cpu_isa_any"

#elif TARGET_X86 /*x86, without jit*/
#define PUBLIC_ISA vanilla
#define ISA_MASK dnnl::impl::cpu::vanilla
#warning "isa_set_once_test using dnnl_cpu_isa_vanilla"

#elif TARGET_VE
#define PUBLIC_ISA vanilla
#define ISA_MASK dnnl::impl::cpu::ve_common
#warning "isa_set_once_test using dnnl_cpu_isa_any"
#else
#error "Please choose an appropriate dnnl_cpu_isa_FOO for this build target"
#endif

#define STR_(...) #__VA_ARGS__
#define STR(...) STR_(__VA_ARGS__)

// NOTE:
// - set_max_cpu_isa uses a dnnl_cpu_isa_foo (equiv to dnnl::foo) argument.
// - Other functions use internal mask/flags values.
// - Cmake uses it's own compile-time "isa/feature limit".
//
// Change: avoid assumption that 'mayiuse' must invoke 'get_max_cpu_isa',
//         which "locks in" the cpu_isa value from further change.
// Now call get_max_cpu_isa explicitly
//
class isa_set_once_test : public ::testing::Test {
    // The dnnl.h enum value should match the C++ version
#define DNNL_CPU_ISA_T \
    CONCAT2(dnnl_cpu_isa_, PUBLIC_ISA) /* MUST use dnnl_isa_FOO dnnl.h value */
#define CXX_CPU_ISA (dnnl::cpu_isa::PUBLIC_ISA)
    static_assert((int)CXX_CPU_ISA == (int)DNNL_CPU_ISA_T,
            "cpu_isa enum mismatch in dnnl.hpp?");

    isa_set_once_test() : ::testing::Test() {}
    virtual void SetUp() {}
};

TEST(isa_set_once_test, TestISASetOnce) {
    // The internal flags value, arg of mayiuse(cpu_isa_t) from cpu_isa_traits.hpp
    using dnnl::impl::cpu::
            mayiuse; // decl as a static inline in dnnl::impl::cpu::anon

    //printf(" interesting values:\n");
    //printf(" dnnl_cpu_isa_vanilla = %d\n", dnnl_cpu_isa_vanilla);
    //printf(" dnnl_cpu_isa_any     = %d\n", dnnl_cpu_isa_any);
    //printf(" dnnl_cpu_isa_full    = %d\n", dnnl_cpu_isa_full);
    //printf(" DNNL_ISA (cmake)     = %d\n", DNNL_ISA);
    //cout << " Test using dnnl_cpu_isa_t " << STR(DNNL_CPU_ISA_T) " = 0x" << hex
    //     << DNNL_CPU_ISA_T << dec << endl;

    impl::cpu::cpu_isa_t cpuisa_flags = impl::cpu::from_dnnl(DNNL_CPU_ISA_T);
    //printf("%s 0x%lx --> cpu_isa_t cpuisa_flags = 0x%lx\n",
    //        STRINGIFY(DNNL_CPU_ISA_T), (long)DNNL_CPU_ISA_T,
    //        (long)cpuisa_flags);
    ASSERT_TRUE(cpuisa_flags != impl::cpu::isa_unknown);

    static const bool c_api = 1; // Both must compile
    if (c_api) {
        // test the 'C' interface (can do either the C or C++ version, not both)
        // used extern "C" version of set_max_cpu_isa
        auto st = dnnl_set_max_cpu_isa(DNNL_CPU_ISA_T);
        //printf("first time calling dnnl_set_max_cpu_isa(0x%lx) returned %s\n",
        //        (long)DNNL_CPU_ISA_T, dnnl_status2str(st));
        ASSERT_TRUE(st == dnnl_success || st == dnnl_unimplemented);

        // safer: also call get_max_cpu_isa directly (ensure "set once" event)
        int max_cpu_isa = (int)impl::cpu::get_max_cpu_isa();
        //printf(" get_max_cpu_isa() returns %d\n", max_cpu_isa);
        ASSERT_TRUE(max_cpu_isa != impl::cpu::isa_unknown);
        // this now becomes a secondary test,
        // desirable, but not really absolutely necessary
        //printf(" mayiuse(cpuisa_flags=%d)?\n", (int)cpuisa_flags);
        ASSERT_TRUE(mayiuse(cpuisa_flags));

        //printf(" dnnl_set_max_cpu_isa(DNNL_CPU_ISA_T=%d)\n",
        //        (int)DNNL_CPU_ISA_T);
        st = dnnl_set_max_cpu_isa(DNNL_CPU_ISA_T);
        //printf(" second time calling dnnl_set_max_cpu_isa(0x%lx) returned %s\n",
        //        (long)DNNL_CPU_ISA_T, dnnl_status2str(st));
        ASSERT_TRUE(st == dnnl_invalid_arguments || st == dnnl_unimplemented);
    } else {
        // equivalent C++ test
        // The C++ version of dnnl.h enum value, arg of set_max_cpu_isa(cpu_isa) in dnnl.hpp.
        // These values are always equivalent to the dnnl.h values.
        dnnl::cpu_isa cpuisa = static_cast<dnnl::cpu_isa>(CXX_CPU_ISA);
        // The C++ function static casts to dnnl_cpu_isa_t and calls the extern "C" function
        using dnnl::set_max_cpu_isa;
        auto st = set_max_cpu_isa(cpuisa);
        //printf("first time calling dnnl::set_max_cpu_isa(0x%lx) returned %s\n",
        //        (long)cpuisa, dnnl_status2str(static_cast<dnnl_status_t>(st)));
        ASSERT_TRUE(st == status::success || st == status::unimplemented);

        // safer: also call get_max_cpu_isa directly
        ASSERT_TRUE(impl::cpu::get_max_cpu_isa() != impl::cpu::isa_unknown);
        // this now becomes a secondary test,
        // desirable, but not really absolutely necessary
        ASSERT_TRUE(mayiuse(cpuisa_flags));

        st = set_max_cpu_isa(cpuisa);
        //printf("second time calling dnnl::set_max_cpu_isa(0x%lx) returned %s\n",
        //        (long)cpuisa, dnnl_status2str(static_cast<dnnl_status_t>(st)));
        ASSERT_TRUE(
                st == status::invalid_arguments || st == status::unimplemented);
    }

    ASSERT_TRUE(mayiuse(dnnl::impl::cpu::cpu_isa_t::vanilla));
    ASSERT_TRUE(mayiuse(dnnl::impl::cpu::vanilla));
};

} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino=+2s,^=l0,\:0,N-s syntax=cpp.doxygen
