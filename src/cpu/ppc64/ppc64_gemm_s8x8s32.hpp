/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#include <altivec.h>
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {

typedef int int_A1 __attribute__((aligned(1)));
#define memcpy_4(_d, _s) *((int_A1 *)(_d)) = *((int_A1 *)(_s));

#define memcpy_14(_d, _s) \
    { \
        int_A1 *_di, *_si; \
        _di = (int *)(_d); \
        _si = (int *)(_s); \
        *_di++ = *_si++; \
        *_di++ = *_si++; \
        *_di++ = *_si++; \
        *((short *)(_di)) = *((short *)(_si)); \
    }

#define memcpy_16(_d, _s) \
    { \
        int_A1 *_di, *_si; \
        _di = (int *)(_d); \
        _si = (int *)(_s); \
        _di[0] = _si[0]; \
        _di[1] = _si[1]; \
        _di[2] = _si[2]; \
        _di[3] = _si[3]; \
    }

#define memcpy_32(_d, _s) \
    { \
        int_A1 *_di, *_si; \
        _di = (int *)(_d); \
        _si = (int *)(_s); \
        _di[0] = _si[0]; \
        _di[1] = _si[1]; \
        _di[2] = _si[2]; \
        _di[3] = _si[3]; \
        _di[4] = _si[4]; \
        _di[5] = _si[5]; \
        _di[6] = _si[6]; \
        _di[7] = _si[7]; \
    }

#define memcpy_32S(_d, _s) \
    { \
        _d[0] = _s[0]; \
        _d[1] = _s[1]; \
        _d[2] = _s[2]; \
        _d[3] = _s[3]; \
        _d[4] = _s[4]; \
        _d[5] = _s[5]; \
        _d[6] = _s[6]; \
        _d[7] = _s[7]; \
        _d[8] = _s[8]; \
        _d[9] = _s[9]; \
        _d[10] = _s[10]; \
        _d[11] = _s[11]; \
        _d[12] = _s[12]; \
        _d[13] = _s[13]; \
        _d[14] = _s[14]; \
        _d[15] = _s[15]; \
    }

#define memcpy_64(_d, _s) \
    { \
        int_A1 *_di, *_si; \
        _di = (int *)(_d); \
        _si = (int *)(_s); \
        _di[0] = _si[0]; \
        _di[1] = _si[1]; \
        _di[2] = _si[2]; \
        _di[3] = _si[3]; \
        _di[4] = _si[4]; \
        _di[5] = _si[5]; \
        _di[6] = _si[6]; \
        _di[7] = _si[7]; \
        _di[8] = _si[8]; \
        _di[9] = _si[9]; \
        _di[10] = _si[10]; \
        _di[11] = _si[11]; \
        _di[12] = _si[12]; \
        _di[13] = _si[13]; \
        _di[14] = _si[14]; \
        _di[15] = _si[15]; \
    }

#define memcpy_n(_d, _s, _n) \
    { \
        int8_t *_di, *_si; \
        int _i; \
        _di = (int8_t *)(_d); \
        _si = (int8_t *)(_s); \
        for (_i = 0; _i < _n; ++_i) \
            *_di++ = *_si++; \
    }

uint64_t mker;

typedef __vector short vec_i16 __attribute__((aligned(2)));
typedef __vector unsigned char vec_t;
typedef __vector signed char vec_st;

int pack_N16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int i, j;
    int fastpath;
    short *a_offset;
    short *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    short *a_offset5, *a_offset6, *a_offset7, *a_offset8;
    short *a_offset9, *a_offset10, *a_offset11, *a_offset12;
    short *a_offset13, *a_offset14, *a_offset15, *a_offset16;
    short *ap_offset;
    int m_cap = (m + 3) & ~3;
    int k_cap = (k + 1) & ~1;
    a_offset = a;
    ap_offset = ap;
    fastpath = (((k & 1) == 0) && (m & 3) == 0);

    j = (m_cap >> 4);
    int m_skip = (j != (m >> 4));
    while (j) {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset3 = a_offset2 + lda;
        a_offset4 = a_offset3 + lda;
        a_offset5 = a_offset4 + lda;
        a_offset6 = a_offset5 + lda;
        a_offset7 = a_offset6 + lda;
        a_offset8 = a_offset7 + lda;
        a_offset9 = a_offset8 + lda;
        a_offset10 = a_offset9 + lda;
        a_offset11 = a_offset10 + lda;
        a_offset12 = a_offset11 + lda;
        a_offset13 = a_offset12 + lda;
        a_offset14 = a_offset13 + lda;
        a_offset15 = a_offset14 + lda;
        a_offset16 = a_offset15 + lda;
        a_offset += 16 * lda;

        i = (k >> 1);
        while (i) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 1) = *(a_offset1 + 1);
            *(ap_offset + 2) = *(a_offset2 + 0);
            *(ap_offset + 3) = *(a_offset2 + 1);
            *(ap_offset + 4) = *(a_offset3 + 0);
            *(ap_offset + 5) = *(a_offset3 + 1);
            *(ap_offset + 6) = *(a_offset4 + 0);
            *(ap_offset + 7) = *(a_offset4 + 1);

            *(ap_offset + 8) = *(a_offset5 + 0);
            *(ap_offset + 9) = *(a_offset5 + 1);
            *(ap_offset + 10) = *(a_offset6 + 0);
            *(ap_offset + 11) = *(a_offset6 + 1);
            *(ap_offset + 12) = *(a_offset7 + 0);
            *(ap_offset + 13) = *(a_offset7 + 1);
            *(ap_offset + 14) = *(a_offset8 + 0);
            *(ap_offset + 15) = *(a_offset8 + 1);

            *(ap_offset + 16) = *(a_offset9 + 0);
            *(ap_offset + 17) = *(a_offset9 + 1);
            *(ap_offset + 18) = *(a_offset10 + 0);
            *(ap_offset + 19) = *(a_offset10 + 1);
            *(ap_offset + 20) = *(a_offset11 + 0);
            *(ap_offset + 21) = *(a_offset11 + 1);
            *(ap_offset + 22) = *(a_offset12 + 0);
            *(ap_offset + 23) = *(a_offset12 + 1);

            *(ap_offset + 24) = *(a_offset13 + 0);
            *(ap_offset + 25) = *(a_offset13 + 1);
            if (fastpath) {
                *(ap_offset + 26) = *(a_offset14 + 0);
                *(ap_offset + 27) = *(a_offset14 + 1);
                *(ap_offset + 28) = *(a_offset15 + 0);
                *(ap_offset + 29) = *(a_offset15 + 1);
                *(ap_offset + 30) = *(a_offset16 + 0);
                *(ap_offset + 31) = *(a_offset16 + 1);
            } else {
                if ((j != 1) || (!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 26) = *(a_offset14 + 0);
                if ((j != 1) || (!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 27) = *(a_offset14 + 1);
                if ((j != 1) || (!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 28) = *(a_offset15 + 0);
                if ((j != 1) || (!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 29) = *(a_offset15 + 1);
                if ((j != 1) || (!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 30) = *(a_offset16 + 0);
                if ((j != 1) || (!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 31) = *(a_offset16 + 1);
            }

            a_offset1 += 2;
            a_offset2 += 2;
            a_offset3 += 2;
            a_offset4 += 2;
            a_offset5 += 2;
            a_offset6 += 2;
            a_offset7 += 2;
            a_offset8 += 2;

            a_offset9 += 2;
            a_offset10 += 2;
            a_offset11 += 2;
            a_offset12 += 2;
            a_offset13 += 2;
            a_offset14 += 2;
            a_offset15 += 2;
            a_offset16 += 2;
            ap_offset += 32;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 2) = *(a_offset2 + 0);
            *(ap_offset + 4) = *(a_offset3 + 0);
            *(ap_offset + 6) = *(a_offset4 + 0);

            *(ap_offset + 8) = *(a_offset5 + 0);
            *(ap_offset + 10) = *(a_offset6 + 0);
            *(ap_offset + 12) = *(a_offset7 + 0);
            *(ap_offset + 14) = *(a_offset8 + 0);

            *(ap_offset + 16) = *(a_offset9 + 0);
            *(ap_offset + 18) = *(a_offset10 + 0);
            *(ap_offset + 20) = *(a_offset11 + 0);
            *(ap_offset + 22) = *(a_offset12 + 0);

            *(ap_offset + 24) = *(a_offset13 + 0);
            if ((j != 1) || (!m_skip) || (m_cap - m < 3))
                *(ap_offset + 26) = *(a_offset14 + 0);
            if ((j != 1) || (!m_skip) || (m_cap - m < 2))
                *(ap_offset + 28) = *(a_offset15 + 0);
            if ((j != 1) || (!m_skip) || (m_cap - m < 1))
                *(ap_offset + 30) = *(a_offset16 + 0);

            for (int ii = 1; ii < 32; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 32;
        }
        j--;
    } // end of while(j)

    if (m_cap & 8) {
        m_skip = (m & 8) != (m_cap & 8);
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset3 = a_offset2 + lda;
        a_offset4 = a_offset3 + lda;
        a_offset5 = a_offset4 + lda;
        a_offset6 = a_offset5 + lda;
        a_offset7 = a_offset6 + lda;
        a_offset8 = a_offset7 + lda;
        a_offset += 8 * lda;

        i = (k >> 1);
        while (i) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 1) = *(a_offset1 + 1);
            *(ap_offset + 2) = *(a_offset2 + 0);
            *(ap_offset + 3) = *(a_offset2 + 1);
            *(ap_offset + 4) = *(a_offset3 + 0);
            *(ap_offset + 5) = *(a_offset3 + 1);
            *(ap_offset + 6) = *(a_offset4 + 0);
            *(ap_offset + 7) = *(a_offset4 + 1);

            *(ap_offset + 8) = *(a_offset5 + 0);
            *(ap_offset + 9) = *(a_offset5 + 1);
            if (fastpath) {
                *(ap_offset + 10) = *(a_offset6 + 0);
                *(ap_offset + 11) = *(a_offset6 + 1);
                *(ap_offset + 12) = *(a_offset7 + 0);
                *(ap_offset + 13) = *(a_offset7 + 1);
                *(ap_offset + 14) = *(a_offset8 + 0);
                *(ap_offset + 15) = *(a_offset8 + 1);
            } else {
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 10) = *(a_offset6 + 0);
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 11) = *(a_offset6 + 1);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 12) = *(a_offset7 + 0);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 13) = *(a_offset7 + 1);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 14) = *(a_offset8 + 0);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 15) = *(a_offset8 + 1);
            }

            a_offset1 += 2;
            a_offset2 += 2;
            a_offset3 += 2;
            a_offset4 += 2;
            a_offset5 += 2;
            a_offset6 += 2;
            a_offset7 += 2;
            a_offset8 += 2;
            ap_offset += 16;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 2) = *(a_offset2 + 0);
            *(ap_offset + 4) = *(a_offset3 + 0);
            *(ap_offset + 6) = *(a_offset4 + 0);

            *(ap_offset + 8) = *(a_offset5 + 0);
            if ((!m_skip) || (m_cap - m < 3))
                *(ap_offset + 10) = *(a_offset6 + 0);
            if ((!m_skip) || (m_cap - m < 2))
                *(ap_offset + 12) = *(a_offset7 + 0);
            if ((!m_skip) || (m_cap - m < 1))
                *(ap_offset + 14) = *(a_offset8 + 0);

            for (int ii = 1; ii < 16; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 16;
        }
    }

    if (m_cap & 4) {
        m_skip = (m & 4) != (m_cap & 4);
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset3 = a_offset2 + lda;
        a_offset4 = a_offset3 + lda;
        a_offset += 4 * lda;

        i = (k >> 1);
        while (i) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 1) = *(a_offset1 + 1);
            if (fastpath) {
                *(ap_offset + 2) = *(a_offset2 + 0);
                *(ap_offset + 3) = *(a_offset2 + 1);
                *(ap_offset + 4) = *(a_offset3 + 0);
                *(ap_offset + 5) = *(a_offset3 + 1);
                *(ap_offset + 6) = *(a_offset4 + 0);
                *(ap_offset + 7) = *(a_offset4 + 1);
            } else {
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 2) = *(a_offset2 + 0);
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 3) = *(a_offset2 + 1);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 4) = *(a_offset3 + 0);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 5) = *(a_offset3 + 1);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 6) = *(a_offset4 + 0);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 7) = *(a_offset4 + 1);
            }

            a_offset1 += 2;
            a_offset2 += 2;
            a_offset3 += 2;
            a_offset4 += 2;
            ap_offset += 8;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            if ((!m_skip) || (m_cap - m < 3))
                *(ap_offset + 2) = *(a_offset2 + 0);
            if ((!m_skip) || (m_cap - m < 2))
                *(ap_offset + 4) = *(a_offset3 + 0);
            if ((!m_skip) || (m_cap - m < 1))
                *(ap_offset + 6) = *(a_offset4 + 0);

            for (int ii = 1; ii < 8; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 8;
        }
    }
    return 0;
}

int pack_T16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int i, j;
    int fastpath;
    short *a_offset;
    short *a_offset1, *a_offset2;
    short *ap_offset;
    vec_i16 vtemp01, vtemp02, vtemp03, vtemp04;
    int m_cap = (m + 3) & ~3;
    int k_cap = (k + 1) & ~1;
    a_offset = a;
    ap_offset = ap;
    fastpath = (((k & 1) == 0) && (m & 3) == 0);

    j = (m_cap >> 4);
    int m_skip = (j != (m >> 4));
    while (j) {
        a_offset1 = a_offset;
        a_offset2 = a_offset + lda;
        a_offset += 16;

        i = (k >> 1);
        while (i > 1) {
            vtemp01 = *(vec_i16 *)(a_offset1);
            vtemp02 = *(vec_i16 *)(a_offset1 + 8);
            vtemp03 = *(vec_i16 *)(a_offset2);
            vtemp04 = *(vec_i16 *)(a_offset2 + 8);
            *(vec_i16 *)(ap_offset + 0) = vec_mergeh(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 8) = vec_mergel(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 16) = vec_mergeh(vtemp02, vtemp04);
            *(vec_i16 *)(ap_offset + 24) = vec_mergel(vtemp02, vtemp04);
            a_offset1 += 2 * lda;
            a_offset2 += 2 * lda;
            ap_offset += 32;

            i--;
        } // end of while (i)
        if (i == 1) {
            if ((j > 1) || (!m_skip)) {
                vtemp01 = *(vec_i16 *)(a_offset1);
                vtemp02 = *(vec_i16 *)(a_offset1 + 8);
                vtemp03 = *(vec_i16 *)(a_offset2);
                vtemp04 = *(vec_i16 *)(a_offset2 + 8);
                *(vec_i16 *)(ap_offset + 0) = vec_mergeh(vtemp01, vtemp03);
                *(vec_i16 *)(ap_offset + 8) = vec_mergel(vtemp01, vtemp03);
                *(vec_i16 *)(ap_offset + 16) = vec_mergeh(vtemp02, vtemp04);
                *(vec_i16 *)(ap_offset + 24) = vec_mergel(vtemp02, vtemp04);
            } else {
                for (int i16 = 0; i16 < 13; ++i16) {
                    *(ap_offset + 2 * i16 + 0) = *(a_offset1 + i16);
                    *(ap_offset + 2 * i16 + 1) = *(a_offset2 + i16);
                }
                if (m_cap - m < 3) {
                    *(ap_offset + 26) = *(a_offset1 + 13);
                    *(ap_offset + 27) = *(a_offset2 + 13);
                } else {
                    *(ap_offset + 26) = 0;
                    *(ap_offset + 27) = 0;
                }
                if (m_cap - m < 2) {
                    *(ap_offset + 28) = *(a_offset1 + 14);
                    *(ap_offset + 29) = *(a_offset2 + 14);
                } else {
                    *(ap_offset + 28) = 0;
                    *(ap_offset + 29) = 0;
                }
                if (m_cap - m < 1) {
                    *(ap_offset + 30) = *(a_offset1 + 15);
                    *(ap_offset + 31) = *(a_offset2 + 15);
                } else {
                    *(ap_offset + 30) = 0;
                    *(ap_offset + 31) = 0;
                }
            }
            a_offset1 += 2 * lda;
            a_offset2 += 2 * lda;
            ap_offset += 32;
        }

        if (k < k_cap) {
            vtemp01 = *(vec_i16 *)(a_offset1);
            vtemp02 = *(vec_i16 *)(a_offset1 + 8);
            vtemp03 = *(vec_i16 *)(a_offset1); // garbage, never read
            vtemp04 = *(vec_i16 *)(a_offset1 + 8); // garbage, never read
            *(vec_i16 *)(ap_offset + 0) = vec_mergeh(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 8) = vec_mergel(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 16) = vec_mergeh(vtemp02, vtemp04);
            *(vec_i16 *)(ap_offset + 24) = vec_mergel(vtemp02, vtemp04);

            for (int ii = 1; ii < 32; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 32;
        }

        j--;
    } // end of while (j)

    if (m_cap & 8) {
        m_skip = (m & 8) != (m_cap & 8);
        a_offset1 = a_offset;
        a_offset2 = a_offset + lda;
        a_offset += 8;

        i = (k >> 1);
        while (i) {
            vtemp01 = *(vec_i16 *)(a_offset1);
            vtemp03 = *(vec_i16 *)(a_offset2);
            *(vec_i16 *)(ap_offset + 0) = vec_mergeh(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 8) = vec_mergel(vtemp01, vtemp03);

            a_offset1 += 2 * lda;
            a_offset2 += 2 * lda;
            ap_offset += 16;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            vtemp01 = *(vec_i16 *)(a_offset1);
            vtemp03 = *(vec_i16 *)(a_offset1); // garbage, never read
            *(vec_i16 *)(ap_offset + 0) = vec_mergeh(vtemp01, vtemp03);
            *(vec_i16 *)(ap_offset + 8) = vec_mergel(vtemp01, vtemp03);

            for (int ii = 1; ii < 16; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 16;
        }
    }

    if (m_cap & 4) {
        m_skip = (m & 4) != (m_cap & 4);
        a_offset1 = a_offset;
        a_offset2 = a_offset + lda;
        a_offset += 4;

        i = (k >> 1);
        while (i) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            *(ap_offset + 1) = *(a_offset2 + 0);
            if (fastpath) {
                *(ap_offset + 2) = *(a_offset1 + 1);
                *(ap_offset + 3) = *(a_offset2 + 1);
                *(ap_offset + 4) = *(a_offset1 + 2);
                *(ap_offset + 5) = *(a_offset2 + 2);
                *(ap_offset + 6) = *(a_offset1 + 3);
                *(ap_offset + 7) = *(a_offset2 + 3);
            } else {
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 2) = *(a_offset1 + 1);
                if ((!m_skip) || (m_cap - m < 3))
                    *(ap_offset + 3) = *(a_offset2 + 1);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 4) = *(a_offset1 + 2);
                if ((!m_skip) || (m_cap - m < 2))
                    *(ap_offset + 5) = *(a_offset2 + 2);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 6) = *(a_offset1 + 3);
                if ((!m_skip) || (m_cap - m < 1))
                    *(ap_offset + 7) = *(a_offset2 + 3);
            }

            a_offset1 += 2 * lda;
            a_offset2 += 2 * lda;
            ap_offset += 8;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            *(ap_offset + 0) = *(a_offset1 + 0);
            if ((!m_skip) || (m_cap - m < 3))
                *(ap_offset + 2) = *(a_offset1 + 1);
            if ((!m_skip) || (m_cap - m < 2))
                *(ap_offset + 4) = *(a_offset1 + 2);
            if ((!m_skip) || (m_cap - m < 1))
                *(ap_offset + 6) = *(a_offset1 + 3);

            for (int ii = 1; ii < 8; ii += 2)
                *(ap_offset + ii) = 0;

            ap_offset += 8;
        }
    }
    return 0;
}

int pack_T8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int i, j;
    int fastpath;
    short *b_offset, *b_offset1, *b_offset2, *b_offset3, *b_offset4, *b_offset5,
            *b_offset6, *b_offset7, *b_offset8;
    short *bp_offset, *bp_offset1, *bp_offset2;
    vec_i16 vtemp01, vtemp02, vtemp03, vtemp04;
    vec_i16 vtemp05, vtemp06, vtemp07, vtemp08;
    int n_cap = (n + 3) & (~3);
    int k_cap = (k + 1) & (~1);
    fastpath = (((k & 1) == 0) && (n & 3) == 0);

    b_offset = b;
    bp_offset = bp;
    bp_offset2 = bp + k_cap * (n_cap & ~7);

    j = (k_cap >> 3);
    int k_skip = ((k >> 3) != j);
    while (j) {
        b_offset1 = b_offset;
        b_offset2 = b_offset1 + ldb;
        b_offset3 = b_offset2 + ldb;
        b_offset4 = b_offset3 + ldb;
        b_offset5 = b_offset4 + ldb;
        b_offset6 = b_offset5 + ldb;
        b_offset7 = b_offset6 + ldb;
        b_offset8 = b_offset7 + ldb;
        b_offset += 8 * ldb;

        bp_offset1 = bp_offset;
        bp_offset += 64;

        i = (n_cap >> 3);
        // we need to be careful about not going past the end of the B array if n is less than n_cap.
        // fortunately, we can only go out-of-bounds by accessing elements of b_offset8, so the others
        // can just load garbage in the not_used elements of the vtemp vectors.
        while (i) {
            vtemp01 = *(vec_i16 *)(b_offset1);
            vtemp02 = *(vec_i16 *)(b_offset2);
            vtemp03 = *(vec_i16 *)(b_offset3);
            vtemp04 = *(vec_i16 *)(b_offset4);
            vtemp05 = *(vec_i16 *)(b_offset5);
            vtemp06 = *(vec_i16 *)(b_offset6);
            vtemp07 = *(vec_i16 *)(b_offset7);
            if ((j == 1) && k_skip) {
                short temp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                vtemp08 = *(vec_i16 *)(temp);
            } else {
                if ((i == 1) && (!(n_cap & 4))) {
                    memcpy_16(&vtemp08, b_offset8);
                } else {
                    vtemp08 = *(vec_i16 *)(b_offset8);
                }
            }
            b_offset1 += 8;
            b_offset2 += 8;
            b_offset3 += 8;
            b_offset4 += 8;
            b_offset5 += 8;
            b_offset6 += 8;
            b_offset7 += 8;
            b_offset8 += 8;

            *(vec_i16 *)(bp_offset1 + 0) = vec_mergeh(vtemp01, vtemp02);
            *(vec_i16 *)(bp_offset1 + 8) = vec_mergel(vtemp01, vtemp02);
            *(vec_i16 *)(bp_offset1 + 16) = vec_mergeh(vtemp03, vtemp04);
            *(vec_i16 *)(bp_offset1 + 24) = vec_mergel(vtemp03, vtemp04);
            *(vec_i16 *)(bp_offset1 + 32) = vec_mergeh(vtemp05, vtemp06);
            *(vec_i16 *)(bp_offset1 + 40) = vec_mergel(vtemp05, vtemp06);
            *(vec_i16 *)(bp_offset1 + 48) = vec_mergeh(vtemp07, vtemp08);
            *(vec_i16 *)(bp_offset1 + 56) = vec_mergel(vtemp07, vtemp08);

            bp_offset1 += k_cap * 8;
            i--;
        } // end of while (i)

        if (n_cap & 4) {
            *(bp_offset2 + 0) = *(b_offset1 + 0);
            *(bp_offset2 + 1) = *(b_offset2 + 0);
            *(bp_offset2 + 2) = *(b_offset1 + 1);
            *(bp_offset2 + 3) = *(b_offset2 + 1);
            *(bp_offset2 + 4) = *(b_offset1 + 2);
            *(bp_offset2 + 5) = *(b_offset2 + 2);
            *(bp_offset2 + 6) = *(b_offset1 + 3);
            *(bp_offset2 + 7) = *(b_offset2 + 3);

            *(bp_offset2 + 8) = *(b_offset3 + 0);
            *(bp_offset2 + 9) = *(b_offset4 + 0);
            *(bp_offset2 + 10) = *(b_offset3 + 1);
            *(bp_offset2 + 11) = *(b_offset4 + 1);
            *(bp_offset2 + 12) = *(b_offset3 + 2);
            *(bp_offset2 + 13) = *(b_offset4 + 2);
            *(bp_offset2 + 14) = *(b_offset3 + 3);
            *(bp_offset2 + 15) = *(b_offset4 + 3);

            // same story here, if n is less than n_cap, we have to be careful with accessing b_offset8
            *(bp_offset2 + 16) = *(b_offset5 + 0);
            if (fastpath) {
                *(bp_offset2 + 17) = *(b_offset6 + 0);
                *(bp_offset2 + 18) = *(b_offset5 + 1);
                *(bp_offset2 + 19) = *(b_offset6 + 1);
                *(bp_offset2 + 20) = *(b_offset5 + 2);
                *(bp_offset2 + 21) = *(b_offset6 + 2);
                *(bp_offset2 + 22) = *(b_offset5 + 3);
                *(bp_offset2 + 23) = *(b_offset6 + 3);
            } else {
                *(bp_offset2 + 17) = *(b_offset6 + 0);
                *(bp_offset2 + 18) = *(b_offset5 + 1);
                *(bp_offset2 + 19) = *(b_offset6 + 1);
                *(bp_offset2 + 20) = *(b_offset5 + 2);
                *(bp_offset2 + 21) = *(b_offset6 + 2);
                *(bp_offset2 + 22) = *(b_offset5 + 3);
                if (!k_skip || j > 1)
                    *(bp_offset2 + 23)
                            = *(b_offset6 + ((n_cap - n < 3) ? 3 : 0));
            }
            *(bp_offset2 + 24) = *(b_offset7 + 0);
            if (fastpath) {
                *(bp_offset2 + 25) = *(b_offset8 + 0);
                *(bp_offset2 + 26) = *(b_offset7 + 1);
                *(bp_offset2 + 27) = *(b_offset8 + 1);
                *(bp_offset2 + 28) = *(b_offset7 + 2);
                *(bp_offset2 + 29) = *(b_offset8 + 2);
                *(bp_offset2 + 30) = *(b_offset7 + 3);
                *(bp_offset2 + 31) = *(b_offset8 + 3);
            } else {
                if (!k_skip || j > 1) *(bp_offset2 + 25) = *(b_offset8 + 0);
                *(bp_offset2 + 26) = *(b_offset7 + 1);
                if (!k_skip || j > 1)
                    *(bp_offset2 + 27)
                            = *(b_offset8 + ((n_cap - n < 3) ? 1 : 0));

                if (!k_skip || j > 1)
                    *(bp_offset2 + 28)
                            = *(b_offset7 + ((n_cap - n < 3) ? 2 : 0));

                if (!k_skip || j > 1)
                    *(bp_offset2 + 29)
                            = *(b_offset8 + ((n_cap - n < 2) ? 2 : 0));

                if (!k_skip || j > 1)
                    *(bp_offset2 + 30)
                            = *(b_offset7 + ((n_cap - n < 2) ? 3 : 0));

                if (!k_skip || j > 1)
                    *(bp_offset2 + 31)
                            = *(b_offset8 + ((n_cap - n < 1) ? 3 : 0));
            }

            b_offset1 += 4;
            b_offset2 += 4;
            b_offset3 += 4;
            b_offset4 += 4;
            b_offset5 += 4;
            b_offset6 += 4;
            b_offset7 += 4;
            b_offset8 += 4;
            bp_offset2 += 32;
        }

        j--;
    } // end of while (j)

    if (k_cap & 4) {
        k_skip = (k & 4) != (k_cap & 4);

        b_offset1 = b_offset;
        b_offset2 = b_offset1 + ldb;
        b_offset3 = b_offset2 + ldb;
        b_offset4 = b_offset3 + ldb;
        b_offset += 4 * ldb;

        bp_offset1 = bp_offset;
        bp_offset += 32;

        i = (n_cap >> 3);
        // we need to be careful about not going past the end of the B array if n is less than n_cap.
        // fortunately, we can only go out-of-bounds by accessing elements of b_offset4, so the others
        // can just load garbage in the not_used elements of the vtemp vectors.
        while (i) {
            if (fastpath) {
                vtemp01 = *(vec_i16 *)(b_offset1);
                vtemp02 = *(vec_i16 *)(b_offset2);
                vtemp03 = *(vec_i16 *)(b_offset3);
                vtemp04 = *(vec_i16 *)(b_offset4);
                *(vec_i16 *)(bp_offset1 + 0) = vec_mergeh(vtemp01, vtemp02);
                *(vec_i16 *)(bp_offset1 + 8) = vec_mergel(vtemp01, vtemp02);
                *(vec_i16 *)(bp_offset1 + 16) = vec_mergeh(vtemp03, vtemp04);
                *(vec_i16 *)(bp_offset1 + 24) = vec_mergel(vtemp03, vtemp04);
            } else {
                *(bp_offset1 + 0) = *(b_offset1 + 0);
                *(bp_offset1 + 1) = *(b_offset2 + 0);
                *(bp_offset1 + 2) = *(b_offset1 + 1);
                *(bp_offset1 + 3) = *(b_offset2 + 1);
                *(bp_offset1 + 4) = *(b_offset1 + 2);
                *(bp_offset1 + 5) = *(b_offset2 + 2);
                *(bp_offset1 + 6) = *(b_offset1 + 3);
                *(bp_offset1 + 7) = *(b_offset2 + 3);

                *(bp_offset1 + 8) = *(b_offset1 + 4);
                *(bp_offset1 + 9) = *(b_offset2 + 4);
                *(bp_offset1 + 10) = *(b_offset1 + 5);
                *(bp_offset1 + 11) = *(b_offset2 + 5);
                *(bp_offset1 + 12) = *(b_offset1 + 6);
                *(bp_offset1 + 13) = *(b_offset2 + 6);
                *(bp_offset1 + 14) = *(b_offset1 + 7);
                *(bp_offset1 + 15) = *(b_offset2 + 7);

                *(bp_offset1 + 16) = *(b_offset3 + 0);
                if (!k_skip) *(bp_offset1 + 17) = *(b_offset4 + 0);
                *(bp_offset1 + 18) = *(b_offset3 + 1);
                if (!k_skip) *(bp_offset1 + 19) = *(b_offset4 + 1);
                *(bp_offset1 + 20) = *(b_offset3 + 2);
                if (!k_skip) *(bp_offset1 + 21) = *(b_offset4 + 2);
                *(bp_offset1 + 22) = *(b_offset3 + 3);
                if (!k_skip) *(bp_offset1 + 23) = *(b_offset4 + 3);

                *(bp_offset1 + 24) = *(b_offset3 + 4);
                if (!k_skip) *(bp_offset1 + 25) = *(b_offset4 + 4);
                *(bp_offset1 + 26) = *(b_offset3 + 5);
                if (!k_skip)
                    *(bp_offset1 + 27) = *(b_offset4
                            + (((i > 1) || (n_cap & 4) || (n_cap - n < 3))
                                            ? 5
                                            : 0));
                *(bp_offset1 + 28) = *(b_offset3 + 6);
                if (!k_skip)
                    *(bp_offset1 + 29) = *(b_offset4
                            + (((i > 1) || (n_cap & 4) || (n_cap - n < 2))
                                            ? 6
                                            : 0));
                *(bp_offset1 + 30) = *(b_offset3 + 7);
                if (!k_skip)
                    *(bp_offset1 + 31) = *(b_offset4
                            + (((i > 1) || (n_cap & 4) || (n_cap - n < 1))
                                            ? 7
                                            : 0));
            }

            b_offset1 += 8;
            b_offset2 += 8;
            b_offset3 += 8;
            b_offset4 += 8;
            bp_offset1 += 8 * k_cap;
            i--;
        } // end of while (i)

        if (n_cap & 4) {
            *(bp_offset2 + 0) = *(b_offset1 + 0);
            *(bp_offset2 + 1) = *(b_offset2 + 0);
            *(bp_offset2 + 2) = *(b_offset1 + 1);
            *(bp_offset2 + 3) = *(b_offset2 + 1);
            *(bp_offset2 + 4) = *(b_offset1 + 2);
            *(bp_offset2 + 5) = *(b_offset2 + 2);
            *(bp_offset2 + 6) = *(b_offset1 + 3);
            *(bp_offset2 + 7) = *(b_offset2 + 3);

            if (fastpath) {
                *(bp_offset2 + 8) = *(b_offset3 + 0);
                *(bp_offset2 + 9) = *(b_offset4 + 0);
                *(bp_offset2 + 10) = *(b_offset3 + 1);
                *(bp_offset2 + 11) = *(b_offset4 + 1);
                *(bp_offset2 + 12) = *(b_offset3 + 2);
                *(bp_offset2 + 13) = *(b_offset4 + 2);
                *(bp_offset2 + 14) = *(b_offset3 + 3);
                *(bp_offset2 + 15) = *(b_offset4 + 3);
            } else {
                *(bp_offset2 + 8) = *(b_offset3 + 0);
                if (!k_skip) *(bp_offset2 + 9) = *(b_offset4 + 0);
                *(bp_offset2 + 10) = *(b_offset3 + 1);
                if (!k_skip)
                    *(bp_offset2 + 11)
                            = *(b_offset4 + ((n_cap - n < 3) ? 1 : 0));
                *(bp_offset2 + 12) = *(b_offset3 + 2);
                if (!k_skip)
                    *(bp_offset2 + 13)
                            = *(b_offset4 + ((n_cap - n < 2) ? 2 : 0));
                *(bp_offset2 + 14) = *(b_offset3 + 3);
                if (!k_skip)
                    *(bp_offset2 + 15)
                            = *(b_offset4 + ((n_cap - n < 1) ? 3 : 0));
            }
            b_offset1 += 4;
            b_offset2 += 4;
            b_offset3 += 4;
            b_offset4 += 4;
            bp_offset2 += 16;
        }
    }

    if (k_cap & 2) {
        k_skip = (k & 2) != (k_cap & 2);

        b_offset1 = b_offset;
        b_offset2 = b_offset1 + ldb;
        b_offset += 2 * ldb;

        bp_offset1 = bp_offset;
        bp_offset += 16;

        i = (n_cap >> 3);
        if (fastpath) {
            while (i) {
                vtemp01 = *(vec_i16 *)(b_offset1);
                vtemp02 = *(vec_i16 *)(b_offset2);
                *(vec_i16 *)(bp_offset1 + 0) = vec_mergeh(vtemp01, vtemp02);
                *(vec_i16 *)(bp_offset1 + 8) = vec_mergel(vtemp01, vtemp02);

                b_offset1 += 8;
                b_offset2 += 8;
                bp_offset1 += 8 * k_cap;
                i--;
            } // end of while (i)

            if (n_cap & 4) {
                *(bp_offset2 + 0) = *(b_offset1 + 0);
                *(bp_offset2 + 1) = *(b_offset2 + 0);
                *(bp_offset2 + 2) = *(b_offset1 + 1);
                *(bp_offset2 + 3) = *(b_offset2 + 1);
                *(bp_offset2 + 4) = *(b_offset1 + 2);
                *(bp_offset2 + 5) = *(b_offset2 + 2);
                *(bp_offset2 + 6) = *(b_offset1 + 3);
                *(bp_offset2 + 7) = *(b_offset2 + 3);
                b_offset1 += 4;
                b_offset2 += 4;
                bp_offset2 += 8;
            }
        } else {
            // we need to be careful about not going past the end of the B array if n is less than n_cap.
            // fortunately, we can only go out-of-bounds by accessing elements of b_offset2, so the others
            // can just load garbage in the not_used elements of the vtemp vectors.
            while (i) {
                *(bp_offset1 + 0) = *(b_offset1 + 0);
                if (!k_skip) *(bp_offset1 + 1) = *(b_offset2 + 0);
                *(bp_offset1 + 2) = *(b_offset1 + 1);
                if (!k_skip) *(bp_offset1 + 3) = *(b_offset2 + 1);
                *(bp_offset1 + 4) = *(b_offset1 + 2);
                if (!k_skip) *(bp_offset1 + 5) = *(b_offset2 + 2);
                *(bp_offset1 + 6) = *(b_offset1 + 3);
                if (!k_skip) *(bp_offset1 + 7) = *(b_offset2 + 3);

                *(bp_offset1 + 8) = *(b_offset1 + 4);
                if (!k_skip) *(bp_offset1 + 9) = *(b_offset2 + 4);
                *(bp_offset1 + 10) = *(b_offset1 + 5);
                if (!k_skip)
                    *(bp_offset1 + 11) = *(b_offset2
                            + ((i > 1) || (n_cap & 4) || ((n_cap - n < 3))
                                            ? 5
                                            : 0));
                *(bp_offset1 + 12) = *(b_offset1 + 6);
                if (!k_skip)
                    *(bp_offset1 + 13) = *(b_offset2
                            + ((i > 1) || (n_cap & 4) || ((n_cap - n < 2))
                                            ? 6
                                            : 0));
                *(bp_offset1 + 14) = *(b_offset1 + 7);
                if (!k_skip)
                    *(bp_offset1 + 15) = *(b_offset2
                            + ((i > 1) || (n_cap & 4) || ((n_cap - n < 1))
                                            ? 7
                                            : 0));

                b_offset1 += 8;
                b_offset2 += 8;
                bp_offset1 += 8 * k_cap;
                i--;
            } // end of while (i)

            if (n_cap & 4) {
                *(bp_offset2 + 0) = *(b_offset1 + 0);
                if (!k_skip) *(bp_offset2 + 1) = *(b_offset2 + 0);

                *(bp_offset2 + 2) = *(b_offset1 + ((n_cap - n < 3) ? 1 : 0));
                if (!k_skip)
                    *(bp_offset2 + 3)
                            = *(b_offset2 + ((n_cap - n < 3) ? 1 : 0));

                *(bp_offset2 + 4) = *(b_offset1 + ((n_cap - n < 2) ? 2 : 0));
                if (!k_skip)
                    *(bp_offset2 + 5)
                            = *(b_offset2 + ((n_cap - n < 2) ? 2 : 0));

                *(bp_offset2 + 6) = *(b_offset1 + ((n_cap - n < 1) ? 3 : 0));
                if (!k_skip)
                    *(bp_offset2 + 7)
                            = *(b_offset2 + ((n_cap - n < 1) ? 3 : 0));

                b_offset1 += 4;
                b_offset2 += 4;
                bp_offset2 += 8;
            }
        }
    }

    return 0;
}

int pack_N8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int i, j, k_skip;
    int fastpath;
    short *b_offset;
    short *b_offset1, *b_offset2, *b_offset3, *b_offset4;
    short *b_offset5, *b_offset6, *b_offset7, *b_offset8;
    short *bp_offset;
    int n_cap = (n + 3) & ~3;
    int k_cap = (k + 1) & ~1;
    vec_i16 vtemp01, vtemp02, vtemp03, vtemp04;
    vec_i16 vtemp05, vtemp06, vtemp07, vtemp08;
    vec_i16 vtemp09, vtemp10, vtemp11, vtemp12;
    vec_t mask = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t mask1
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    fastpath = (((k & 1) == 0) && (n & 3) == 0);

    b_offset = b;
    bp_offset = bp;

    j = (n_cap >> 3);
    while (j) {
        b_offset1 = b_offset;
        b_offset2 = b_offset1 + ldb;
        b_offset3 = b_offset2 + ldb;
        b_offset4 = b_offset3 + ldb;
        b_offset5 = b_offset4 + ldb;
        b_offset6 = b_offset5 + ldb;
        b_offset7 = b_offset6 + ldb;
        b_offset8 = b_offset7 + ldb;
        b_offset += 8 * ldb;

        i = (k_cap >> 3);
        k_skip = ((k >> 3) != i);
        int copy6 = (j != 1) || (n_cap & 4) || (n_cap - n < 3);
        int copy7 = (j != 1) || (n_cap & 4) || (n_cap - n < 2);
        int copy8 = (j != 1) || (n_cap & 4) || (n_cap - n < 1);
        while (i) {
            if (i == 1 && k_skip) {
                short temp1[8], temp2[8], temp3[8], temp4[8], temp5[8],
                        temp6[8], temp7[8], temp8[8];
                memcpy_14(temp1, b_offset1);
                memcpy_14(temp2, b_offset2);
                memcpy_14(temp3, b_offset3);
                memcpy_14(temp4, b_offset4);
                memcpy_14(temp5, b_offset5);
                short *s6, *s7, *s8;
                s6 = copy6 ? b_offset6 : b_offset5;
                s7 = copy7 ? b_offset7 : b_offset5;
                s8 = copy8 ? b_offset8 : b_offset5;
                memcpy_14(temp6, s6);
                memcpy_14(temp7, s7);
                memcpy_14(temp8, s8);
                vtemp01 = *(vec_i16 *)(temp1);
                vtemp02 = *(vec_i16 *)(temp2);
                vtemp03 = *(vec_i16 *)(temp3);
                vtemp04 = *(vec_i16 *)(temp4);
                vtemp05 = *(vec_i16 *)(temp5);
                vtemp06 = *(vec_i16 *)(temp6);
                vtemp07 = *(vec_i16 *)(temp7);
                vtemp08 = *(vec_i16 *)(temp8);
            } else {
                vtemp01 = *(vec_i16 *)(b_offset1);
                vtemp02 = *(vec_i16 *)(b_offset2);
                vtemp03 = *(vec_i16 *)(b_offset3);
                vtemp04 = *(vec_i16 *)(b_offset4);
                vtemp05 = *(vec_i16 *)(b_offset5);
                if (fastpath) {
                    vtemp06 = *(vec_i16 *)(b_offset6);
                    vtemp07 = *(vec_i16 *)(b_offset7);
                    vtemp08 = *(vec_i16 *)(b_offset8);
                } else {
                    vtemp06 = *(vec_i16 *)(copy6 ? b_offset6 : b_offset5);
                    vtemp07 = *(vec_i16 *)(copy7 ? b_offset7 : b_offset5);
                    vtemp08 = *(vec_i16 *)(copy8 ? b_offset8 : b_offset5);
                }
            }

            vtemp09 = vec_perm(vtemp01, vtemp02, mask);
            vtemp10 = vec_perm(vtemp03, vtemp04, mask);
            vtemp11 = vec_perm(vtemp05, vtemp06, mask);
            vtemp12 = vec_perm(vtemp07, vtemp08, mask);

            *(vec_i16 *)(bp_offset + 0) = vec_xxpermdi(vtemp09, vtemp10, 0);
            *(vec_i16 *)(bp_offset + 8) = vec_xxpermdi(vtemp11, vtemp12, 0);
            *(vec_i16 *)(bp_offset + 16) = vec_xxpermdi(vtemp09, vtemp10, 3);
            *(vec_i16 *)(bp_offset + 24) = vec_xxpermdi(vtemp11, vtemp12, 3);

            vtemp09 = vec_perm(vtemp01, vtemp02, mask1);
            vtemp10 = vec_perm(vtemp03, vtemp04, mask1);
            vtemp11 = vec_perm(vtemp05, vtemp06, mask1);
            vtemp12 = vec_perm(vtemp07, vtemp08, mask1);

            *(vec_i16 *)(bp_offset + 32) = vec_xxpermdi(vtemp09, vtemp10, 0);
            *(vec_i16 *)(bp_offset + 40) = vec_xxpermdi(vtemp11, vtemp12, 0);
            *(vec_i16 *)(bp_offset + 48) = vec_xxpermdi(vtemp09, vtemp10, 3);
            *(vec_i16 *)(bp_offset + 56) = vec_xxpermdi(vtemp11, vtemp12, 3);

            if (i == 1 && k_skip) {
                b_offset1 += 7;
                b_offset2 += 7;
                b_offset3 += 7;
                b_offset4 += 7;
                b_offset5 += 7;
                b_offset6 += 7;
                b_offset7 += 7;
                b_offset8 += 7;
            } else {
                b_offset1 += 8;
                b_offset2 += 8;
                b_offset3 += 8;
                b_offset4 += 8;
                b_offset5 += 8;
                b_offset6 += 8;
                b_offset7 += 8;
                b_offset8 += 8;
            }
            bp_offset += 64;
            i--;
        } // end of while (i)

        if (k_skip) goto endloop;

        i = (k & 7);
        while (i > 1) {
            *(bp_offset + 0) = *(b_offset1 + 0);
            *(bp_offset + 1) = *(b_offset1 + 1);
            *(bp_offset + 2) = *(b_offset2 + 0);
            *(bp_offset + 3) = *(b_offset2 + 1);
            *(bp_offset + 4) = *(b_offset3 + 0);
            *(bp_offset + 5) = *(b_offset3 + 1);
            *(bp_offset + 6) = *(b_offset4 + 0);
            *(bp_offset + 7) = *(b_offset4 + 1);

            *(bp_offset + 8) = *(b_offset5 + 0);
            *(bp_offset + 9) = *(b_offset5 + 1);
            if (fastpath) {
                *(bp_offset + 10) = *(b_offset6 + 0);
                *(bp_offset + 11) = *(b_offset6 + 1);
                *(bp_offset + 12) = *(b_offset7 + 0);
                *(bp_offset + 13) = *(b_offset7 + 1);
                *(bp_offset + 14) = *(b_offset8 + 0);
                *(bp_offset + 15) = *(b_offset8 + 1);
            } else {
                *(bp_offset + 10) = *((copy6 ? b_offset6 : b_offset5) + 0);
                *(bp_offset + 11) = *((copy6 ? b_offset6 : b_offset5) + 1);
                *(bp_offset + 12) = *((copy7 ? b_offset7 : b_offset5) + 0);
                *(bp_offset + 13) = *((copy7 ? b_offset7 : b_offset5) + 1);
                *(bp_offset + 14) = *((copy8 ? b_offset8 : b_offset5) + 0);
                *(bp_offset + 15) = *((copy8 ? b_offset8 : b_offset5) + 1);
            }

            b_offset1 += 2;
            b_offset2 += 2;
            b_offset3 += 2;
            b_offset4 += 2;
            b_offset5 += 2;
            b_offset6 += 2;
            b_offset7 += 2;
            b_offset8 += 2;
            bp_offset += 16;
            i -= 2;
        } // end of while (i)
        if (k < k_cap) {
            *(bp_offset + 0) = *(b_offset1 + 0);
            *(bp_offset + 2) = *(b_offset2 + 0);
            *(bp_offset + 4) = *(b_offset3 + 0);
            *(bp_offset + 6) = *(b_offset4 + 0);
            *(bp_offset + 8) = *(b_offset5 + 0);
            *(bp_offset + 10) = *((copy6 ? b_offset6 : b_offset5) + 0);
            *(bp_offset + 12) = *((copy7 ? b_offset7 : b_offset5) + 0);
            *(bp_offset + 14) = *((copy8 ? b_offset8 : b_offset5) + 0);

            b_offset1++;
            b_offset2++;
            b_offset3++;
            b_offset4++;
            b_offset5++;
            b_offset6++;
            b_offset7++;
            b_offset8++;
            bp_offset += 16;
        }

    endloop:
        j--;
    } // end of while (j)

    if (n_cap & 4) {
        b_offset1 = b_offset;
        b_offset2 = b_offset1 + ldb;
        b_offset3 = b_offset2 + ldb;
        b_offset4 = b_offset3 + ldb;
        b_offset += 4 * ldb;

        i = (k_cap >> 2);
        k_skip = ((k >> 2) != i);
        int copy2 = (n_cap - n < 3);
        int copy3 = (n_cap - n < 2);
        int copy4 = (n_cap - n < 1);
        while (i) {
            *(bp_offset + 0) = *(b_offset1 + 0);
            *(bp_offset + 1) = *(b_offset1 + 1);
            if (fastpath) {
                *(bp_offset + 2) = *(b_offset2 + 0);
                *(bp_offset + 3) = *(b_offset2 + 1);
                *(bp_offset + 4) = *(b_offset3 + 0);
                *(bp_offset + 5) = *(b_offset3 + 1);
                *(bp_offset + 6) = *(b_offset4 + 0);
                *(bp_offset + 7) = *(b_offset4 + 1);
                *(bp_offset + 8) = *(b_offset1 + 2);
                *(bp_offset + 9) = *(b_offset1 + 3);
                *(bp_offset + 10) = *(b_offset2 + 2);
                *(bp_offset + 11) = *(b_offset2 + 3);
                *(bp_offset + 12) = *(b_offset3 + 2);
                *(bp_offset + 13) = *(b_offset3 + 3);
                *(bp_offset + 14) = *(b_offset4 + 2);
                *(bp_offset + 15) = *(b_offset4 + 3);
            } else {
                *(bp_offset + 2) = *((copy2 ? b_offset2 : b_offset1) + 0);
                *(bp_offset + 3) = *((copy2 ? b_offset2 : b_offset1) + 1);
                *(bp_offset + 4) = *((copy3 ? b_offset3 : b_offset1) + 0);
                *(bp_offset + 5) = *((copy3 ? b_offset3 : b_offset1) + 1);
                *(bp_offset + 6) = *((copy4 ? b_offset4 : b_offset1) + 0);
                *(bp_offset + 7) = *((copy4 ? b_offset4 : b_offset1) + 1);

                *(bp_offset + 8) = *(b_offset1 + 2);
                if ((i > 1) || (!k_skip)) *(bp_offset + 9) = *(b_offset1 + 3);
                *(bp_offset + 10) = *((copy2 ? b_offset2 : b_offset1) + 2);
                if ((i > 1) || (!k_skip))
                    *(bp_offset + 11) = *((copy2 ? b_offset2 : b_offset1) + 3);
                *(bp_offset + 12) = *((copy3 ? b_offset3 : b_offset1) + 2);
                if ((i > 1) || (!k_skip))
                    *(bp_offset + 13) = *((copy3 ? b_offset3 : b_offset1) + 3);
                *(bp_offset + 14) = *((copy4 ? b_offset4 : b_offset1) + 2);
                if ((i > 1) || (!k_skip))
                    *(bp_offset + 15) = *((copy4 ? b_offset4 : b_offset1) + 3);
            }

            b_offset1 += 4;
            b_offset2 += 4;
            b_offset3 += 4;
            b_offset4 += 4;
            bp_offset += 16;
            i--;
        } // end of while (i)

        if (k_skip) goto exit;

        if (k & 2) {
            *(bp_offset + 0) = *(b_offset1 + 0);
            *(bp_offset + 1) = *(b_offset1 + 1);
            if (fastpath) {
                *(bp_offset + 2) = *(b_offset2 + 0);
                *(bp_offset + 3) = *(b_offset2 + 1);
                *(bp_offset + 4) = *(b_offset3 + 0);
                *(bp_offset + 5) = *(b_offset3 + 1);
                *(bp_offset + 6) = *(b_offset4 + 0);
                *(bp_offset + 7) = *(b_offset4 + 1);
            } else {
                *(bp_offset + 2) = *((copy2 ? b_offset2 : b_offset1) + 0);
                *(bp_offset + 3) = *((copy2 ? b_offset2 : b_offset1) + 1);
                *(bp_offset + 4) = *((copy3 ? b_offset3 : b_offset1) + 0);
                *(bp_offset + 5) = *((copy3 ? b_offset3 : b_offset1) + 1);
                *(bp_offset + 6) = *((copy4 ? b_offset4 : b_offset1) + 0);
                *(bp_offset + 7) = *((copy4 ? b_offset4 : b_offset1) + 1);
            }

            b_offset1 += 2;
            b_offset2 += 2;
            b_offset3 += 2;
            b_offset4 += 2;
            bp_offset += 8;
        }

        if (k & 1) {
            *(bp_offset + 0) = *(b_offset1 + 0);
            *(bp_offset + 2) = *((copy2 ? b_offset2 : b_offset1) + 0);
            *(bp_offset + 4) = *((copy3 ? b_offset3 : b_offset1) + 0);
            *(bp_offset + 6) = *((copy4 ? b_offset4 : b_offset1) + 0);
            b_offset1++;
            b_offset2++;
            b_offset3++;
            b_offset4++;
            bp_offset += 8;
        }
    }

exit:
    return 0;
}

int pack_T16_8bit(dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int i, j;
    int fastpath;
    const int8_t *a_offset;
    const int8_t *a_off[4];
    int8_t *ap_offset;
    int m_cap = (m + 3) & ~3;
    int k_cap = (k + 3) & ~3;
    vec_st vw0, vw1, vw2, vw3, vap[4];
    vec_t swiz = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    a_offset = a;
    ap_offset = ap;
    fastpath = (((k & 3) == 0) && (m & 3) == 0);

    j = (m_cap >> 4);
    int m_skip = (j != (m >> 4));
    while (j) {
        a_off[0] = a_offset;
        a_off[1] = a_off[0] + lda;
        a_off[2] = a_off[1] + lda;
        a_off[3] = a_off[2] + lda;
        a_offset += 16;

        i = (k >> 2);
        while (i) {
            vw0 = vec_splat_s8((int8_t)0);
            vw1 = vec_splat_s8((int8_t)0);
            vw2 = vec_splat_s8((int8_t)0);
            vw3 = vec_splat_s8((int8_t)0);
            int *temp0, *temp1, *temp2, *temp3;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            temp2 = (int *)&vw2;
            temp3 = (int *)&vw3;
            memcpy_4(&temp0[0], &a_off[0][0]);
            memcpy_4(&temp0[1], &a_off[1][0]);
            memcpy_4(&temp0[2], &a_off[2][0]);
            memcpy_4(&temp0[3], &a_off[3][0]);
            memcpy_4(&temp1[0], &a_off[0][4]);
            memcpy_4(&temp1[1], &a_off[1][4]);
            memcpy_4(&temp1[2], &a_off[2][4]);
            memcpy_4(&temp1[3], &a_off[3][4]);
            memcpy_4(&temp2[0], &a_off[0][8]);
            memcpy_4(&temp2[1], &a_off[1][8]);
            memcpy_4(&temp2[2], &a_off[2][8]);
            memcpy_4(&temp2[3], &a_off[3][8]);
            if (fastpath) {
                memcpy_4(&temp3[0], &a_off[0][12]);
                memcpy_4(&temp3[1], &a_off[1][12]);
                memcpy_4(&temp3[2], &a_off[2][12]);
                memcpy_4(&temp3[3], &a_off[3][12]);
            } else {
                int nbytes = ((j > 1) || (!m_skip)) ? 4 : 4 - (m_cap - m);
                memcpy_n(&temp3[0], &a_off[0][12], nbytes);
                memcpy_n(&temp3[1], &a_off[1][12], nbytes);
                memcpy_n(&temp3[2], &a_off[2][12], nbytes);
                memcpy_n(&temp3[3], &a_off[3][12], nbytes);
            }
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vap[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            vap[2] = vec_perm(vw2, vw2, swiz); // 4x4 transpose
            vap[3] = vec_perm(vw3, vw3, swiz); // 4x4 transpose
            memcpy_64(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 64;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            int delk = k_cap - k;
            int nbytes = ((j > 1) || (!m_skip)) ? 4 : 4 - (m_cap - m);
            vw0 = vec_splat_s8((int8_t)0);
            vw1 = vec_splat_s8((int8_t)0);
            vw2 = vec_splat_s8((int8_t)0);
            vw3 = vec_splat_s8((int8_t)0);
            int *temp0, *temp1, *temp2, *temp3;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            temp2 = (int *)&vw2;
            temp3 = (int *)&vw3;
            memcpy_4(&temp0[0], &a_off[0][0]);
            memcpy_4(&temp1[0], &a_off[0][4]);
            memcpy_4(&temp2[0], &a_off[0][8]);
            memcpy_n(&temp3[0], &a_off[0][12], nbytes);
            if (delk < 3) {
                memcpy_4(&temp0[1], &a_off[1][0]);
                memcpy_4(&temp1[1], &a_off[1][4]);
                memcpy_4(&temp2[1], &a_off[1][8]);
                memcpy_n(&temp3[1], &a_off[1][12], nbytes);
            }
            if (delk < 2) {
                memcpy_4(&temp0[2], &a_off[2][0]);
                memcpy_4(&temp1[2], &a_off[2][4]);
                memcpy_4(&temp2[2], &a_off[2][8]);
                memcpy_n(&temp3[2], &a_off[2][12], nbytes);
            }
            if (delk < 1) {
                memcpy_4(&temp0[3], &a_off[3][0]);
                memcpy_4(&temp1[3], &a_off[3][4]);
                memcpy_4(&temp2[3], &a_off[3][8]);
                memcpy_n(&temp3[3], &a_off[3][12], nbytes);
            }
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vap[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            vap[2] = vec_perm(vw2, vw2, swiz); // 4x4 transpose
            vap[3] = vec_perm(vw3, vw3, swiz); // 4x4 transpose
            memcpy_64(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 64;
        }

        j--;
    } // end of while (j)

    if (m_cap & 8) {
        m_skip = (m & 8) != (m_cap & 8);
        a_off[0] = a_offset;
        a_off[1] = a_off[0] + lda;
        a_off[2] = a_off[1] + lda;
        a_off[3] = a_off[2] + lda;
        a_offset += 8;

        i = (k >> 2);
        while (i) {
            vw0 = vec_splat_s8((int8_t)0);
            vw1 = vec_splat_s8((int8_t)0);
            int *temp0, *temp1;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            memcpy_4(&temp0[0], &a_off[0][0]);
            memcpy_4(&temp0[1], &a_off[1][0]);
            memcpy_4(&temp0[2], &a_off[2][0]);
            memcpy_4(&temp0[3], &a_off[3][0]);
            if (fastpath) {
                memcpy_4(&temp1[0], &a_off[0][4]);
                memcpy_4(&temp1[1], &a_off[1][4]);
                memcpy_4(&temp1[2], &a_off[2][4]);
                memcpy_4(&temp1[3], &a_off[3][4]);
            } else {
                int nbytes = (!m_skip) ? 4 : 4 - (m_cap - m);
                memcpy_n(&temp1[0], &a_off[0][4], nbytes);
                memcpy_n(&temp1[1], &a_off[1][4], nbytes);
                memcpy_n(&temp1[2], &a_off[2][4], nbytes);
                memcpy_n(&temp1[3], &a_off[3][4], nbytes);
            }
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vap[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            memcpy_32(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 32;
            i--;
        } // end of while (i)

        if (k < k_cap) {
            int delk = k_cap - k;
            int nbytes = (!m_skip) ? 4 : 4 - (m_cap - m);
            vw0 = vec_splat_s8((int8_t)0);
            vw1 = vec_splat_s8((int8_t)0);
            int *temp0, *temp1;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            memcpy_4(&temp0[0], &a_off[0][0]);
            memcpy_n(&temp1[0], &a_off[0][4], nbytes);
            if (delk < 3) {
                memcpy_4(&temp0[1], &a_off[1][0]);
                memcpy_n(&temp1[1], &a_off[1][4], nbytes);
            }
            if (delk < 2) {
                memcpy_4(&temp0[2], &a_off[2][0]);
                memcpy_n(&temp1[2], &a_off[2][4], nbytes);
            }
            if (delk < 1) {
                memcpy_4(&temp0[3], &a_off[3][0]);
                memcpy_n(&temp1[3], &a_off[3][4], nbytes);
            }
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vap[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            memcpy_32(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 32;
        }
    }

    if (m_cap & 4) {
        m_skip = (m & 4) != (m_cap & 4);
        a_off[0] = a_offset;
        a_off[1] = a_off[0] + lda;
        a_off[2] = a_off[1] + lda;
        a_off[3] = a_off[2] + lda;
        a_offset += 4;

        i = (k >> 2);
        while (i) {
            vw0 = vec_splat_s8((int8_t)0);
            int *temp0;
            temp0 = (int *)&vw0;
            if (fastpath) {
                memcpy_4(&temp0[0], &a_off[0][0]);
                memcpy_4(&temp0[1], &a_off[1][0]);
                memcpy_4(&temp0[2], &a_off[2][0]);
                memcpy_4(&temp0[3], &a_off[3][0]);
            } else {
                int nbytes = (!m_skip) ? 4 : 4 - (m_cap - m);
                memcpy_n(&temp0[0], &a_off[0][0], nbytes);
                memcpy_n(&temp0[1], &a_off[1][0], nbytes);
                memcpy_n(&temp0[2], &a_off[2][0], nbytes);
                memcpy_n(&temp0[3], &a_off[3][0], nbytes);
            }
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            memcpy_16(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 16;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            int delk = k_cap - k;
            int nbytes = (!m_skip) ? 4 : 4 - (m_cap - m);
            vw0 = vec_splat_s8((int8_t)0);
            int *temp0;
            temp0 = (int *)&vw0;
            memcpy_n(&temp0[0], &a_off[0][0], nbytes);
            if (delk < 3) memcpy_n(&temp0[1], &a_off[1][0], nbytes);
            if (delk < 2) memcpy_n(&temp0[2], &a_off[2][0], nbytes);
            if (delk < 1) memcpy_n(&temp0[3], &a_off[3][0], nbytes);
            vap[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            memcpy_16(ap_offset, vap);
            a_off[0] += 4 * lda;
            a_off[1] += 4 * lda;
            a_off[2] += 4 * lda;
            a_off[3] += 4 * lda;
            ap_offset += 16;
        }
    }
    return 0;
}

int pack_N8_8bit(dim_t k, dim_t n, const uint8_t *b, dim_t ldb, uint8_t *bp) {
    int i, j;
    int fastpath;
    const uint8_t *b_offset;
    const uint8_t *b_off[8];
    uint8_t *bp_offset;
    int n_cap = (n + 3) & ~3;
    int k_cap = (k + 3) & ~3;
    fastpath = (((k & 3) == 0) && (n & 3) == 0);

    b_offset = b;
    bp_offset = bp;

    j = (n_cap >> 3);
    int n_skip = (j != (n >> 3));
    while (j) {
        b_off[0] = b_offset;
        b_off[1] = b_off[0] + ldb;
        b_off[2] = b_off[1] + ldb;
        b_off[3] = b_off[2] + ldb;
        b_off[4] = b_off[3] + ldb;
        b_off[5] = b_off[4] + ldb;
        b_off[6] = b_off[5] + ldb;
        b_off[7] = b_off[6] + ldb;
        b_offset += 8 * ldb;

        i = (k >> 2);
        while (i) {
            int *temp = (int *)bp_offset;
            temp[5] = temp[6] = temp[7] = 0;
            memcpy_4(&bp_offset[0], b_off[0]);
            memcpy_4(&bp_offset[4], b_off[1]);
            memcpy_4(&bp_offset[8], b_off[2]);
            memcpy_4(&bp_offset[12], b_off[3]);
            memcpy_4(&bp_offset[16], b_off[4]);
            if (fastpath) {
                memcpy_4(&bp_offset[20], b_off[5]);
                memcpy_4(&bp_offset[24], b_off[6]);
                memcpy_4(&bp_offset[28], b_off[7]);
            } else {
                if ((j > 1) || (!n_skip) || (n_cap - n < 3))
                    memcpy_4(&bp_offset[20], b_off[5]);
                if ((j > 1) || (!n_skip) || (n_cap - n < 2))
                    memcpy_4(&bp_offset[24], b_off[6]);
                if ((j > 1) || (!n_skip) || (n_cap - n < 1))
                    memcpy_4(&bp_offset[28], b_off[7]);
            }
            b_off[0] += 4;
            b_off[1] += 4;
            b_off[2] += 4;
            b_off[3] += 4;
            b_off[4] += 4;
            b_off[5] += 4;
            b_off[6] += 4;
            b_off[7] += 4;
            bp_offset += 32;
            i--;
        } // end of while (i)

        if (k < k_cap) {
            int delk, ii;
            delk = 4 - (k_cap - k);
            for (ii = 0; ii < delk; ++ii)
                *(bp_offset + 0 + ii) = *(b_off[0] + ii);
            for (ii = 0; ii < delk; ++ii)
                *(bp_offset + 4 + ii) = *(b_off[1] + ii);
            for (ii = 0; ii < delk; ++ii)
                *(bp_offset + 8 + ii) = *(b_off[2] + ii);
            for (ii = 0; ii < delk; ++ii)
                *(bp_offset + 12 + ii) = *(b_off[3] + ii);
            for (ii = 0; ii < delk; ++ii)
                *(bp_offset + 16 + ii) = *(b_off[4] + ii);
            if ((j > 1) || (!n_skip) || (n_cap - n < 3))
                for (ii = 0; ii < delk; ++ii)
                    *(bp_offset + 20 + ii) = *(b_off[5] + ii);
            if ((j > 1) || (!n_skip) || (n_cap - n < 2))
                for (ii = 0; ii < delk; ++ii)
                    *(bp_offset + 24 + ii) = *(b_off[6] + ii);
            if ((j > 1) || (!n_skip) || (n_cap - n < 1))
                for (ii = 0; ii < delk; ++ii)
                    *(bp_offset + 28 + ii) = *(b_off[7] + ii);

            for (ii = delk; ii < 4; ++ii)
                *(bp_offset + 0 + ii) = 0;
            for (ii = delk; ii < 4; ++ii)
                *(bp_offset + 4 + ii) = 0;
            for (ii = delk; ii < 4; ++ii)
                *(bp_offset + 8 + ii) = 0;
            for (ii = delk; ii < 4; ++ii)
                *(bp_offset + 12 + ii) = 0;
            for (ii = delk; ii < 4; ++ii)
                *(bp_offset + 16 + ii) = 0;
            if ((j > 1) || (!n_skip) || (n_cap - n < 3))
                for (ii = delk; ii < 4; ++ii)
                    *(bp_offset + 20 + ii) = 0;
            if ((j > 1) || (!n_skip) || (n_cap - n < 2))
                for (ii = delk; ii < 4; ++ii)
                    *(bp_offset + 24 + ii) = 0;
            if ((j > 1) || (!n_skip) || (n_cap - n < 1))
                for (ii = delk; ii < 4; ++ii)
                    *(bp_offset + 28 + ii) = 0;
            bp_offset += 32;
        }

        j--;
    } // end of while (j)

    if (n_cap & 4) {
        b_off[0] = b_offset;
        b_off[1] = b_off[0] + ldb;
        b_off[2] = b_off[1] + ldb;
        b_off[3] = b_off[2] + ldb;

        i = (k_cap >> 2);
        if (fastpath) {
            while (i) {
                *(bp_offset + 0) = *(b_off[0] + 0);
                *(bp_offset + 1) = *(b_off[0] + 1);
                *(bp_offset + 2) = *(b_off[0] + 2);
                *(bp_offset + 3) = *(b_off[0] + 3);
                *(bp_offset + 4) = *(b_off[1] + 0);
                *(bp_offset + 5) = *(b_off[1] + 1);
                *(bp_offset + 6) = *(b_off[1] + 2);
                *(bp_offset + 7) = *(b_off[1] + 3);
                *(bp_offset + 8) = *(b_off[2] + 0);
                *(bp_offset + 9) = *(b_off[2] + 1);
                *(bp_offset + 10) = *(b_off[2] + 2);
                *(bp_offset + 11) = *(b_off[2] + 3);
                *(bp_offset + 12) = *(b_off[3] + 0);
                *(bp_offset + 13) = *(b_off[3] + 1);
                *(bp_offset + 14) = *(b_off[3] + 2);
                *(bp_offset + 15) = *(b_off[3] + 3);
                b_off[0] += 4;
                b_off[1] += 4;
                b_off[2] += 4;
                b_off[3] += 4;
                bp_offset += 16;
                i--;
            } // end of while (i)
        } else {
            int ncopy1 = (n_cap - n < 3);
            int ncopy2 = (n_cap - n < 2);
            int ncopy3 = (n_cap - n < 1);
            while (i > 1) {
                *(bp_offset + 0) = *(b_off[0] + 0);
                *(bp_offset + 1) = *(b_off[0] + 1);
                *(bp_offset + 2) = *(b_off[0] + 2);
                *(bp_offset + 3) = *(b_off[0] + 3);
                *(bp_offset + 4) = ncopy1 ? *(b_off[1] + 0) : 0;
                *(bp_offset + 5) = ncopy1 ? *(b_off[1] + 1) : 0;
                *(bp_offset + 6) = ncopy1 ? *(b_off[1] + 2) : 0;
                *(bp_offset + 7) = ncopy1 ? *(b_off[1] + 3) : 0;
                *(bp_offset + 8) = ncopy2 ? *(b_off[2] + 0) : 0;
                *(bp_offset + 9) = ncopy2 ? *(b_off[2] + 1) : 0;
                *(bp_offset + 10) = ncopy2 ? *(b_off[2] + 2) : 0;
                *(bp_offset + 11) = ncopy2 ? *(b_off[2] + 3) : 0;
                *(bp_offset + 12) = ncopy3 ? *(b_off[3] + 0) : 0;
                *(bp_offset + 13) = ncopy3 ? *(b_off[3] + 1) : 0;
                *(bp_offset + 14) = ncopy3 ? *(b_off[3] + 2) : 0;
                *(bp_offset + 15) = ncopy3 ? *(b_off[3] + 3) : 0;
                b_off[0] += 4;
                b_off[1] += 4;
                b_off[2] += 4;
                b_off[3] += 4;
                bp_offset += 16;
                i--;
            } // end of while (i>1)

            int kcopy1 = (k_cap - k < 3);
            int kcopy2 = (k_cap - k < 2);
            int kcopy3 = (k_cap - k < 1);

            *(bp_offset + 0) = *(b_off[0] + 0);
            *(bp_offset + 1) = (kcopy1) ? *(b_off[0] + 1) : 0;
            *(bp_offset + 2) = (kcopy2) ? *(b_off[0] + 2) : 0;
            *(bp_offset + 3) = (kcopy3) ? *(b_off[0] + 3) : 0;
            *(bp_offset + 4) = (ncopy1) ? *(b_off[1] + 0) : 0;
            *(bp_offset + 5) = (kcopy1 && ncopy1) ? *(b_off[1] + 1) : 0;
            *(bp_offset + 6) = (kcopy2 && ncopy1) ? *(b_off[1] + 2) : 0;
            *(bp_offset + 7) = (kcopy3 && ncopy1) ? *(b_off[1] + 3) : 0;
            *(bp_offset + 8) = (ncopy2) ? *(b_off[2] + 0) : 0;
            *(bp_offset + 9) = (kcopy1 && ncopy2) ? *(b_off[2] + 1) : 0;
            *(bp_offset + 10) = (kcopy2 && ncopy2) ? *(b_off[2] + 2) : 0;
            *(bp_offset + 11) = (kcopy3 && ncopy2) ? *(b_off[2] + 3) : 0;
            *(bp_offset + 12) = (ncopy3) ? *(b_off[3] + 0) : 0;
            *(bp_offset + 13) = (kcopy1 && ncopy3) ? *(b_off[3] + 1) : 0;
            *(bp_offset + 14) = (kcopy2 && ncopy3) ? *(b_off[3] + 2) : 0;
            *(bp_offset + 15) = (kcopy3 && ncopy3) ? *(b_off[3] + 3) : 0;
        }
    }

    return 0;
}

int pack_N16_8bit(dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int i, j, ii;
    int fastpath;
    const int8_t *a_offset, *a_off[16];
    int8_t *ap_offset;
    int m_cap = (m + 3) & ~3;
    int k_cap = (k + 3) & ~3;
    a_offset = a;
    ap_offset = ap;
    fastpath = (((k & 3) == 0) && (m & 3) == 0);

    j = (m_cap >> 4);
    int m_skip = (j != (m >> 4));
    while (j) {
        for (ii = 0; ii < 16; ++ii)
            a_off[ii] = a_offset + ii * lda;
        a_offset += 16 * lda;
        i = (k >> 2);
        while (i) {
            int *temp = (int *)ap_offset;
            temp[13] = temp[14] = temp[15] = 0;
            memcpy_4(&ap_offset[0], a_off[0]);
            memcpy_4(&ap_offset[4], a_off[1]);
            memcpy_4(&ap_offset[8], a_off[2]);
            memcpy_4(&ap_offset[12], a_off[3]);
            memcpy_4(&ap_offset[16], a_off[4]);
            memcpy_4(&ap_offset[20], a_off[5]);
            memcpy_4(&ap_offset[24], a_off[6]);
            memcpy_4(&ap_offset[28], a_off[7]);
            memcpy_4(&ap_offset[32], a_off[8]);
            memcpy_4(&ap_offset[36], a_off[9]);
            memcpy_4(&ap_offset[40], a_off[10]);
            memcpy_4(&ap_offset[44], a_off[11]);
            memcpy_4(&ap_offset[48], a_off[12]);
            if (fastpath) {
                memcpy_4(&ap_offset[52], a_off[13]);
                memcpy_4(&ap_offset[56], a_off[14]);
                memcpy_4(&ap_offset[60], a_off[15]);
            } else {
                if ((j > 1) || (!m_skip) || (m_cap - m < 3))
                    memcpy_4(&ap_offset[52], a_off[13]);
                if ((j > 1) || (!m_skip) || (m_cap - m < 2))
                    memcpy_4(&ap_offset[56], a_off[14]);
                if ((j > 1) || (!m_skip) || (m_cap - m < 1))
                    memcpy_4(&ap_offset[60], a_off[15]);
            }
            for (ii = 0; ii < 16; ++ii)
                a_off[ii] += 4;
            ap_offset += 64;
            i--;
        } // end of while (i)

        if (k < k_cap) {
            int *temp = (int *)ap_offset;
            for (int ii = 0; ii < 16; ++ii)
                temp[ii] = 0;
            int delk = 4 - (k_cap - k);
            memcpy_n(&ap_offset[0], a_off[0], delk);
            memcpy_n(&ap_offset[4], a_off[1], delk);
            memcpy_n(&ap_offset[8], a_off[2], delk);
            memcpy_n(&ap_offset[12], a_off[3], delk);
            memcpy_n(&ap_offset[16], a_off[4], delk);
            memcpy_n(&ap_offset[20], a_off[5], delk);
            memcpy_n(&ap_offset[24], a_off[6], delk);
            memcpy_n(&ap_offset[28], a_off[7], delk);
            memcpy_n(&ap_offset[32], a_off[8], delk);
            memcpy_n(&ap_offset[36], a_off[9], delk);
            memcpy_n(&ap_offset[40], a_off[10], delk);
            memcpy_n(&ap_offset[44], a_off[11], delk);
            memcpy_n(&ap_offset[48], a_off[12], delk);
            if ((j != 1) || (!m_skip) || (m_cap - m < 3))
                memcpy_n(&ap_offset[52], a_off[13], delk);
            if ((j != 1) || (!m_skip) || (m_cap - m < 2))
                memcpy_n(&ap_offset[56], a_off[14], delk);
            if ((j != 1) || (!m_skip) || (m_cap - m < 1))
                memcpy_n(&ap_offset[60], a_off[15], delk);
            ap_offset += 64;
        }
        j--;
    } // end of while(j)

    if (m_cap & 8) {
        m_skip = (m & 8) != (m_cap & 8);
        for (ii = 0; ii < 8; ++ii)
            a_off[ii] = a_offset + ii * lda;
        a_offset += 8 * lda;
        i = (k >> 2);
        while (i) {
            int *temp = (int *)ap_offset;
            temp[5] = temp[6] = temp[7] = 0;
            memcpy_4(&ap_offset[0], a_off[0]);
            memcpy_4(&ap_offset[4], a_off[1]);
            memcpy_4(&ap_offset[8], a_off[2]);
            memcpy_4(&ap_offset[12], a_off[3]);
            memcpy_4(&ap_offset[16], a_off[4]);
            if (fastpath) {
                memcpy_4(&ap_offset[20], a_off[5]);
                memcpy_4(&ap_offset[24], a_off[6]);
                memcpy_4(&ap_offset[28], a_off[7]);
            } else {
                if ((!m_skip) || (m_cap - m < 3))
                    memcpy_4(&ap_offset[20], a_off[5]);
                if ((!m_skip) || (m_cap - m < 2))
                    memcpy_4(&ap_offset[24], a_off[6]);
                if ((!m_skip) || (m_cap - m < 1))
                    memcpy_4(&ap_offset[28], a_off[7]);
            }
            for (ii = 0; ii < 8; ++ii)
                a_off[ii] += 4;
            ap_offset += 32;
            i--;
        } // end of while (i)

        if (k < k_cap) {
            int *temp = (int *)ap_offset;
            for (int ii = 0; ii < 8; ++ii)
                temp[ii] = 0;
            int delk = 4 - (k_cap - k);
            memcpy_n(&ap_offset[0], a_off[0], delk);
            memcpy_n(&ap_offset[4], a_off[1], delk);
            memcpy_n(&ap_offset[8], a_off[2], delk);
            memcpy_n(&ap_offset[12], a_off[3], delk);
            memcpy_n(&ap_offset[16], a_off[4], delk);
            if ((!m_skip) || (m_cap - m < 3))
                memcpy_n(&ap_offset[20], a_off[5], delk);
            if ((!m_skip) || (m_cap - m < 2))
                memcpy_n(&ap_offset[24], a_off[6], delk);
            if ((!m_skip) || (m_cap - m < 1))
                memcpy_n(&ap_offset[28], a_off[7], delk);
            ap_offset += 32;
        }
    }

    if (m_cap & 4) {
        m_skip = (m & 4) != (m_cap & 4);
        for (ii = 0; ii < 4; ++ii)
            a_off[ii] = a_offset + ii * lda;
        a_offset += 4 * lda;
        i = (k >> 2);
        while (i) {
            int *temp = (int *)ap_offset;
            temp[1] = temp[2] = temp[3] = 0;
            memcpy_4(&ap_offset[0], a_off[0]);
            if (fastpath) {
                memcpy_4(&ap_offset[4], a_off[1]);
                memcpy_4(&ap_offset[8], a_off[2]);
                memcpy_4(&ap_offset[12], a_off[3]);
            } else {
                if ((!m_skip) || (m_cap - m < 3))
                    memcpy_4(&ap_offset[4], a_off[1]);
                if ((!m_skip) || (m_cap - m < 2))
                    memcpy_4(&ap_offset[8], a_off[2]);
                if ((!m_skip) || (m_cap - m < 1))
                    memcpy_4(&ap_offset[12], a_off[3]);
            }
            for (ii = 0; ii < 4; ++ii)
                a_off[ii] += 4;
            ap_offset += 16;

            i--;
        } // end of while (i)

        if (k < k_cap) {
            int *temp = (int *)ap_offset;
            for (int ii = 0; ii < 4; ++ii)
                temp[ii] = 0;
            int delk = 4 - (k_cap - k);
            memcpy_n(&ap_offset[0], a_off[0], delk);
            if ((!m_skip) || (m_cap - m < 3))
                memcpy_n(&ap_offset[4], a_off[1], delk);
            if ((!m_skip) || (m_cap - m < 2))
                memcpy_n(&ap_offset[8], a_off[2], delk);
            if ((!m_skip) || (m_cap - m < 1))
                memcpy_n(&ap_offset[12], a_off[3], delk);
            ap_offset += 16;
        }
    }
    return 0;
}

int pack_T8_8bit(dim_t k, dim_t n, const uint8_t *b, dim_t ldb, uint8_t *bp) {
    int i, j, ii;
    int fastpath;
    const uint8_t *b_offset;
    const uint8_t *b_off[8];
    uint8_t *bp_offset, *bp_offset1, *bp_offset2;
    int n_cap = (n + 3) & ~3;
    int k_cap = (k + 3) & ~3;
    int delk = k_cap - k;
    vec_t vw0, vw1, vw2, vw3, vbp[4];
    vec_t swiz = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    b_offset = b;
    bp_offset = bp;
    bp_offset2 = bp + k_cap * (n_cap & ~7);
    fastpath = (((k & 3) == 0) && (n & 3) == 0);

    j = (k_cap >> 3);
    int k_skip = (j != (k >> 3));
    while (j) {
        for (ii = 0; ii < 8; ++ii)
            b_off[ii] = b_offset + ii * ldb;
        b_offset += 8 * ldb;
        bp_offset1 = bp_offset;
        bp_offset += 64;

        i = (n_cap >> 3);
        // we need to be careful about not going past the end of the B array if n is less than n_cap.
        // fortunately, we can only go out-of-bounds by accessing elements of b_off[5/6/7], so the others
        // can just load garbage in the not_used elements of the vtemp vectors.
        int n_skip = (i != (n >> 3));
        while (i) {
            vw0 = vec_splat_u8((uint8_t)0);
            vw1 = vec_splat_u8((uint8_t)0);
            vw2 = vec_splat_u8((uint8_t)0);
            vw3 = vec_splat_u8((uint8_t)0);
            int *temp0, *temp1, *temp2, *temp3;
            int zerodata = 0;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            temp2 = (int *)&vw2;
            temp3 = (int *)&vw3;
            memcpy_4(&temp0[0], &b_off[0][0]);
            memcpy_4(&temp0[1], &b_off[1][0]);
            memcpy_4(&temp0[2], &b_off[2][0]);
            memcpy_4(&temp0[3], &b_off[3][0]);
            memcpy_4(&temp2[0], &b_off[4][0]);
            if (j > 1) {
                memcpy_4(&temp2[1], &b_off[5][0]);
                memcpy_4(&temp2[2], &b_off[6][0]);
                memcpy_4(&temp2[3], &b_off[7][0]);
            } else {
                memcpy_4(&temp2[1], (const uint8_t *)&zerodata);
                memcpy_4(&temp2[2], (const uint8_t *)&zerodata);
                memcpy_4(&temp2[3], (const uint8_t *)&zerodata);
                if ((!(k_skip)) || (delk < 3))
                    memcpy_4(&temp2[1], &b_off[5][0]);
                if ((!(k_skip)) || (delk < 2))
                    memcpy_4(&temp2[2], &b_off[6][0]);
                if ((!(k_skip)) || (delk < 1))
                    memcpy_4(&temp2[3], &b_off[7][0]);
            }
            if (fastpath) {
                memcpy_4(&temp1[0], &b_off[0][4]);
                memcpy_4(&temp1[1], &b_off[1][4]);
                memcpy_4(&temp1[2], &b_off[2][4]);
                memcpy_4(&temp1[3], &b_off[3][4]);
                memcpy_4(&temp3[0], &b_off[4][4]);
                if (j > 1) {
                    memcpy_4(&temp3[1], &b_off[5][4]);
                    memcpy_4(&temp3[2], &b_off[6][4]);
                    memcpy_4(&temp3[3], &b_off[7][4]);
                } else {
                    memcpy_4(&temp3[1], (const uint8_t *)&zerodata);
                    memcpy_4(&temp3[2], (const uint8_t *)&zerodata);
                    memcpy_4(&temp3[3], (const uint8_t *)&zerodata);
                    if (delk < 3) memcpy_4(&temp3[1], &b_off[5][4]);
                    if (delk < 2) memcpy_4(&temp3[2], &b_off[6][4]);
                    if (delk < 1) memcpy_4(&temp3[3], &b_off[7][4]);
                }
            } else {
                int nbytes = ((i > 1) || (!(n_skip))) ? 4 : 4 - (n_cap - n);
                memcpy_n(&temp1[0], &b_off[0][4], nbytes);
                memcpy_n(&temp1[1], &b_off[1][4], nbytes);
                memcpy_n(&temp1[2], &b_off[2][4], nbytes);
                memcpy_n(&temp1[3], &b_off[3][4], nbytes);
                memcpy_n(&temp3[0], &b_off[4][4], nbytes);
                if (j > 1) {
                    memcpy_n(&temp3[1], &b_off[5][4], nbytes);
                    memcpy_n(&temp3[2], &b_off[6][4], nbytes);
                    memcpy_n(&temp3[3], &b_off[7][4], nbytes);
                } else {
                    memcpy_n(&temp3[1], (const uint8_t *)&zerodata, nbytes);
                    memcpy_n(&temp3[2], (const uint8_t *)&zerodata, nbytes);
                    memcpy_n(&temp3[3], (const uint8_t *)&zerodata, nbytes);
                    if ((!(k_skip)) || (delk < 3))
                        memcpy_n(&temp3[1], &b_off[5][4], nbytes);
                    if ((!(k_skip)) || (delk < 2))
                        memcpy_n(&temp3[2], &b_off[6][4], nbytes);
                    if ((!(k_skip)) || (delk < 1))
                        memcpy_n(&temp3[3], &b_off[7][4], nbytes);
                }
            }
            vbp[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vbp[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            vbp[2] = vec_perm(vw2, vw2, swiz); // 4x4 transpose
            vbp[3] = vec_perm(vw3, vw3, swiz); // 4x4 transpose
            memcpy_64(bp_offset1, vbp);
            for (ii = 0; ii < 8; ++ii)
                b_off[ii] += 8;
            bp_offset1 += k_cap * 8;
            i--;
        } // end of while (i)

        if (n_cap & 4) {
            vw0 = vec_splat_u8((uint8_t)0);
            vw1 = vec_splat_u8((uint8_t)0);
            int *temp0, *temp1;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            if (fastpath) {
                memcpy_4(&temp0[0], b_off[0]);
                memcpy_4(&temp0[1], b_off[1]);
                memcpy_4(&temp0[2], b_off[2]);
                memcpy_4(&temp0[3], b_off[3]);
                memcpy_4(&temp1[0], b_off[4]);
                memcpy_4(&temp1[1], b_off[5]);
                memcpy_4(&temp1[2], b_off[6]);
                memcpy_4(&temp1[3], b_off[7]);
            } else {
                int nbytes = 4 - (n_cap - n);
                memcpy_n(&temp0[0], b_off[0], nbytes);
                memcpy_n(&temp0[1], b_off[1], nbytes);
                memcpy_n(&temp0[2], b_off[2], nbytes);
                memcpy_n(&temp0[3], b_off[3], nbytes);
                memcpy_n(&temp1[0], b_off[4], nbytes);
                if (j > 1 || (!k_skip) || delk < 3)
                    memcpy_n(&temp1[1], b_off[5], nbytes);
                if (j > 1 || (!k_skip) || delk < 2)
                    memcpy_n(&temp1[2], b_off[6], nbytes);
                if (j > 1 || (!k_skip) || delk < 1)
                    memcpy_n(&temp1[3], b_off[7], nbytes);
            }
            vbp[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vbp[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            memcpy_32(bp_offset2, vbp);
            for (ii = 0; ii < 8; ++ii)
                b_off[ii] += 4;
            bp_offset2 += 32;
        }

        j--;
    } // end of while (j)

    if (k_cap & 4) {
        for (ii = 0; ii < 4; ++ii)
            b_off[ii] = b_offset + ii * ldb;
        b_offset += 4 * ldb;
        bp_offset1 = bp_offset;
        bp_offset += 32;

        i = (n_cap >> 3);
        int n_skip = (i != (n >> 3));
        while (i) {
            vw0 = vec_splat_u8((uint8_t)0);
            vw1 = vec_splat_u8((uint8_t)0);
            int *temp0, *temp1;
            temp0 = (int *)&vw0;
            temp1 = (int *)&vw1;
            memcpy_4(&temp0[0], &b_off[0][0]);
            if (fastpath) {
                memcpy_4(&temp0[1], &b_off[1][0]);
                memcpy_4(&temp0[2], &b_off[2][0]);
                memcpy_4(&temp0[3], &b_off[3][0]);
                memcpy_4(&temp1[0], &b_off[0][4]);
                memcpy_4(&temp1[1], &b_off[1][4]);
                memcpy_4(&temp1[2], &b_off[2][4]);
                memcpy_4(&temp1[3], &b_off[3][4]);
            } else {
                int nbytes = ((i > 1) || (!(n_skip))) ? 4 : 4 - (n_cap - n);
                if (delk < 3) memcpy_4(&temp0[1], &b_off[1][0]);
                if (delk < 2) memcpy_4(&temp0[2], &b_off[2][0]);
                if (delk < 1) memcpy_4(&temp0[3], &b_off[3][0]);
                memcpy_n(&temp1[0], &b_off[0][4], nbytes);
                if (delk < 3) memcpy_n(&temp1[1], &b_off[1][4], nbytes);
                if (delk < 2) memcpy_n(&temp1[2], &b_off[2][4], nbytes);
                if (delk < 1) memcpy_n(&temp1[3], &b_off[3][4], nbytes);
            }
            vbp[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            vbp[1] = vec_perm(vw1, vw1, swiz); // 4x4 transpose
            memcpy_32(bp_offset1, vbp);
            for (ii = 0; ii < 4; ++ii)
                b_off[ii] += 8;
            bp_offset1 += k_cap * 8;
            i--;
        } // end of while (i)

        if (n_cap & 4) {
            int nbytes = 4 - (n_cap - n);
            vw0 = vec_splat_u8((uint8_t)0);
            int *temp0;
            temp0 = (int *)&vw0;
            if (fastpath) {
                memcpy_4(&temp0[0], b_off[0]);
                memcpy_4(&temp0[1], b_off[1]);
                memcpy_4(&temp0[2], b_off[2]);
                memcpy_4(&temp0[3], b_off[3]);
            } else {
                memcpy_n(&temp0[0], b_off[0], nbytes);
                if (delk < 3) memcpy_n(&temp0[1], b_off[1], nbytes);
                if (delk < 2) memcpy_n(&temp0[2], b_off[2], nbytes);
                if (delk < 1) memcpy_n(&temp0[3], b_off[3], nbytes);
            }
            vbp[0] = vec_perm(vw0, vw0, swiz); // 4x4 transpose
            memcpy_16(bp_offset2, vbp);
        }
    }

    return 0;
}

typedef __vector signed int v4si_t __attribute__((aligned(4)));

#define SWIZZLE_4x4 \
    { \
        result_i[0] = vec_perm(result[0], result[1], swizA); \
        result_i[1] = vec_perm(result[0], result[1], swizB); \
        result_i[2] = vec_perm(result[2], result[3], swizA); \
        result_i[3] = vec_perm(result[2], result[3], swizB); \
        result_t[0] = vec_perm(result_i[0], result_i[2], swizC); \
        result_t[1] = vec_perm(result_i[0], result_i[2], swizD); \
        result_t[2] = vec_perm(result_i[1], result_i[3], swizC); \
        result_t[3] = vec_perm(result_i[1], result_i[3], swizD); \
    }

#define SAVE_ACC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC1(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC1_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC1_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SAVE_ACC1_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SET_ACC_ZERO4() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3);

#define SET_ACC_ZERO8() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3); \
    __builtin_mma_xxsetaccz(&acc4); \
    __builtin_mma_xxsetaccz(&acc5); \
    __builtin_mma_xxsetaccz(&acc6); \
    __builtin_mma_xxsetaccz(&acc7);

#define PREFETCH1(x, y) \
    asm volatile("dcbt %0, %1" : : "r"(x), "b"(y) : "memory");

#define MMA __builtin_mma_xvi16ger2pp

void gemm_kernel_16bit(dim_t m, dim_t n, dim_t k, float alpha, short *A,
        short *B, int *C, float beta, dim_t ldc) {
    int i;
    int m_cap = (m + 3) & ~3;
    int n_cap = (n + 3) & ~3;
    int k_cap = (k + 1) & ~1;
    int m_skip;
    int n_skip = (n & 8) != (n_cap & 8);
    int fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int j;
        int *CO;
        short *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l = 0;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int j;
        int *CO;
        short *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            short *BO = B;
            short *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int count = 4 - (m_cap - m);
                        int ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int count = 4 - (m_cap - m);
                        int ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int count = 4 - (m_cap - m);
                    int ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int count = 4 - (m_cap - m);
                    int ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

#undef MMA
#define MMA __builtin_mma_xvi8ger4pp

void gemm_kernel_8bit(dim_t m, dim_t n, dim_t k, float alpha, int8_t *A,
        uint8_t *B, int *C, float beta, dim_t ldc) {
    int i;
    int m_cap = (m + 3) & ~3;
    int n_cap = (n + 3) & ~3;
    int k_cap = (k + 3) & ~3;
    int m_skip;
    int n_skip = (n & 8) != (n_cap & 8);
    int fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int j;
        int *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l = 0;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int count = 4 - (m_cap - m);
                        int ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int j;
        int *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            uint8_t *BO = B;
            int8_t *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int count = 4 - (m_cap - m);
                        int ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int count = 4 - (m_cap - m);
                        int ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int ii;
                        int count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int count = 4 - (m_cap - m);
                    int ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int count = 4 - (m_cap - m);
                    int ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

} // namespace impl
} // namespace dnnl
