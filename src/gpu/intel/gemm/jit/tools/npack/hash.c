/*******************************************************************************
* Copyright 2019 Intel Corporation
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

/*********************************************/
/* A Jenkins hash function, as found in NEO: */
/*    runtime/helpers/hash.h                 */
/*********************************************/

#include "hash.h"

// clang-format off
#define HASH_JENKINS_MIX(a,b,c) \
{ \
    a -= b; a -= c; a ^= (c>>13); \
    b -= c; b -= a; b ^= (a<<8);  \
    c -= a; c -= b; c ^= (b>>13); \
    a -= b; a -= c; a ^= (c>>12); \
    b -= c; b -= a; b ^= (a<<16); \
    c -= a; c -= b; c ^= (b>>5);  \
    a -= b; a -= c; a ^= (c>>3);  \
    b -= c; b -= a; b ^= (a<<10); \
    c -= a; c -= b; c ^= (b>>15); \
}
// clang-format on

uint32_t neo_hash(const char *buf, size_t len)
{
    const uint32_t *ubuf = (const uint32_t *)buf;

    uint32_t a = 0x428a2f98;
    uint32_t hi = 0x71374491;
    uint32_t lo = 0xb5c0fbcf;

    for (; len >= 4; len -= 4) {
        a ^= *ubuf++;
        HASH_JENKINS_MIX(a, hi, lo);
    }

    if (len > 0) {
        const uint8_t *rbuf = (const uint8_t *)ubuf;
        uint32_t rem = 0;
        for (; len > 0; len--)
            rem = (rem | *rbuf++) << 8;
        HASH_JENKINS_MIX(rem, hi, lo);
    }

    return lo;
}
