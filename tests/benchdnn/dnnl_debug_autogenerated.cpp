/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

// DO NOT EDIT, AUTO-GENERATED

// clang-format off

#include <assert.h>
#include <string.h>

#include "dnnl_debug.h"
#include "dnnl_debug.hpp"

#include "src/common/z_magic.hpp"

dnnl_data_type_t str2dt(const char *str) {
#define CASE(_case) do { \
    if (!strcmp(STRINGIFY(_case), str) \
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \
        return CONCAT2(dnnl_, _case); \
} while (0)
    CASE(f16);
    CASE(bf16);
    CASE(f32);
    CASE(s32);
    CASE(s8);
    CASE(u8);
#undef CASE
    if (!strcmp("undef", str) || !strcmp("dnnl_data_type_undef", str))
        return dnnl_data_type_undef;
    assert(!"unknown dt");
    return dnnl_data_type_undef;
}

dnnl_format_tag_t str2fmt_tag(const char *str) {
#define CASE(_case) do { \
    if (!strcmp(STRINGIFY(_case), str) \
            || !strcmp("dnnl_" STRINGIFY(_case), str)) \
        return CONCAT2(dnnl_, _case); \
} while (0)
    CASE(a);
    CASE(ab);
    CASE(abc);
    CASE(abcd);
    CASE(abcde);
    CASE(abcdef);
    CASE(abdec);
    CASE(acb);
    CASE(acbde);
    CASE(acbdef);
    CASE(acdb);
    CASE(acdeb);
    CASE(ba);
    CASE(bac);
    CASE(bacd);
    CASE(bca);
    CASE(bcda);
    CASE(bcdea);
    CASE(cba);
    CASE(cdba);
    CASE(cdeba);
    CASE(decab);
    CASE(Abc16a);
    CASE(ABc16a16b);
    CASE(ABc4a4b);
    CASE(aBc16b);
    CASE(ABc16b16a);
    CASE(Abc4a);
    CASE(aBc4b);
    CASE(ABc4b16a4b);
    CASE(ABc2b8a4b);
    CASE(ABc4b4a);
    CASE(ABc8a16b2a);
    CASE(ABc8a8b);
    CASE(aBc8b);
    CASE(ABc8b16a2b);
    CASE(BAc8a16b2a);
    CASE(ABc8b8a);
    CASE(Abcd16a);
    CASE(Abcd8a);
    CASE(ABcd16a16b);
    CASE(ABcd32a32b);
    CASE(aBcd16b);
    CASE(ABcd16b16a);
    CASE(aBCd16b16c);
    CASE(aBCd16c16b);
    CASE(Abcd4a);
    CASE(aBcd4b);
    CASE(ABcd4b16a4b);
    CASE(ABcd4b4a);
    CASE(ABcd4a4b);
    CASE(aBCd4c16b4c);
    CASE(aBCd2c8b4c);
    CASE(aBCd4c4b);
    CASE(aBCd4b4c);
    CASE(ABcd8a16b2a);
    CASE(ABcd2b8a4b);
    CASE(ABcd8a8b);
    CASE(aBcd8b);
    CASE(ABcd8b16a2b);
    CASE(aBCd8b16c2b);
    CASE(BAcd8a16b2a);
    CASE(ABcd8b8a);
    CASE(aBCd8b8c);
    CASE(aBCd8c16b2c);
    CASE(ABcde8a16b2a);
    CASE(aCBd8b16c2b);
    CASE(aBCd8c8b);
    CASE(Abcde16a);
    CASE(ABcde16a16b);
    CASE(BAcde8a16b2a);
    CASE(ABcde4b16a4b);
    CASE(aBcde16b);
    CASE(ABcde16b16a);
    CASE(aBCde16b16c);
    CASE(aBCde16c16b);
    CASE(aBCde2c8b4c);
    CASE(Abcde4a);
    CASE(aBcde4b);
    CASE(ABcde4b4a);
    CASE(ABcde4a4b);
    CASE(aBCde4b4c);
    CASE(aBCde4c16b4c);
    CASE(aBCde4c4b);
    CASE(Abcde8a);
    CASE(ABcde8a8b);
    CASE(BAcde16b16a);
    CASE(aBcde8b);
    CASE(ABcde8b16a2b);
    CASE(aBCde8b16c2b);
    CASE(aCBde8b16c2b);
    CASE(ABcde8b8a);
    CASE(aBCde8b8c);
    CASE(ABcd4a8b8a4b);
    CASE(ABcd2a8b8a2b);
    CASE(aBCde4b8c8b4c);
    CASE(aBCde2b8c8b2c);
    CASE(aBCde8c16b2c);
    CASE(aBCde8c8b);
    CASE(aBcdef16b);
    CASE(aBCdef16b16c);
    CASE(aBCdef16c16b);
    CASE(aBCdef4c16b4c);
    CASE(aBcdef4b);
    CASE(aBCdef4c4b);
    CASE(aBCdef4b4c);
    CASE(aBCdef8b8c);
    CASE(aBCdef8c16b2c);
    CASE(aBCdef8b16c2b);
    CASE(aCBdef8b16c2b);
    CASE(aBCdef8c8b);
    CASE(aBdc16b);
    CASE(aBdc4b);
    CASE(aBdc8b);
    CASE(aBdec16b);
    CASE(aBdec32b);
    CASE(aBdec4b);
    CASE(aBdec8b);
    CASE(aBdefc16b);
    CASE(aCBdef16c16b);
    CASE(aBdefc4b);
    CASE(aBdefc8b);
    CASE(Abcdef16a);
    CASE(Acb16a);
    CASE(Acb4a);
    CASE(Acb8a);
    CASE(aCBd16b16c);
    CASE(aCBd16c16b);
    CASE(aCBde16b16c);
    CASE(aCBde16c16b);
    CASE(Acdb16a);
    CASE(Acdb32a);
    CASE(Acdb4a);
    CASE(Acdb8a);
    CASE(Acdeb16a);
    CASE(Acdeb4a);
    CASE(Acdeb8a);
    CASE(BAc16a16b);
    CASE(BAc16b16a);
    CASE(BAcd16a16b);
    CASE(BAcd16b16a);
    CASE(x);
    CASE(nc);
    CASE(cn);
    CASE(tn);
    CASE(nt);
    CASE(ncw);
    CASE(nwc);
    CASE(nchw);
    CASE(nhwc);
    CASE(chwn);
    CASE(ncdhw);
    CASE(ndhwc);
    CASE(oi);
    CASE(io);
    CASE(oiw);
    CASE(owi);
    CASE(wio);
    CASE(iwo);
    CASE(oihw);
    CASE(hwio);
    CASE(ohwi);
    CASE(ihwo);
    CASE(iohw);
    CASE(oidhw);
    CASE(dhwio);
    CASE(odhwi);
    CASE(idhwo);
    CASE(goiw);
    CASE(goihw);
    CASE(hwigo);
    CASE(giohw);
    CASE(goidhw);
    CASE(giodhw);
    CASE(tnc);
    CASE(ntc);
    CASE(ldnc);
    CASE(ldigo);
    CASE(ldgoi);
    CASE(ldgo);
    CASE(nCdhw16c);
    CASE(nCdhw4c);
    CASE(nCdhw8c);
    CASE(nChw16c);
    CASE(nChw4c);
    CASE(nChw8c);
    CASE(nCw16c);
    CASE(nCw4c);
    CASE(nCw8c);
    CASE(NCw16n16c);
    CASE(NCdhw16n16c);
    CASE(NChw16n16c);
    CASE(NChw32n32c);
    CASE(IOw16o16i);
    CASE(IOw16i16o);
    CASE(OIw16i16o);
    CASE(OIw16o16i);
    CASE(Oiw16o);
    CASE(OIw4i16o4i);
    CASE(OIw2i8o4i);
    CASE(OIw4i4o);
    CASE(OIw4o4i);
    CASE(Oiw4o);
    CASE(OIw8i16o2i);
    CASE(OIw8i8o);
    CASE(OIw8o16i2o);
    CASE(IOw8o16i2o);
    CASE(OIw8o8i);
    CASE(Owi16o);
    CASE(Owi4o);
    CASE(Owi8o);
    CASE(IOhw16i16o);
    CASE(IOhw16o16i);
    CASE(Ohwi16o);
    CASE(Ohwi32o);
    CASE(Ohwi4o);
    CASE(Ohwi8o);
    CASE(OIhw16i16o);
    CASE(OIhw16o16i);
    CASE(Oihw16o);
    CASE(OIhw4i16o4i);
    CASE(OIhw4i4o);
    CASE(OIhw4o4i);
    CASE(Oihw4o);
    CASE(OIhw8i16o2i);
    CASE(OIhw8i8o);
    CASE(OIhw8o16i2o);
    CASE(OIhw2i8o4i);
    CASE(IOhw8o16i2o);
    CASE(OIhw8o8i);
    CASE(Odhwi16o);
    CASE(Odhwi4o);
    CASE(Odhwi8o);
    CASE(OIdhw16i16o);
    CASE(OIdhw16o16i);
    CASE(Oidhw16o);
    CASE(OIdhw4i4o);
    CASE(OIdhw4o4i);
    CASE(Oidhw4o);
    CASE(OIdhw8i16o2i);
    CASE(OIdhw8i8o);
    CASE(OIdhw8o16i2o);
    CASE(IOdhw8o16i2o);
    CASE(OIdhw4i16o4i);
    CASE(OIdhw8o8i);
    CASE(IOdhw16i16o);
    CASE(Goiw16g);
    CASE(Goiw8g);
    CASE(gIOw16o16i);
    CASE(gIOw16i16o);
    CASE(gOIw16i16o);
    CASE(gOIw16o16i);
    CASE(gOiw16o);
    CASE(gOIw4i16o4i);
    CASE(gOIw2i8o4i);
    CASE(gOIw4i4o);
    CASE(gOIw4o4i);
    CASE(gOiw4o);
    CASE(gOIw8i16o2i);
    CASE(gOIw8i8o);
    CASE(gOIw8o16i2o);
    CASE(gIOw8o16i2o);
    CASE(gOIw8o8i);
    CASE(gOwi16o);
    CASE(gOwi4o);
    CASE(gOwi8o);
    CASE(gIOhw16i16o);
    CASE(gIOhw16o16i);
    CASE(gOhwi16o);
    CASE(gOhwi32o);
    CASE(gOhwi4o);
    CASE(gOhwi8o);
    CASE(Goihw16g);
    CASE(gOIhw16i16o);
    CASE(gOIhw16o16i);
    CASE(gOihw16o);
    CASE(gOIhw2i8o4i);
    CASE(gOIhw4i16o4i);
    CASE(gOIhw4i4o);
    CASE(gOIhw4o4i);
    CASE(gOihw4o);
    CASE(Goihw8g);
    CASE(gOIhw8i16o2i);
    CASE(gOIhw8i8o);
    CASE(gOIhw8o16i2o);
    CASE(gIOhw8o16i2o);
    CASE(gOIhw8o8i);
    CASE(OIhw4o8i8o4i);
    CASE(OIhw2o8i8o2i);
    CASE(gOIhw4o8i8o4i);
    CASE(gOIhw2o8i8o2i);
    CASE(gIOdhw16i16o);
    CASE(gOdhwi16o);
    CASE(gOdhwi4o);
    CASE(gOdhwi8o);
    CASE(gOIdhw16i16o);
    CASE(gOIdhw4i16o4i);
    CASE(gOIdhw16o16i);
    CASE(gOidhw16o);
    CASE(gOIdhw4i4o);
    CASE(gOIdhw4o4i);
    CASE(gOidhw4o);
    CASE(gOIdhw8i16o2i);
    CASE(gOIdhw8i8o);
    CASE(gOIdhw8o16i2o);
    CASE(gIOdhw8o16i2o);
    CASE(gOIdhw8o8i);
    CASE(Goidhw16g);
#undef CASE
    if (!strcmp("undef", str) || !strcmp("dnnl_format_tag_undef", str))
        return dnnl_format_tag_undef;
    if (!strcmp("any", str) || !strcmp("dnnl_format_tag_any", str))
        return dnnl_format_tag_any;
    if (!strcmp("last", str) || !strcmp("dnnl_format_tag_last", str))
        return dnnl_format_tag_last;
    assert(!"unknown fmt_tag");
    return dnnl_format_tag_undef;
}

const char *status2str(dnnl_status_t status) {
    return dnnl_status2str(status);
}

const char *dt2str(dnnl_data_type_t dt) {
    return dnnl_dt2str(dt);
}

const char *fmt_tag2str(dnnl_format_tag_t tag) {
    return dnnl_fmt_tag2str(tag);
}
