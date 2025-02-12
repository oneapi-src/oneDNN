/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

// clang-format off
const char** get_plan_registry_entries() {
    static const char *entries[] = {
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64oc64ow32 2d=1 ext=out_b2,out_b4,stream_k model=015DBA59441308D13E08C8673FEB49D441F6FFA74102487A344669B2AD3F030000E345DA4B333C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64oc32ow8 tg=oc2ow4 2d=1 prefetch=x1 ext=out_b2,out_b4,stream_k model=0179BFD54311B8EE3EFDCF963F65618D3F5EA3713C0298A0D54590B01740030000DB458B60333C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64oc64ow16 tg=ic8 2d=1 ext=out_b2,out_b4,stream_k model=014799AD44FF6F923F2B9A4F3E338BD63FFD8F9A3F0240EB9C45D3A535410300002345B20F263C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64oc64ow16 tg=ow4 2d=1 prefetch=x1 ext=out_b2,out_b4,stream_k model=010EF322441180EA3EFC278A3F4B95AB3FFB450640027BB2F6450FBD0B40030080D9458B83313C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64mb32oc64 tg=ow2 2d=1 ext=out_b2,out_b4,stream_k model=017E567644124CEC3E02145D3F80570D40F6FFA7410260586246FD5E15400300002E450093373C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64mb16oc64 tg=ic8 2d=1 ext=out_b2,out_b4,stream_k model=0105CFA64400408B3F2B1A573E64EA9A3FFEBF923F0200B88C4566793A41030000374570922B3C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64mb16oc64 tg=ow16 2d=1 prefetch=x1 ext=out_b2,out_b4,stream_k model=01E2332E441794B83EFE5F933F62B09B3F08E0403F02D2F42D462A151641030000254522FD363C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic64mb8oc64 tg=ow8 2d=1 prefetch=x1 ext=out_b2,out_b4,stream_k model=019F6E0B44111CDB3EFD77993F9D358A3F29DA553E020E774346C90B454003000023455EC8363C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic32oc16ow8 ext=out_b2,out_b4,stream_k model=016A335745FFC7B23F02FC5B3F0000803F34005B3E02603B91464538E73F030000224550B8353C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic32oc16ow8 tg=ic8 ext=out_b2,out_b4,stream_k model=017A514445FF378E3F05303C3F66B6833F1300D93E02803449468A9D9441030080DB45AB8B2D3C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic32oc16ow16 ext=out_b2,out_b4,stream_k model=0131847B4514E8BE3E0130823F0000803F6494C23D02C76B3847DAB616400300002745CAEC353C",
        "hw=xehpc prop=fwd src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=dpas simd=16 regs=256 iter=ic32oc16ow8 tg=ic2 ext=out_b2,out_b4,stream_k model=01B9145345FE67A43F03D0553F0000803F1A809F3E02C8290547D9EF6C40030000274501E8343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64oc16ow8 tg=oc2ow2 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=01D1461744037C4A3FFEA7A03F357F863F17B7D13802CF9B46461C49C33F030000274536F6353C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32oc64ow32 tg=oc2ow4 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=01E63AE14332D43B3E05C6553FFBAE8040F6FFA741020739CB46C2B0AB400300002E45E7FC353C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32oc32ow32 tg=oc2 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=01A33207441218DF3E0470293F34311540F7FF9D41028CC8F9450FF3AB3F03000036451252343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64oc32ow16 tg=oc2ow4 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=0163D639440458323FFFAB833FCDE1A23FFD1F9C3F02B5F0E64542169440030000354597CF353C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32mb16oc64 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=019489524403684F3F0316473F0000803FF5FFBB41024C40E9458D42853F030080EA4505D0343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64mb8oc32 tg=ow8 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=01B74D2B44062C0E3F01E0863F654AA73FCB68703D02DC9F8246153C4E400300002645AB14353C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32mb32oc64 tg=mb2 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=0157EB2F441670963E05A8413FA5F0EB40F6FFA74102775383460C6DF63F030000EF4503AF343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64mb8oc64 tg=ow8 2d=1 prefetch=x1 ext=out_b1,out_b4,stream_k model=01978B6644021E303F04B07F3F6764813F1440F33E028A550F469C2AA440030000324520ED343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc32ow16 tg=ic8 ext=out_b1,out_b4,stream_k model=01CA9A2F4501B86D3F009C8E3F98F1913F1000ED3E023BA4A046A3AF5241030080DF45D5B5303C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc32ow16 ext=out_b1,out_b4,stream_k model=0173AD02451380EA3E0308503FCE6C873FFD6FA13F024EDF6946FF70A73F030080DC456E31343C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc16ow8 tg=ic8 ext=out_b1,out_b4,stream_k model=01B8EADA440012823FFEBFAB3F0000803FF100083D0259EB49471741E5400300002545F7C7313C",
        "hw=xehpc prop=fwd src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc32ow16 tg=ic4 ext=out_b1,out_b4,stream_k model=017AC7234501904A3F0048833F996F893FD268163D020641CD466BD8B540030000E245E12C323C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic8oc32ow16 tg=ow2 2d=1 ext=out_b1,out_b2,stream_k model=01B7DE43447330D73D01B8633F980ACF3FF4FFCF4102CDBA0B4616FD223F0300000D45E5392F3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16oc32ow8 tg=ow4 2d=1 prefetch=x1 ext=out_b1,out_b2,stream_k model=013E95854408D0193F07E83D3F983F833F04003A3F0200500F465403A63F0300000445233B2F3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16oc32ow8 tg=oc2 2d=1 ext=out_b1,out_b2,stream_k model=0198618B440038613F0500303F0041843F0180553F028D048F45F4F96D3F030000024508612D3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16oc32ow8 tg=ic4 2d=1 ext=out_b1,out_b2,stream_k model=01908AC244F927EE3F08300A3F0000803F17B7D1380219F5CC45DB1612400300000845AEC72E3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16mb8oc32 tg=ic2 2d=1 ext=out_b1,out_b2,stream_k model=012187B144FD6FC43F04BC233F0000803F33806C3E028009994584B7843F030080E94540E02C3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16mb8oc32 tg=oc2 2d=1 ext=out_b1,out_b2,stream_k model=0170A29644FDFFA23F04F8283F6A3B803F0680143F0268F1AB455A05523F030000F1454B242C3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16mb8oc32 tg=ic4 2d=1 ext=out_b1,out_b2,stream_k model=012108BF44FBFFBB3F04602A3F0000803F36801C3E0200121F46E73A0940030080E645A08B2C3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic16mb8oc32 tg=ow8 2d=1 prefetch=x1 ext=out_b1,out_b2,stream_k model=01D7C88D440788203F04C83A3F9A05843F0514263F02D9424E4609C32B40030000E6458C802D3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic8oc32ow8 ext=out_b1,out_b2,stream_k model=013EBEAE440B00083FFEDF9F3F0000803F0628533F0200D8E345FFFFCB3E03000003459F1B2D3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic8oc32ow8 tg=ic16 ext=out_b1,out_b2,stream_k model=013D23DC4402783A3FFFB79F3FCCACD13F2E1A073E020C57CA46A9D42A4103000023454C062C3C",
        "hw=xehpc prop=fwd src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic8oc32ow8 tg=ic4 ext=out_b1,out_b2,stream_k model=01A9B1B44406D01E3FFFB78B3F9A41BD3F1440A83E02D8D4964611F1D73F030000244516BF2D3C",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32iw32oc32 tg=ic2iw2 2d=1 prefetch=x1 spec=sd1sh1sw1 model=0167A6114414E0E93EFFEF6C3F68A7DC3FF6FFA741",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32iw8oc64 tg=iw4 2d=1 prefetch=x1 spec=sd1sh1sw1 model=0193592A4402C85D3FFEAF7F3F67BEC23F05002B3F",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64iw32oc32 tg=ic2 2d=1 prefetch=x1 spec=sd1sh1sw1 model=01174837441504CB3E0492433F73FEA142F6FFA741",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32iw16oc64 tg=ic2iw4 2d=1 prefetch=x1 spec=sd1sh1sw1 model=017ABA3D44024C393F01C8803F01B69F3FFB9FA83F",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64mb32oc32 tg=iw2 2d=1 spec=sd1sh1sw1 model=013C0B624413D8DD3E043C3F3FD89E1541F6FFA741",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32mb8oc64 tg=iw8 2d=1 prefetch=x1 spec=sd1sh1sw1 model=01836E274405841A3FFF878E3FFDE1943FA9D1F03C",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64mb16oc32 tg=iw16 2d=1 prefetch=x1 spec=sd1sh1sw1 model=019FEAEF433150743E02986A3F32EBC53FF93FA340",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic64mb8oc64 tg=iw8 2d=1 prefetch=x1 spec=sd1sh1sw1 model=010148534404D0283F04A6753F343B8E3F10D0CC3E",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16mb8oc32 tg=ic2iw4 spec=sd1sh1sw1 model=01FD3B1B4503A0533F0110613F679E803F2A1A343E",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic32iw16oc16 spec=sd1sh1sw1 model=01DFD9FE441250F73E0500443F9AFB8A3FFE3FAE3F",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16iw8oc32 spec=sd1sh1sw1 model=014F1D174501A8643F00406D3F0000803F2A1A393E",
        "hw=xehpc prop=bwd_d src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16iw16oc16 spec=sd1sh1sw1 model=016778C644065C293F0008693F0000803F06800F3F",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw16oc8 2d=1 spec=sd1sh1sw1 ext=stream_k model=01F6655C441858813E06544F3FCE92B13FF6FFA741028C1AED4510A3AE3E030000E2459E802D3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw8oc16 tg=iw4 2d=1 prefetch=x1 spec=sd1sh1sw1 ext=stream_k model=01794F884403A04E3F06C83A3F34D1883FFFDF5E3F0266280546C57FA43F030080DD45E0C32B3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw8oc16 tg=ic2 2d=1 prefetch=x1 spec=sd1sh1sw1 ext=stream_k model=01BAC8A944FB2FD73F0300623FCB2B833F30C05E3E020D1951457BB9923F030000E04593BD2C3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw8oc16 tg=iw2 2d=1 prefetch=x1 spec=sd1sh1sw1 ext=stream_k model=01EE598A440020743F0388393F322D843FFB7F783F027283DC45A184363F030080E14505962D3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32mb16oc8 2d=1 spec=sd1sh1sw1 ext=stream_k model=014CF8824414B0AB3E04085A3FFED38A3FFF8FAE3F02799B06460502AF3E030000E445011D2C3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32mb8oc16 2d=1 prefetch=x1 spec=sd1sh1sw1 ext=stream_k model=010D32AB4402764B3F00C8853F0000803F0630283F02CF7C464572B5133F030000E64566832D3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32mb8oc16 tg=iw4 2d=1 spec=sd1sh1sw1 ext=stream_k model=0137529744FE4F943F04E8383F0000803F0400083F0234E0CF457EE5C53F030080DF4588072D3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32mb8oc16 tg=iw8 2d=1 prefetch=x1 spec=sd1sh1sw1 ext=stream_k model=0138AC8F4405742F3F02A8413F9979823F07C01A3F02D9C338464F032B40030080EB45C1372E3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32mb8oc8 tg=ic2mb2 spec=sd1sh1sw1 ext=stream_k model=01ABB1B4440890133F00588C3FCEEE8F3F15C0823E02570432468130CB3F03000020451C9D2F3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw8oc8 tg=ic2 spec=sd1sh1sw1 ext=stream_k model=0191F8984417509D3E009E643F98E0933FFBE7EC3F02E86A124694BE443F0300001245A2EE2C3C",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw16oc8 spec=sd1sh1sw1 model=01953EB3447120D83D01CC5E3FCC889B3FF9078640",
        "hw=xehpc prop=bwd_d src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=ic32iw8oc8 spec=sd1sh1sw1 ext=stream_k model=01BEAAA7441560F13E00106B3FCB65823F0520603F02F392FE454F03CF3E0300001B45AD212F3C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8oc64ow16 tg=ic4 2d=1 prefetch=x1 ext=out_b4,bias,stream_k model=01F63F7A4331C0683E0340313FCDB821408186453F0208D50546E2F3503F030000FE44D67EFD3B",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8oc32ow64 tg=ic8oc2 2d=1 prefetch=x1 ext=out_b4,bias,stream_k model=01B4F3D7433200563E0440403FCCB41040C780B14002AC04BD456ABBC040030000FE44F86A083C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc64ow16 tg=ic8 2d=1 prefetch=x3 ext=out_b4,bias,stream_k model=0106E66C437380D73D04F85A3F9A09B23FF6FFA74102312420462DCECB3F0300000745E7E5093C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8oc32ow64 tg=ic8 2d=1 prefetch=x1 ext=out_b4,bias,stream_k model=01125DCB4375E0B33D01B8633F9AD9B140F6FFA74102F3B0B7459F656A40030000FE4494940B3C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16mb16oc64 tg=ic4 2d=1 prefetch=x1 ext=out_b4,bias,stream_k model=014EA99443F6FF1942F6FFA74165C6C73FF6FFA74102C324A2468988393F030000FE440782E03B",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16mb16oc16 2d=1 prefetch=x3 ext=out_b4,bias,stream_k model=012D706343F283443C05CE2F3FA5EAA142F6FFA74102FF2D274558F36B3F0300004245EF86F63B",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8mb64oc32 tg=ic4oc2 2d=1 prefetch=x1 ext=out_b4,bias,stream_k model=01284AE243EF01D43CFF7F963F3533CF3FF6FFA74102B3FC7E45926A87400300000345FB87EB3B",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16mb16oc32 tg=ic2 2d=1 prefetch=x3 ext=out_b4,bias,stream_k model=01E5348143F7FF0F42F6FFA741FEF5D441F6FFA74102208DC445D97B6C3F03000013459DCCF93B",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8oc16ow16 tg=ic8 ext=out_b4,bias,stream_k model=019D4C9C44F000673D0178583F66D2B43FF6FFA741029A73CE456FCAD540030000614507AE003C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic16oc32ow16 tg=ic2 ext=out_b4,bias,stream_k model=018EA5FB447400C13D00C0793F31D3DF3FF6FFA74102FF975446712F414003000028456F49103C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8mb16oc16 tg=oc2 ext=out_b4,bias,stream_k model=01CD95AA44F3803C3DFF1F883F330BA53FF9FFFC3F024E2213467787D23F0300006145C55A0B3C",
        "hw=xehpc prop=bwd_w src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=dpas simd=16 regs=256 iter=ic8oc16ow32 ext=out_b4,bias,stream_k model=0107A0EB4471C0B03D0290543FCB44C03FF6FFA74102C1184046AD31BD3F0300002745D1F9093C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8oc16ow16 tg=ic2oc2 2d=1 prefetch=x1 ext=bias,stream_k model=01496D9E43F281DB3C0128623FFDBF7B40F6FFA741028077C8456740353F0300002B45BA9BE43B",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8oc16ow64 tg=ic4oc2 2d=1 prefetch=x1 ext=bias,stream_k model=01E29F5344EE21FF3C0218543FFDDFB041F6FFA74102805197450FA4D54003000013457FF3033C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic16oc16ow8 tg=oc4 2d=1 prefetch=x1 ext=bias,stream_k model=01DF909B43F500033DFE6F923F66F65041F6FFA741022D5D0B46D955203F0300000D4578B6DF3B",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8oc16ow32 tg=ic4 2d=1 prefetch=x1 ext=bias,stream_k model=01F23CEE43F541B73C02D8523FF85FAF40F6FFA7410266990D446C4E09400300002445FBDA013C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8mb64oc16 tg=ic2oc4 2d=1 prefetch=x1 ext=bias,stream_k model=012ED67144F4A1BB3C01B86D3F3551A142F6FFA7410200001844E3AB4F4103000016456498003C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8mb16oc16 tg=ic2 2d=1 prefetch=x1 ext=bias,stream_k model=0189A1AF43F521AF3C0138613F66FA9E40F6FFA7410299DA7D45A68B0C3F03000023454CE8FE3B",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8mb8oc32 tg=ic4oc4 2d=1 prefetch=x1 ext=bias,stream_k model=01CB64BE43F283623CFF3B8A3FCCD43540F6FFA741020D6B004639992E400300002445587EFD3B",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8mb8oc16 tg=ic2oc2 2d=1 prefetch=x3 ext=bias,stream_k model=01FA188543F183713C00EC843F00180340F6FFA741028D8034464B26C73E0300002C45FA2C073C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8oc16ow16 tg=ic8 ext=bias,stream_k model=017EFD2B44EF01DE3C0318543F3551A142F6FFA74102666E5D46299C57400300001B451B54013C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8mb16oc16 tg=ic4oc2 ext=bias,stream_k model=01D36F4044F880193D01F0713F33230E41F6FFA74102AFB4D24654CD53400300005445485DFF3B",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic16oc16ow8 ext=bias,stream_k model=01201A23447380D73D0200493FFE3F8A40F6FF2A410274EE0E460CDDB53E0300001145FC4A043C",
        "hw=xehpc prop=bwd_w src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=16 regs=128 iter=ic8oc16ow16 tg=oc4 ext=bias,stream_k model=015F373144F421B43C02486A3F65964441F6FFA74102ED9B63463F4CDA3F0300006E456180033C",
        "hw=xehpc prop=fwd dw=1 src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=mad simd=32 regs=128 iter=g32mb8 tg=ow2 spec=ic1oc1 ext=out_b2,out_b4,stream_k model=01AB61AB45000A803F0000803F99E1CD3F7000DA3D02CC119C45AEA3FC3F0353E99A479987F33F",
        "hw=xehpc prop=fwd dw=1 src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=mad simd=32 regs=128 iter=g32ow8 tg=ow16 spec=ic1oc1 ext=out_b2,out_b4,stream_k model=01D8FBA5450000803F0000803F9CC9CA3F1500CF3E027A75A44517ED7B4103681979474FEDE33F",
        "hw=xehpc prop=fwd dw=1 src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1 ext=out_b2,out_b4,stream_k model=011791AE450000803F0000803FCD44A33F7300C63D024A68954500D0713F0397BC7647191E0A40",
        "hw=xehpc prop=fwd dw=1 src=axb:s8 wei=axcb:s8 dst=axb:s8 fma=mad simd=32 regs=128 iter=g32ow8 spec=ic1oc1 ext=out_b2,out_b4,stream_k model=01F52BAB450000803F0000803F335BB33F7100EE3D02FE7CA645A785733F031E8A4B4782F41240",
        "hw=xehpc prop=fwd dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32ow8 tg=mb2 spec=ic1oc1 ext=out_b1,out_b4,stream_k model=01A58618450000803F0000803FC98CCF40F6FFA741029F4EAB45ECC9144003A37A5C47FF2D0840",
        "hw=xehpc prop=fwd dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1 ext=out_b1,out_b4,stream_k model=01E45E16450000803F0000803F01006340F6FFA74102A78E9F452DBF893F037EF083471129E53F",
        "hw=xehpc prop=fwd dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32mb8 tg=mb2 spec=ic1oc1 ext=out_b1,out_b4,stream_k model=018A6D16450000803F0000803F5DC09F40F6FFA74102004D9C455D460E40038EAC7E47F43E0640",
        "hw=xehpc prop=fwd dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32ow8 spec=ic1oc1 ext=out_b1,out_b4,stream_k model=01521D17450000803F0000803F67C687400F00F23E028037B04586C9893F0301973A47CB8B0740",
        "hw=xehpc prop=fwd dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32ow8 tg=mb2 spec=ic1oc1 ext=out_b1,out_b2,stream_k model=01BE89E3440000803F0000803F340B08400200623F02C07AAB45FCDA0340037FA64E470794F03F",
        "hw=xehpc prop=fwd dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32ow8 spec=ic1oc1 ext=out_b1,out_b2,stream_k model=01DA38DB440020833F0000803F34D34240F33F5C400220E7A8459A9F723F038636F646E63E1040",
        "hw=xehpc prop=fwd dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1 ext=out_b1,out_b2,stream_k model=010B36E144004A813F0000803F344F004006B0023F0282889B456C73733F03CC968F47FA09B53F",
        "hw=xehpc prop=fwd dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32ow8 tg=ow2 spec=ic1oc1 ext=out_b1,out_b2,stream_k model=0133D7DA440000803F0000803FFF2F06400700173F02F02BAB450498FD3F0351614B47AC29F13F",
        "hw=xehpc prop=bwd_d dw=1 src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1sd1sh1sw1 model=012C8315450000803F0000803F98397E40F6FFA741",
        "hw=xehpc prop=bwd_d dw=1 src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32iw8 tg=iw2 spec=ic1oc1sd1sh1sw1 model=01CA8111450000803F0000803F66E6AC40F6FFA741",
        "hw=xehpc prop=bwd_d dw=1 src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32iw8 tg=iw8 spec=ic1oc1sd1sh1sw1 model=01A61516450000803F0000803FFECF9C40F6FFA741",
        "hw=xehpc prop=bwd_d dw=1 src=axb:bf16 wei=axbc:bf16 dst=axb:bf16 fma=mad simd=32 regs=128 iter=g32iw8 spec=ic1oc1sd1sh1sw1 model=011C5C15450000803F0000803FFF4F2B40FCFF9D3F",
        "hw=xehpc prop=bwd_d dw=1 src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32iw8 tg=iw8 spec=ic1oc1sd1sh1sw1 ext=stream_k model=01D60AD5440040813F0000803F008C0140FA7FD73F02287D9D4510DFEF400347A76A47458AAD3F",
        "hw=xehpc prop=bwd_d dw=1 src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1sd1sh1sw1 ext=stream_k model=017612E4440000803F0000803F9855CC3F05E0223F02DE549A450B00733F03F4B42D47B2CC0740",
        "hw=xehpc prop=bwd_d dw=1 src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32iw8 spec=ic1oc1sd1sh1sw1 ext=stream_k model=01E5A2D7440000803F0000803F996BF83FFA7F0D4002DBBD9E45E84F653F035FDA2747DFDBCB3F",
        "hw=xehpc prop=bwd_d dw=1 src=axb:f32 wei=axbc:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32iw8 tg=iw2 spec=ic1oc1sd1sh1sw1 ext=stream_k model=0193B8C8440050803F0000803F00722740FAFFD43F0210F5A045A71EED3F030B597647AB35AB3F",
        "hw=xehpc prop=bwd_w dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=16 regs=128 iter=g16mb8 spec=ic1oc1 ext=out_b4,bias,stream_k model=01A7516F44F6FFA7411400C03E00408A400000803F026564974649F0A83E030000FE44F5A6973E",
        "hw=xehpc prop=bwd_w dw=1 src=axb:bf16 wei=axcb:bf16 dst=axb:bf16 fma=mad simd=16 regs=128 iter=g16ow8 spec=ic1oc1 ext=out_b4,bias,stream_k model=011E1E5944F3804B3D0220603FCD8C93420000803F0298B3224636B9AD3E0300000045AA47863E",
        "hw=xehpc prop=bwd_w dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32mb8 spec=ic1oc1 ext=bias,stream_k model=013A4E5E440700263F03005D3F666626420000803F02BC42884682C3883F030000FE44D2A9B93E",
        "hw=xehpc prop=bwd_w dw=1 src=axb:f32 wei=axcb:f32 dst=axb:f32 fma=mad simd=32 regs=128 iter=g32ow8 spec=ic1oc1 ext=bias,stream_k model=015B0B4644F480643D014C613FCCCC61420000803F02108F5246FC299A3F030000FE44C863B53E",
        nullptr,
    };
    return entries;
}
// clang-format on

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
