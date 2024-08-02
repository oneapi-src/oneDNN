/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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


#ifdef _MSC_VER
#pragma warning (disable: 4661)     /* missing definition in template instatiation */
#endif

REG_GEN9_ISA(template class BLASKernelGenerator<HW::Gen9>);
REG_GEN11_ISA(template class BLASKernelGenerator<HW::Gen11>);
REG_XELP_ISA(template class BLASKernelGenerator<HW::Gen12LP>);
REG_XEHP_ISA(template class BLASKernelGenerator<HW::XeHP>);
REG_XEHPG_ISA(template class BLASKernelGenerator<HW::XeHPG>);
REG_XEHPC_ISA(template class BLASKernelGenerator<HW::XeHPC>);
REG_XE2_ISA(template class BLASKernelGenerator<HW::Xe2>);
