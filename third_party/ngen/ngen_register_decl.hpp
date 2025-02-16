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

/*
 * When compiling nGEN in C++11 or C++14 mode, this header file should be
 *  #include'd exactly once in your source code.
 */

#if (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS)

#include "ngen.hpp"

#define NGEN_REGISTER_DECL_MAIN(CG, PREFIX) \
PREFIX constexpr NGEN_NAMESPACE::IndirectRegisterFrame CG::indirect; \
\
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r0; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r1; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r2; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r3; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r4; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r5; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r6; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r7; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r8; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r9; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r10; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r11; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r12; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r13; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r14; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r15; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r16; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r17; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r18; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r19; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r20; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r21; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r22; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r23; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r24; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r25; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r26; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r27; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r28; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r29; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r30; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r31; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r32; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r33; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r34; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r35; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r36; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r37; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r38; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r39; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r40; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r41; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r42; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r43; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r44; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r45; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r46; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r47; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r48; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r49; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r50; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r51; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r52; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r53; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r54; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r55; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r56; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r57; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r58; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r59; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r60; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r61; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r62; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r63; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r64; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r65; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r66; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r67; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r68; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r69; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r70; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r71; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r72; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r73; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r74; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r75; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r76; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r77; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r78; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r79; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r80; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r81; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r82; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r83; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r84; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r85; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r86; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r87; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r88; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r89; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r90; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r91; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r92; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r93; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r94; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r95; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r96; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r97; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r98; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r99; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r100; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r101; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r102; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r103; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r104; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r105; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r106; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r107; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r108; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r109; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r110; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r111; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r112; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r113; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r114; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r115; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r116; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r117; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r118; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r119; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r120; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r121; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r122; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r123; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r124; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r125; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r126; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r127; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r128; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r129; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r130; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r131; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r132; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r133; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r134; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r135; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r136; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r137; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r138; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r139; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r140; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r141; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r142; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r143; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r144; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r145; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r146; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r147; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r148; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r149; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r150; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r151; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r152; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r153; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r154; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r155; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r156; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r157; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r158; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r159; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r160; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r161; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r162; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r163; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r164; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r165; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r166; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r167; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r168; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r169; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r170; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r171; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r172; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r173; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r174; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r175; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r176; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r177; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r178; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r179; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r180; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r181; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r182; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r183; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r184; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r185; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r186; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r187; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r188; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r189; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r190; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r191; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r192; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r193; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r194; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r195; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r196; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r197; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r198; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r199; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r200; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r201; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r202; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r203; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r204; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r205; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r206; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r207; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r208; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r209; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r210; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r211; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r212; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r213; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r214; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r215; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r216; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r217; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r218; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r219; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r220; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r221; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r222; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r223; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r224; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r225; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r226; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r227; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r228; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r229; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r230; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r231; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r232; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r233; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r234; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r235; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r236; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r237; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r238; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r239; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r240; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r241; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r242; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r243; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r244; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r245; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r246; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r247; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r248; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r249; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r250; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r251; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r252; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r253; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r254; \
PREFIX constexpr NGEN_NAMESPACE::GRF CG::r255; \
\
PREFIX constexpr NGEN_NAMESPACE::NullRegister CG::null; \
PREFIX constexpr NGEN_NAMESPACE::AddressRegister CG::a0; \
PREFIX constexpr NGEN_NAMESPACE::AccumulatorRegister CG::acc0; \
PREFIX constexpr NGEN_NAMESPACE::AccumulatorRegister CG::acc1; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc2; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc3; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc4; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc5; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc6; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc7; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc8; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::acc9; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme0; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme1; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme2; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme3; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme4; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme5; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme6; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::mme7; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::nomme; \
PREFIX constexpr NGEN_NAMESPACE::SpecialAccumulatorRegister CG::noacc; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f2; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f3; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0_0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f0_1; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1_0; \
PREFIX constexpr NGEN_NAMESPACE::FlagRegister CG::f1_1; \
PREFIX constexpr NGEN_NAMESPACE::ChannelEnableRegister CG::ce0; \
PREFIX constexpr NGEN_NAMESPACE::StackPointerRegister CG::sp; \
PREFIX constexpr NGEN_NAMESPACE::StateRegister CG::sr0; \
PREFIX constexpr NGEN_NAMESPACE::StateRegister CG::sr1; \
PREFIX constexpr NGEN_NAMESPACE::ControlRegister CG::cr0; \
PREFIX constexpr NGEN_NAMESPACE::NotificationRegister CG::n0; \
PREFIX constexpr NGEN_NAMESPACE::InstructionPointerRegister CG::ip; \
PREFIX constexpr NGEN_NAMESPACE::ThreadDependencyRegister CG::tdr0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm1; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm2; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm3; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tm4; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::pm0; \
PREFIX constexpr NGEN_NAMESPACE::PerformanceRegister CG::tp0; \
PREFIX constexpr NGEN_NAMESPACE::DebugRegister CG::dbg0; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc0; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc1; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc2; \
PREFIX constexpr NGEN_NAMESPACE::FlowControlRegister CG::fc3; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoDDClr; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoDDChk; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::AccWrEn; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoSrcDepSet; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Breakpoint; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::sat; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoMask; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ExBSO; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::AutoSWSB; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Serialize; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::EOT; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Align1; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Align16; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Atomic; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::Switch; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::NoPreempt; \
\
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::anyv; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::allv; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any2h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all2h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any4h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all4h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any8h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all8h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any16h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all16h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any32h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all32h; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::any; \
PREFIX constexpr NGEN_NAMESPACE::PredCtrl CG::all; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::x_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::y_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::z_repl; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::w_repl; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ze; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::eq; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::nz; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ne; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::gt; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ge; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::lt; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::le; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::ov; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::un; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::eo; \
\
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M0; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M4; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M8; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M12; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M16; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M20; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M24; \
PREFIX constexpr NGEN_NAMESPACE::InstructionModifier CG::M28; \
\
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb0; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb1; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb2; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb3; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb4; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb5; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb6; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb7; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb8; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb9; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb10; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb11; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb12; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb13; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb14; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb15; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb16; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb17; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb18; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb19; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb20; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb21; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb22; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb23; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb24; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb25; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb26; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb27; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb28; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb29; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb30; \
PREFIX constexpr NGEN_NAMESPACE::SBID CG::sb31; \
PREFIX constexpr NGEN_NAMESPACE::SWSBInfo CG::NoAccSBSet; \
\
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A32; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A32NC; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A64; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::A64NC; \
PREFIX constexpr NGEN_NAMESPACE::AddressBase CG::SLM; \
\
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D64; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8U32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16U32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D64T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D8U32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::D16U32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V1; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V2; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V3; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V4; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V8; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V16; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V32; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V64; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V1T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V2T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V3T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V4T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V8T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V16T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V32T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::V64T; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::transpose; \
PREFIX constexpr NGEN_NAMESPACE::DataSpecLSC CG::vnni; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1IAR_L3C; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WT_L3UC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WT_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1S_L3WB; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1WB_L3WB;

#ifndef NGEN_SHORT_NAMES
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX)
#else
#define NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX) \
PREFIX constexpr const NGEN_NAMESPACE::IndirectRegisterFrame &CG::r; \
PREFIX constexpr const NGEN_NAMESPACE::InstructionModifier &CG::W;
#endif

#define NGEN_REGISTER_DECL_EXTRA2A(CG,PREFIX) \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1C_L3CC; \
PREFIX constexpr NGEN_NAMESPACE::CacheSettingsLSC CG::L1UC_L3CC;

#define NGEN_REGISTER_DECL_EXTRA2(CG,PREFIX) \
PREFIX constexpr NGEN_NAMESPACE::ScalarRegister CG::s0;

#define NGEN_REGISTER_DECL_EXTRA3(CG,PREFIX)

#define NGEN_REGISTER_DECL_EXTRA4(CG,PREFIX)

#define NGEN_REGISTER_DECL(CG,PREFIX) \
NGEN_REGISTER_DECL_MAIN(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA1(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA2A(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA2(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA3(CG,PREFIX) \
NGEN_REGISTER_DECL_EXTRA4(CG,PREFIX)

#include "ngen.hpp"
NGEN_REGISTER_DECL(NGEN_NAMESPACE::BinaryCodeGenerator<hw>, template <NGEN_NAMESPACE::HW hw>)

#ifdef NGEN_ASM
#include "ngen_asm.hpp"
NGEN_REGISTER_DECL(NGEN_NAMESPACE::AsmCodeGenerator, /* nothing */)
#endif

template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Unknown>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen9>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen10>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen11>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Gen12LP>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHP>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHPG>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::XeHPC>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Xe2>;
template class NGEN_NAMESPACE::BinaryCodeGenerator<NGEN_NAMESPACE::HW::Xe3>;

#endif /* (defined(NGEN_CPP11) || defined(NGEN_CPP14)) && !defined(NGEN_GLOBAL_REGS) */
