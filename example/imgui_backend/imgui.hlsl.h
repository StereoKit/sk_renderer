#pragma once

/*
 ________________
|--Performance--
| Instructions |  all | tex | flow |
|       Vertex |   29 |   0 |    0 |
|        Pixel |   14 |   1 |    0 |
|--Buffer Info--
|  $Global - 64 bytes
|    projection_matrix: +0       64b - float4x4
|--Mesh Input--
|  float2 : Position0
|  float2 : TexCoord0
|  float4 : Color0
|--Vertex Shader--
|  b0/s0  : $Global
|--Pixel Shader--
|  t100   : texture0
|________________
*/

/*
================================================================================
 VERTEX SHADER SPIRV
================================================================================
; SPIR-V
; Version: 1.3
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 47
; Schema: 0
               OpCapability Shader
               OpExtension "SPV_GOOGLE_hlsl_functionality1"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %vs "vs" %input_pos %input_uv %input_col %_entryPointOutput_pos %_entryPointOutput_uv %_entryPointOutput_col
               OpSource HLSL 500
               OpName %vs "vs"
               OpName %_Global "$Global"
               OpMemberName %_Global 0 "projection_matrix"
               OpName %_ ""
               OpName %input_pos "input.pos"
               OpName %input_uv "input.uv"
               OpName %input_col "input.col"
               OpName %_entryPointOutput_pos "@entryPointOutput.pos"
               OpName %_entryPointOutput_uv "@entryPointOutput.uv"
               OpName %_entryPointOutput_col "@entryPointOutput.col"
               OpDecorate %_Global Block
               OpMemberDecorate %_Global 0 RowMajor
               OpMemberDecorate %_Global 0 MatrixStride 16
               OpMemberDecorate %_Global 0 Offset 0
               OpDecorate %_ Binding 0
               OpDecorate %_ DescriptorSet 0
               OpDecorate %input_pos Location 0
               OpDecorateString %input_pos UserSemantic "POSITION"
               OpDecorate %input_uv Location 1
               OpDecorateString %input_uv UserSemantic "TEXCOORD0"
               OpDecorate %input_col Location 2
               OpDecorateString %input_col UserSemantic "COLOR0"
               OpDecorate %_entryPointOutput_pos BuiltIn Position
               OpDecorateString %_entryPointOutput_pos UserSemantic "SV_POSITION"
               OpDecorate %_entryPointOutput_uv Location 0
               OpDecorateString %_entryPointOutput_uv UserSemantic "TEXCOORD0"
               OpDecorate %_entryPointOutput_col Location 1
               OpDecorateString %_entryPointOutput_col UserSemantic "COLOR0"
       %void = OpTypeVoid
         %12 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %v2float = OpTypeVector %float 2
    %v4float = OpTypeVector %float 4
%float_2_20000005 = OpConstant %float 2.20000005
         %18 = OpConstantComposite %v3float %float_2_20000005 %float_2_20000005 %float_2_20000005
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%mat4v4float = OpTypeMatrix %v4float 4
    %_Global = OpTypeStruct %mat4v4float
%_ptr_Uniform__Global = OpTypePointer Uniform %_Global
          %_ = OpVariable %_ptr_Uniform__Global Uniform
%_ptr_Uniform_mat4v4float = OpTypePointer Uniform %mat4v4float
%_ptr_Input_v2float = OpTypePointer Input %v2float
  %input_pos = OpVariable %_ptr_Input_v2float Input
   %input_uv = OpVariable %_ptr_Input_v2float Input
%_ptr_Input_v4float = OpTypePointer Input %v4float
  %input_col = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput_pos = OpVariable %_ptr_Output_v4float Output
%_ptr_Output_v2float = OpTypePointer Output %v2float
%_entryPointOutput_uv = OpVariable %_ptr_Output_v2float Output
%_entryPointOutput_col = OpVariable %_ptr_Output_v4float Output
         %vs = OpFunction %void None %12
         %30 = OpLabel
         %31 = OpLoad %v2float %input_pos
         %32 = OpLoad %v2float %input_uv
         %33 = OpLoad %v4float %input_col
         %34 = OpCompositeExtract %float %31 0
         %35 = OpCompositeExtract %float %31 1
         %36 = OpCompositeConstruct %v4float %34 %35 %float_0 %float_1
         %37 = OpAccessChain %_ptr_Uniform_mat4v4float %_ %int_0
         %38 = OpLoad %mat4v4float %37
         %39 = OpVectorTimesMatrix %v4float %36 %38
         %40 = OpVectorShuffle %v3float %33 %33 0 1 2
         %41 = OpExtInst %v3float %1 Pow %40 %18
         %42 = OpCompositeExtract %float %33 3
         %43 = OpCompositeExtract %float %41 0
         %44 = OpCompositeExtract %float %41 1
         %45 = OpCompositeExtract %float %41 2
         %46 = OpCompositeConstruct %v4float %43 %44 %45 %42
               OpStore %_entryPointOutput_pos %39
               OpStore %_entryPointOutput_uv %32
               OpStore %_entryPointOutput_col %46
               OpReturn
               OpFunctionEnd
*/

/*
================================================================================
 PIXEL SHADER SPIRV
================================================================================
; SPIR-V
; Version: 1.3
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 24
; Schema: 0
               OpCapability Shader
               OpExtension "SPV_GOOGLE_hlsl_functionality1"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %ps "ps" %input_uv %input_col %_entryPointOutput
               OpExecutionMode %ps OriginUpperLeft
               OpSource HLSL 500
               OpName %ps "ps"
               OpName %texture0 "texture0"
               OpName %input_uv "input.uv"
               OpName %input_col "input.col"
               OpName %_entryPointOutput "@entryPointOutput"
               OpDecorate %texture0 Binding 100
               OpDecorate %texture0 DescriptorSet 0
               OpDecorate %input_uv Location 0
               OpDecorateString %input_uv UserSemantic "TEXCOORD0"
               OpDecorate %input_col Location 1
               OpDecorateString %input_col UserSemantic "COLOR0"
               OpDecorate %_entryPointOutput Location 0
               OpDecorateString %_entryPointOutput UserSemantic "SV_TARGET"
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
    %v2float = OpTypeVector %float 2
         %12 = OpTypeImage %float 2D 0 0 0 1 Unknown
         %13 = OpTypeSampledImage %12
%_ptr_UniformConstant_13 = OpTypePointer UniformConstant %13
   %texture0 = OpVariable %_ptr_UniformConstant_13 UniformConstant
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Input_v2float = OpTypePointer Input %v2float
   %input_uv = OpVariable %_ptr_Input_v2float Input
  %input_col = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%_entryPointOutput = OpVariable %_ptr_Output_v4float Output
         %ps = OpFunction %void None %8
         %18 = OpLabel
         %19 = OpLoad %v2float %input_uv
         %20 = OpLoad %v4float %input_col
         %21 = OpLoad %13 %texture0
         %22 = OpImageSampleImplicitLod %v4float %21 %19
         %23 = OpFMul %v4float %20 %22
               OpStore %_entryPointOutput %23
               OpReturn
               OpFunctionEnd
*/

const unsigned char sks_imgui_hlsl[3037] = {83,75,83,72,65,68,69,82,5,0,2,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,3,0,0,0,29,0,0,0,0,0,0,0,0,0,0,0,14,0,0,0,1,0,0,
0,0,0,0,0,36,71,108,111,98,97,108,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,3,64,0,0,0,1,0,0,0,0,0,0,0,112,114,111,106,101,99,116,105,111,110,95,109,
97,116,114,105,120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,102,108,111,97,116,52,120,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,64,0,0,0,4,0,16,0,2,0,0,0,2,1,0,0,0,0,2,0,0,0,2,2,0,0,0,0,2,0,0,
0,4,6,0,0,0,0,116,101,120,116,117,114,101,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,100,0,2,4,0,0,0,0,1,0,0,0,1,0,0,0,8,6,0,0,3,2,35,7,0,3,1,0,
11,0,8,0,47,0,0,0,0,0,0,0,17,0,2,0,1,0,0,0,10,0,9,0,83,80,86,95,71,79,79,71,76,69,
95,104,108,115,108,95,102,117,110,99,116,105,111,110,97,108,105,116,121,49,0,0,11,
0,6,0,1,0,0,0,71,76,83,76,46,115,116,100,46,52,53,48,0,0,0,0,14,0,3,0,0,0,0,0,1,0,
0,0,15,0,10,0,0,0,0,0,2,0,0,0,118,115,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0,
8,0,0,0,3,0,3,0,5,0,0,0,244,1,0,0,5,0,3,0,2,0,0,0,118,115,0,0,5,0,4,0,9,0,0,0,36,
71,108,111,98,97,108,0,6,0,8,0,9,0,0,0,0,0,0,0,112,114,111,106,101,99,116,105,111,
110,95,109,97,116,114,105,120,0,0,0,5,0,3,0,10,0,0,0,0,0,0,0,5,0,5,0,3,0,0,0,105,
110,112,117,116,46,112,111,115,0,0,0,5,0,5,0,4,0,0,0,105,110,112,117,116,46,117,118,
0,0,0,0,5,0,5,0,5,0,0,0,105,110,112,117,116,46,99,111,108,0,0,0,5,0,8,0,6,0,0,0,64,
101,110,116,114,121,80,111,105,110,116,79,117,116,112,117,116,46,112,111,115,0,0,
0,5,0,8,0,7,0,0,0,64,101,110,116,114,121,80,111,105,110,116,79,117,116,112,117,116,
46,117,118,0,0,0,0,5,0,8,0,8,0,0,0,64,101,110,116,114,121,80,111,105,110,116,79,117,
116,112,117,116,46,99,111,108,0,0,0,71,0,3,0,9,0,0,0,2,0,0,0,72,0,4,0,9,0,0,0,0,0,
0,0,4,0,0,0,72,0,5,0,9,0,0,0,0,0,0,0,7,0,0,0,16,0,0,0,72,0,5,0,9,0,0,0,0,0,0,0,35,
0,0,0,0,0,0,0,71,0,4,0,10,0,0,0,33,0,0,0,0,0,0,0,71,0,4,0,10,0,0,0,34,0,0,0,0,0,0,
0,71,0,4,0,3,0,0,0,30,0,0,0,0,0,0,0,0,22,6,0,3,0,0,0,3,22,0,0,80,79,83,73,84,73,79,
78,0,0,0,0,71,0,4,0,4,0,0,0,30,0,0,0,1,0,0,0,0,22,6,0,4,0,0,0,3,22,0,0,84,69,88,67,
79,79,82,68,48,0,0,0,71,0,4,0,5,0,0,0,30,0,0,0,2,0,0,0,0,22,5,0,5,0,0,0,3,22,0,0,
67,79,76,79,82,48,0,0,71,0,4,0,6,0,0,0,11,0,0,0,0,0,0,0,0,22,6,0,6,0,0,0,3,22,0,0,
83,86,95,80,79,83,73,84,73,79,78,0,71,0,4,0,7,0,0,0,30,0,0,0,0,0,0,0,0,22,6,0,7,0,
0,0,3,22,0,0,84,69,88,67,79,79,82,68,48,0,0,0,71,0,4,0,8,0,0,0,30,0,0,0,1,0,0,0,0,
22,5,0,8,0,0,0,3,22,0,0,67,79,76,79,82,48,0,0,19,0,2,0,11,0,0,0,33,0,3,0,12,0,0,0,
11,0,0,0,22,0,3,0,13,0,0,0,32,0,0,0,23,0,4,0,14,0,0,0,13,0,0,0,3,0,0,0,23,0,4,0,15,
0,0,0,13,0,0,0,2,0,0,0,23,0,4,0,16,0,0,0,13,0,0,0,4,0,0,0,43,0,4,0,13,0,0,0,17,0,
0,0,205,204,12,64,44,0,6,0,14,0,0,0,18,0,0,0,17,0,0,0,17,0,0,0,17,0,0,0,21,0,4,0,
19,0,0,0,32,0,0,0,1,0,0,0,43,0,4,0,19,0,0,0,20,0,0,0,0,0,0,0,43,0,4,0,13,0,0,0,21,
0,0,0,0,0,0,0,43,0,4,0,13,0,0,0,22,0,0,0,0,0,128,63,24,0,4,0,23,0,0,0,16,0,0,0,4,
0,0,0,30,0,3,0,9,0,0,0,23,0,0,0,32,0,4,0,24,0,0,0,2,0,0,0,9,0,0,0,59,0,4,0,24,0,0,
0,10,0,0,0,2,0,0,0,32,0,4,0,25,0,0,0,2,0,0,0,23,0,0,0,32,0,4,0,26,0,0,0,1,0,0,0,15,
0,0,0,59,0,4,0,26,0,0,0,3,0,0,0,1,0,0,0,59,0,4,0,26,0,0,0,4,0,0,0,1,0,0,0,32,0,4,
0,27,0,0,0,1,0,0,0,16,0,0,0,59,0,4,0,27,0,0,0,5,0,0,0,1,0,0,0,32,0,4,0,28,0,0,0,3,
0,0,0,16,0,0,0,59,0,4,0,28,0,0,0,6,0,0,0,3,0,0,0,32,0,4,0,29,0,0,0,3,0,0,0,15,0,0,
0,59,0,4,0,29,0,0,0,7,0,0,0,3,0,0,0,59,0,4,0,28,0,0,0,8,0,0,0,3,0,0,0,54,0,5,0,11,
0,0,0,2,0,0,0,0,0,0,0,12,0,0,0,248,0,2,0,30,0,0,0,61,0,4,0,15,0,0,0,31,0,0,0,3,0,
0,0,61,0,4,0,15,0,0,0,32,0,0,0,4,0,0,0,61,0,4,0,16,0,0,0,33,0,0,0,5,0,0,0,81,0,5,
0,13,0,0,0,34,0,0,0,31,0,0,0,0,0,0,0,81,0,5,0,13,0,0,0,35,0,0,0,31,0,0,0,1,0,0,0,
80,0,7,0,16,0,0,0,36,0,0,0,34,0,0,0,35,0,0,0,21,0,0,0,22,0,0,0,65,0,5,0,25,0,0,0,
37,0,0,0,10,0,0,0,20,0,0,0,61,0,4,0,23,0,0,0,38,0,0,0,37,0,0,0,144,0,5,0,16,0,0,0,
39,0,0,0,36,0,0,0,38,0,0,0,79,0,8,0,14,0,0,0,40,0,0,0,33,0,0,0,33,0,0,0,0,0,0,0,1,
0,0,0,2,0,0,0,12,0,7,0,14,0,0,0,41,0,0,0,1,0,0,0,26,0,0,0,40,0,0,0,18,0,0,0,81,0,
5,0,13,0,0,0,42,0,0,0,33,0,0,0,3,0,0,0,81,0,5,0,13,0,0,0,43,0,0,0,41,0,0,0,0,0,0,
0,81,0,5,0,13,0,0,0,44,0,0,0,41,0,0,0,1,0,0,0,81,0,5,0,13,0,0,0,45,0,0,0,41,0,0,0,
2,0,0,0,80,0,7,0,16,0,0,0,46,0,0,0,43,0,0,0,44,0,0,0,45,0,0,0,42,0,0,0,62,0,3,0,6,
0,0,0,39,0,0,0,62,0,3,0,7,0,0,0,32,0,0,0,62,0,3,0,8,0,0,0,46,0,0,0,253,0,1,0,56,0,
1,0,1,0,0,0,2,0,0,0,8,3,0,0,3,2,35,7,0,3,1,0,11,0,8,0,24,0,0,0,0,0,0,0,17,0,2,0,1,
0,0,0,10,0,9,0,83,80,86,95,71,79,79,71,76,69,95,104,108,115,108,95,102,117,110,99,
116,105,111,110,97,108,105,116,121,49,0,0,11,0,6,0,1,0,0,0,71,76,83,76,46,115,116,
100,46,52,53,48,0,0,0,0,14,0,3,0,0,0,0,0,1,0,0,0,15,0,7,0,4,0,0,0,2,0,0,0,112,115,
0,0,3,0,0,0,4,0,0,0,5,0,0,0,16,0,3,0,2,0,0,0,7,0,0,0,3,0,3,0,5,0,0,0,244,1,0,0,5,
0,3,0,2,0,0,0,112,115,0,0,5,0,5,0,6,0,0,0,116,101,120,116,117,114,101,48,0,0,0,0,
5,0,5,0,3,0,0,0,105,110,112,117,116,46,117,118,0,0,0,0,5,0,5,0,4,0,0,0,105,110,112,
117,116,46,99,111,108,0,0,0,5,0,7,0,5,0,0,0,64,101,110,116,114,121,80,111,105,110,
116,79,117,116,112,117,116,0,0,0,71,0,4,0,6,0,0,0,33,0,0,0,100,0,0,0,71,0,4,0,6,0,
0,0,34,0,0,0,0,0,0,0,71,0,4,0,3,0,0,0,30,0,0,0,0,0,0,0,0,22,6,0,3,0,0,0,3,22,0,0,
84,69,88,67,79,79,82,68,48,0,0,0,71,0,4,0,4,0,0,0,30,0,0,0,1,0,0,0,0,22,5,0,4,0,0,
0,3,22,0,0,67,79,76,79,82,48,0,0,71,0,4,0,5,0,0,0,30,0,0,0,0,0,0,0,0,22,6,0,5,0,0,
0,3,22,0,0,83,86,95,84,65,82,71,69,84,0,0,0,19,0,2,0,7,0,0,0,33,0,3,0,8,0,0,0,7,0,
0,0,22,0,3,0,9,0,0,0,32,0,0,0,23,0,4,0,10,0,0,0,9,0,0,0,4,0,0,0,23,0,4,0,11,0,0,0,
9,0,0,0,2,0,0,0,25,0,9,0,12,0,0,0,9,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
0,0,0,0,27,0,3,0,13,0,0,0,12,0,0,0,32,0,4,0,14,0,0,0,0,0,0,0,13,0,0,0,59,0,4,0,14,
0,0,0,6,0,0,0,0,0,0,0,32,0,4,0,15,0,0,0,1,0,0,0,10,0,0,0,32,0,4,0,16,0,0,0,1,0,0,
0,11,0,0,0,59,0,4,0,16,0,0,0,3,0,0,0,1,0,0,0,59,0,4,0,15,0,0,0,4,0,0,0,1,0,0,0,32,
0,4,0,17,0,0,0,3,0,0,0,10,0,0,0,59,0,4,0,17,0,0,0,5,0,0,0,3,0,0,0,54,0,5,0,7,0,0,
0,2,0,0,0,0,0,0,0,8,0,0,0,248,0,2,0,18,0,0,0,61,0,4,0,11,0,0,0,19,0,0,0,3,0,0,0,61,
0,4,0,10,0,0,0,20,0,0,0,4,0,0,0,61,0,4,0,13,0,0,0,21,0,0,0,6,0,0,0,87,0,5,0,10,0,
0,0,22,0,0,0,21,0,0,0,19,0,0,0,133,0,5,0,10,0,0,0,23,0,0,0,20,0,0,0,22,0,0,0,62,0,
3,0,5,0,0,0,23,0,0,0,253,0,1,0,56,0,1,0,};
