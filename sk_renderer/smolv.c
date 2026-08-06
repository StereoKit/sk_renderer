// SMOL-V, ported to C11 from smol-v by Aras Pranckevicius
// (https://github.com/aras-p/smol-v), MIT / public domain. See smolv.h for what
// this changes from upstream.

#include "smolv.h"

#include <string.h>

///////////////////////////////////////////////////////////////////////////////
// Known SPIR-V operations
///////////////////////////////////////////////////////////////////////////////

typedef enum {
	smolv_op_nop                    = 0,
	smolv_op_undef                  = 1,
	smolv_op_source_continued       = 2,
	smolv_op_source                 = 3,
	smolv_op_source_extension       = 4,
	smolv_op_name                   = 5,
	smolv_op_member_name            = 6,
	smolv_op_string                 = 7,
	smolv_op_line                   = 8,
	smolv_op_extension              = 10,
	smolv_op_ext_inst_import        = 11,
	smolv_op_ext_inst               = 12,
	smolv_op_vector_shuffle_compact = 13, // not SPIR-V, borrows an unused slot
	smolv_op_memory_model           = 14,
	smolv_op_entry_point            = 15,
	smolv_op_type_pointer           = 32,
	smolv_op_variable               = 59,
	smolv_op_load                   = 61,
	smolv_op_store                  = 62,
	smolv_op_access_chain           = 65,
	smolv_op_decorate               = 71,
	smolv_op_member_decorate        = 72,
	smolv_op_vector_shuffle         = 79,
	smolv_op_f_negate               = 127,
	smolv_op_f_add                  = 129,
	smolv_op_f_mul                  = 133,
	smolv_op_label                  = 248,
	smolv_op_no_line                = 317,
	smolv_op_module_processed       = 330,
	smolv_op_group_non_uniform_quad_swap = 366,
} smolv_op_;

// Per-opcode encoding rules, indexed by SPIR-V opcode. These decide the byte
// stream, so a change here bumps SKSC_FILE_VERSION.
#define SMOLV_R    0x01       // has a result ID
#define SMOLV_T    0x02       // has a type ID
#define SMOLV_V    0x04       // varint-encode the words left after the ones below
#define SMOLV_D(n) ((n) << 4) // words after type+result that are result-relative

static const uint8_t smolv_op_data[] = {
	0,                                  // Nop
	SMOLV_R|SMOLV_T,                    // Undef
	0,                                  // SourceContinued
	SMOLV_V,                            // Source
	0,                                  // SourceExtension
	0,                                  // Name
	0,                                  // MemberName
	0,                                  // String
	SMOLV_V,                            // Line
	SMOLV_R|SMOLV_T,                    // #9
	0,                                  // Extension
	SMOLV_R,                            // ExtInstImport
	SMOLV_R|SMOLV_T|SMOLV_V,            // ExtInst
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // VectorShuffleCompact - new in SMOL-V
	SMOLV_V,                            // MemoryModel
	SMOLV_V,                            // EntryPoint
	SMOLV_V,                            // ExecutionMode
	SMOLV_V,                            // Capability
	SMOLV_R|SMOLV_T,                    // #18
	SMOLV_R|SMOLV_V,                    // TypeVoid
	SMOLV_R|SMOLV_V,                    // TypeBool
	SMOLV_R|SMOLV_V,                    // TypeInt
	SMOLV_R|SMOLV_V,                    // TypeFloat
	SMOLV_R|SMOLV_V,                    // TypeVector
	SMOLV_R|SMOLV_V,                    // TypeMatrix
	SMOLV_R|SMOLV_V,                    // TypeImage
	SMOLV_R|SMOLV_V,                    // TypeSampler
	SMOLV_R|SMOLV_V,                    // TypeSampledImage
	SMOLV_R|SMOLV_V,                    // TypeArray
	SMOLV_R|SMOLV_V,                    // TypeRuntimeArray
	SMOLV_R|SMOLV_V,                    // TypeStruct
	SMOLV_R|SMOLV_V,                    // TypeOpaque
	SMOLV_R|SMOLV_V,                    // TypePointer
	SMOLV_R|SMOLV_V,                    // TypeFunction
	SMOLV_R|SMOLV_V,                    // TypeEvent
	SMOLV_R|SMOLV_V,                    // TypeDeviceEvent
	SMOLV_R|SMOLV_V,                    // TypeReserveId
	SMOLV_R|SMOLV_V,                    // TypeQueue
	SMOLV_R|SMOLV_V,                    // TypePipe
	SMOLV_V,                            // TypeForwardPointer
	SMOLV_R|SMOLV_T,                    // #40
	SMOLV_R|SMOLV_T,                    // ConstantTrue
	SMOLV_R|SMOLV_T,                    // ConstantFalse
	SMOLV_R|SMOLV_T,                    // Constant
	SMOLV_R|SMOLV_T|SMOLV_D(9),         // ConstantComposite
	SMOLV_R|SMOLV_T|SMOLV_V,            // ConstantSampler
	SMOLV_R|SMOLV_T,                    // ConstantNull
	SMOLV_R|SMOLV_T,                    // #47
	SMOLV_R|SMOLV_T,                    // SpecConstantTrue
	SMOLV_R|SMOLV_T,                    // SpecConstantFalse
	SMOLV_R|SMOLV_T,                    // SpecConstant
	SMOLV_R|SMOLV_T|SMOLV_D(9),         // SpecConstantComposite
	SMOLV_R|SMOLV_T,                    // SpecConstantOp
	SMOLV_R|SMOLV_T,                    // #53
	SMOLV_R|SMOLV_T|SMOLV_V,            // Function
	SMOLV_R|SMOLV_T,                    // FunctionParameter
	0,                                  // FunctionEnd
	SMOLV_R|SMOLV_T|SMOLV_D(9),         // FunctionCall
	SMOLV_R|SMOLV_T,                    // #58
	SMOLV_R|SMOLV_T|SMOLV_V,            // Variable
	SMOLV_R|SMOLV_T,                    // ImageTexelPointer
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // Load
	SMOLV_D(2)|SMOLV_V,                 // Store
	0,                                  // CopyMemory
	0,                                  // CopyMemorySized
	SMOLV_R|SMOLV_T|SMOLV_V,            // AccessChain
	SMOLV_R|SMOLV_T,                    // InBoundsAccessChain
	SMOLV_R|SMOLV_T,                    // PtrAccessChain
	SMOLV_R|SMOLV_T,                    // ArrayLength
	SMOLV_R|SMOLV_T,                    // GenericPtrMemSemantics
	SMOLV_R|SMOLV_T,                    // InBoundsPtrAccessChain
	SMOLV_V,                            // Decorate
	SMOLV_V,                            // MemberDecorate
	SMOLV_R,                            // DecorationGroup
	0,                                  // GroupDecorate
	0,                                  // GroupMemberDecorate
	SMOLV_R|SMOLV_T,                    // #76
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // VectorExtractDynamic
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // VectorInsertDynamic
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // VectorShuffle
	SMOLV_R|SMOLV_T|SMOLV_D(9),         // CompositeConstruct
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // CompositeExtract
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // CompositeInsert
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // CopyObject
	SMOLV_R|SMOLV_T,                    // Transpose
	SMOLV_R|SMOLV_T,                    // #85
	SMOLV_R|SMOLV_T,                    // SampledImage
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSampleImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSampleExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSampleDrefImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSampleDrefExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSampleProjImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSampleProjExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSampleProjDrefImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSampleProjDrefExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageFetch
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageGather
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageDrefGather
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageRead
	SMOLV_D(3)|SMOLV_V,                 // ImageWrite
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // Image
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageQueryFormat
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageQueryOrder
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ImageQuerySizeLod
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageQuerySize
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ImageQueryLod
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageQueryLevels
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageQuerySamples
	SMOLV_R|SMOLV_T,                    // #108
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertFToU
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertFToS
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertSToF
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertUToF
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // UConvert
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // SConvert
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // FConvert
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // QuantizeToF16
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertPtrToU
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // SatConvertSToU
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // SatConvertUToS
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ConvertUToPtr
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // PtrCastToGeneric
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // GenericCastToPtr
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GenericCastToPtrExplicit
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // Bitcast
	SMOLV_R|SMOLV_T,                    // #125
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // SNegate
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // FNegate
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // IAdd
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FAdd
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ISub
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FSub
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // IMul
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FMul
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // UDiv
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SDiv
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FDiv
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // UMod
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SRem
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SMod
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FRem
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FMod
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // VectorTimesScalar
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // MatrixTimesScalar
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // VectorTimesMatrix
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // MatrixTimesVector
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // MatrixTimesMatrix
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // OuterProduct
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // Dot
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // IAddCarry
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ISubBorrow
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // UMulExtended
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SMulExtended
	SMOLV_R|SMOLV_T,                    // #153
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // Any
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // All
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // IsNan
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // IsInf
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // IsFinite
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // IsNormal
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // SignBitSet
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // LessOrGreater
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // Ordered
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // Unordered
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // LogicalEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // LogicalNotEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // LogicalOr
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // LogicalAnd
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // LogicalNot
	SMOLV_R|SMOLV_T|SMOLV_D(3),         // Select
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // IEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // INotEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // UGreaterThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SGreaterThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // UGreaterThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SGreaterThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ULessThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SLessThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ULessThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // SLessThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdNotEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordNotEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdLessThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordLessThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdGreaterThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordGreaterThan
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdLessThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordLessThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FOrdGreaterThanEqual
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // FUnordGreaterThanEqual
	SMOLV_R|SMOLV_T,                    // #192
	SMOLV_R|SMOLV_T,                    // #193
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ShiftRightLogical
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ShiftRightArithmetic
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // ShiftLeftLogical
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // BitwiseOr
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // BitwiseXor
	SMOLV_R|SMOLV_T|SMOLV_D(2),         // BitwiseAnd
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // Not
	SMOLV_R|SMOLV_T|SMOLV_D(4),         // BitFieldInsert
	SMOLV_R|SMOLV_T|SMOLV_D(3),         // BitFieldSExtract
	SMOLV_R|SMOLV_T|SMOLV_D(3),         // BitFieldUExtract
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // BitReverse
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // BitCount
	SMOLV_R|SMOLV_T,                    // #206
	SMOLV_R|SMOLV_T,                    // DPdx
	SMOLV_R|SMOLV_T,                    // DPdy
	SMOLV_R|SMOLV_T,                    // Fwidth
	SMOLV_R|SMOLV_T,                    // DPdxFine
	SMOLV_R|SMOLV_T,                    // DPdyFine
	SMOLV_R|SMOLV_T,                    // FwidthFine
	SMOLV_R|SMOLV_T,                    // DPdxCoarse
	SMOLV_R|SMOLV_T,                    // DPdyCoarse
	SMOLV_R|SMOLV_T,                    // FwidthCoarse
	SMOLV_R|SMOLV_T,                    // #216
	SMOLV_R|SMOLV_T,                    // #217
	0,                                  // EmitVertex
	0,                                  // EndPrimitive
	0,                                  // EmitStreamVertex
	0,                                  // EndStreamPrimitive
	SMOLV_R|SMOLV_T,                    // #222
	SMOLV_R|SMOLV_T,                    // #223
	SMOLV_D(3),                         // ControlBarrier
	SMOLV_D(2),                         // MemoryBarrier
	SMOLV_R|SMOLV_T,                    // #226
	SMOLV_R|SMOLV_T,                    // AtomicLoad
	0,                                  // AtomicStore
	SMOLV_R|SMOLV_T,                    // AtomicExchange
	SMOLV_R|SMOLV_T,                    // AtomicCompareExchange
	SMOLV_R|SMOLV_T,                    // AtomicCompareExchangeWeak
	SMOLV_R|SMOLV_T,                    // AtomicIIncrement
	SMOLV_R|SMOLV_T,                    // AtomicIDecrement
	SMOLV_R|SMOLV_T,                    // AtomicIAdd
	SMOLV_R|SMOLV_T,                    // AtomicISub
	SMOLV_R|SMOLV_T,                    // AtomicSMin
	SMOLV_R|SMOLV_T,                    // AtomicUMin
	SMOLV_R|SMOLV_T,                    // AtomicSMax
	SMOLV_R|SMOLV_T,                    // AtomicUMax
	SMOLV_R|SMOLV_T,                    // AtomicAnd
	SMOLV_R|SMOLV_T,                    // AtomicOr
	SMOLV_R|SMOLV_T,                    // AtomicXor
	SMOLV_R|SMOLV_T,                    // #243
	SMOLV_R|SMOLV_T,                    // #244
	SMOLV_R|SMOLV_T,                    // Phi
	SMOLV_D(2)|SMOLV_V,                 // LoopMerge
	SMOLV_D(1)|SMOLV_V,                 // SelectionMerge
	SMOLV_R,                            // Label
	SMOLV_D(1),                         // Branch
	SMOLV_D(3)|SMOLV_V,                 // BranchConditional
	0,                                  // Switch
	0,                                  // Kill
	0,                                  // Return
	0,                                  // ReturnValue
	0,                                  // Unreachable
	0,                                  // LifetimeStart
	0,                                  // LifetimeStop
	SMOLV_R|SMOLV_T,                    // #258
	SMOLV_R|SMOLV_T,                    // GroupAsyncCopy
	0,                                  // GroupWaitEvents
	SMOLV_R|SMOLV_T,                    // GroupAll
	SMOLV_R|SMOLV_T,                    // GroupAny
	SMOLV_R|SMOLV_T,                    // GroupBroadcast
	SMOLV_R|SMOLV_T,                    // GroupIAdd
	SMOLV_R|SMOLV_T,                    // GroupFAdd
	SMOLV_R|SMOLV_T,                    // GroupFMin
	SMOLV_R|SMOLV_T,                    // GroupUMin
	SMOLV_R|SMOLV_T,                    // GroupSMin
	SMOLV_R|SMOLV_T,                    // GroupFMax
	SMOLV_R|SMOLV_T,                    // GroupUMax
	SMOLV_R|SMOLV_T,                    // GroupSMax
	SMOLV_R|SMOLV_T,                    // #272
	SMOLV_R|SMOLV_T,                    // #273
	SMOLV_R|SMOLV_T,                    // ReadPipe
	SMOLV_R|SMOLV_T,                    // WritePipe
	SMOLV_R|SMOLV_T,                    // ReservedReadPipe
	SMOLV_R|SMOLV_T,                    // ReservedWritePipe
	SMOLV_R|SMOLV_T,                    // ReserveReadPipePackets
	SMOLV_R|SMOLV_T,                    // ReserveWritePipePackets
	0,                                  // CommitReadPipe
	0,                                  // CommitWritePipe
	SMOLV_R|SMOLV_T,                    // IsValidReserveId
	SMOLV_R|SMOLV_T,                    // GetNumPipePackets
	SMOLV_R|SMOLV_T,                    // GetMaxPipePackets
	SMOLV_R|SMOLV_T,                    // GroupReserveReadPipePackets
	SMOLV_R|SMOLV_T,                    // GroupReserveWritePipePackets
	0,                                  // GroupCommitReadPipe
	0,                                  // GroupCommitWritePipe
	SMOLV_R|SMOLV_T,                    // #289
	SMOLV_R|SMOLV_T,                    // #290
	SMOLV_R|SMOLV_T,                    // EnqueueMarker
	SMOLV_R|SMOLV_T,                    // EnqueueKernel
	SMOLV_R|SMOLV_T,                    // GetKernelNDrangeSubGroupCount
	SMOLV_R|SMOLV_T,                    // GetKernelNDrangeMaxSubGroupSize
	SMOLV_R|SMOLV_T,                    // GetKernelWorkGroupSize
	SMOLV_R|SMOLV_T,                    // GetKernelPreferredWorkGroupSizeMultiple
	0,                                  // RetainEvent
	0,                                  // ReleaseEvent
	SMOLV_R|SMOLV_T,                    // CreateUserEvent
	SMOLV_R|SMOLV_T,                    // IsValidEvent
	0,                                  // SetUserEventStatus
	0,                                  // CaptureEventProfilingInfo
	SMOLV_R|SMOLV_T,                    // GetDefaultQueue
	SMOLV_R|SMOLV_T,                    // BuildNDRange
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSparseSampleImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSparseSampleExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseSampleDrefImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseSampleDrefExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSparseSampleProjImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSparseSampleProjExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseSampleProjDrefImplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseSampleProjDrefExplicitLod
	SMOLV_R|SMOLV_T|SMOLV_D(2)|SMOLV_V, // ImageSparseFetch
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseGather
	SMOLV_R|SMOLV_T|SMOLV_D(3)|SMOLV_V, // ImageSparseDrefGather
	SMOLV_R|SMOLV_T|SMOLV_D(1),         // ImageSparseTexelsResident
	0,                                  // NoLine
	SMOLV_R|SMOLV_T,                    // AtomicFlagTestAndSet
	0,                                  // AtomicFlagClear
	SMOLV_R|SMOLV_T,                    // ImageSparseRead
	SMOLV_R|SMOLV_T,                    // SizeOf
	SMOLV_R|SMOLV_T,                    // TypePipeStorage
	SMOLV_R|SMOLV_T,                    // ConstantPipeStorage
	SMOLV_R|SMOLV_T,                    // CreatePipeFromPipeStorage
	SMOLV_R|SMOLV_T,                    // GetKernelLocalSizeForSubgroupCount
	SMOLV_R|SMOLV_T,                    // GetKernelMaxNumSubgroups
	SMOLV_R|SMOLV_T,                    // TypeNamedBarrier
	SMOLV_R|SMOLV_T|SMOLV_V,            // NamedBarrierInitialize
	SMOLV_D(2)|SMOLV_V,                 // MemoryNamedBarrier
	SMOLV_R|SMOLV_T,                    // ModuleProcessed
	SMOLV_V,                            // ExecutionModeId
	SMOLV_V,                            // DecorateId
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformElect
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformAll
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformAny
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformAllEqual
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBroadcast
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBroadcastFirst
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBallot
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformInverseBallot
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBallotBitExtract
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBallotBitCount
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBallotFindLSB
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBallotFindMSB
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformShuffle
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformShuffleXor
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformShuffleUp
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformShuffleDown
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformIAdd
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformFAdd
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformIMul
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformFMul
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformSMin
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformUMin
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformFMin
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformSMax
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformUMax
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformFMax
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBitwiseAnd
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBitwiseOr
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformBitwiseXor
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformLogicalAnd
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformLogicalOr
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformLogicalXor
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformQuadBroadcast
	SMOLV_R|SMOLV_T|SMOLV_D(1)|SMOLV_V, // GroupNonUniformQuadSwap
};

#define SMOLV_KNOWN_OPS_COUNT ((int32_t)(sizeof(smolv_op_data) / sizeof(smolv_op_data[0])))

_Static_assert(SMOLV_KNOWN_OPS_COUNT == smolv_op_group_non_uniform_quad_swap + 1,
	"smolv_op_data table doesn't cover exactly the known SPIR-V ops");

#define SMOLV_SPIRV_HEADER_MAGIC 0x07230203
#define SMOLV_HEADER_MAGIC       0x534D4F4C // "SMOL"

// One output stream per field kind. Grouping like fields costs a length each but
// gives deflate uniform statistics instead of a rotating per-instruction pattern.
typedef enum {
	smolv_s_oplen,  // instruction length + opcode
	smolv_s_type,   // type IDs
	smolv_s_result, // result ID deltas
	smolv_s_relid,  // IDs stored relative to the result ID
	smolv_s_var,    // trailing words, varint encoded
	smolv_s_raw,    // trailing words, stored as-is
	smolv_s_decor,  // Decorate targets and MemberDecorate runs
	smolv_s_swiz,   // VectorShuffleCompact swizzle bytes
	smolv_s_count,
} smolv_s_;

// int32_t rather than smolv_op_, so a bogus opcode from a stream still lands in
// the range check instead of undefined enum territory.
static bool smolv_op_has_result(int32_t op) {
	if (op < 0 || op >= SMOLV_KNOWN_OPS_COUNT) return false;
	return (smolv_op_data[op] & SMOLV_R) != 0;
}

static bool smolv_op_has_type(int32_t op) {
	if (op < 0 || op >= SMOLV_KNOWN_OPS_COUNT) return false;
	return (smolv_op_data[op] & SMOLV_T) != 0;
}

static int32_t smolv_op_delta_from_result(int32_t op) {
	if (op < 0 || op >= SMOLV_KNOWN_OPS_COUNT) return 0;
	return smolv_op_data[op] >> 4;
}

static bool smolv_op_var_rest(int32_t op) {
	if (op < 0 || op >= SMOLV_KNOWN_OPS_COUNT) return false;
	return (smolv_op_data[op] & SMOLV_V) != 0;
}

static bool smolv_op_debug_info(int32_t op) {
	return op == smolv_op_source_continued ||
	       op == smolv_op_source           ||
	       op == smolv_op_source_extension ||
	       op == smolv_op_name             ||
	       op == smolv_op_member_name      ||
	       op == smolv_op_string           ||
	       op == smolv_op_line             ||
	       op == smolv_op_no_line          ||
	       op == smolv_op_module_processed;
}

// Operand word count for decorations common enough to skip storing a length on.
// -1 means the length is written out.
static int32_t smolv_decoration_extra_ops(uint32_t decoration) {
	if (decoration == 0 || (decoration >= 2 && decoration <= 5)) return 0;  // RelaxedPrecision, Block..ColMajor
	if (decoration >= 29 && decoration <= 37)                    return 1;  // Stream..XfbStride
	return -1;
}

///////////////////////////////////////////////////////////////////////////////
// Stream primitives
///////////////////////////////////////////////////////////////////////////////

// Writes are bounds-checked and latch ok low, so callers check once at the end.
typedef struct {
	uint8_t *data;
	size_t   at;
	size_t   cap;
	bool     ok;
} smolv_write_t;

static void smolv_put(smolv_write_t *w, uint8_t b) {
	if (w->at + 1 > w->cap) { w->ok = false; return; }
	if (w->data) w->data[w->at] = b;
	w->at++;
}

// Little-endian on every host, so blobs move between machines unchanged.
static void smolv_put4(smolv_write_t *w, uint32_t v) {
	if (w->at + 4 > w->cap) { w->ok = false; return; }
	if (w->data) {
		w->data[w->at + 0] = (uint8_t)( v        & 0xFF);
		w->data[w->at + 1] = (uint8_t)((v >>  8) & 0xFF);
		w->data[w->at + 2] = (uint8_t)((v >> 16) & 0xFF);
		w->data[w->at + 3] = (uint8_t)((v >> 24) & 0xFF);
	}
	w->at += 4;
}

// Host order, matching how the driver reads them and how the encoder read them in.
static void smolv_put_word(smolv_write_t *w, uint32_t v) {
	if (w->at + 4 > w->cap) { w->ok = false; return; }
	memcpy(w->data + w->at, &v, sizeof(v));
	w->at += 4;
}

static bool smolv_read4(const uint8_t **data, const uint8_t *data_end, uint32_t *out_val) {
	if (*data + 4 > data_end) return false;
	const uint8_t *d = *data;
	*out_val = ((uint32_t)d[0]) | ((uint32_t)d[1] << 8) | ((uint32_t)d[2] << 16) | ((uint32_t)d[3] << 24);
	*data += 4;
	return true;
}

// High bit says more bytes follow, low 7 are payload, so values under 128 cost a
// single byte. 1-5 bytes total.

static void smolv_put_varint(smolv_write_t *w, uint32_t v) {
	while (v > 127) {
		smolv_put(w, (uint8_t)((v & 127) | 128));
		v >>= 7;
	}
	smolv_put(w, (uint8_t)(v & 127));
}

static bool smolv_read_varint(const uint8_t **data, const uint8_t *data_end, uint32_t *out_val) {
	uint32_t v     = 0;
	uint32_t shift = 0;
	while (*data < data_end) {
		uint8_t b = **data;
		v |= (uint32_t)(b & 127) << shift;
		shift += 7;
		(*data)++;
		if (!(b & 128)) {
			*out_val = v;
			return true;
		}
		// A 32-bit value never needs a sixth byte, and shifting further is UB
		if (shift >= 35) return false;
	}
	return false; // ran out of input mid-value
}

// Folds the sign into the low bit so small negative deltas stay small under
// varint. Unsigned arithmetic throughout to stay defined at the edges.
static uint32_t smolv_zig_encode(uint32_t v) {
	return (v << 1) ^ (0u - (v >> 31));
}

static uint32_t smolv_zig_decode(uint32_t u) {
	return (u & 1) ? ((u >> 1) ^ 0xFFFFFFFFu) : (u >> 1);
}

// Swaps the most common opcodes down into the 0-15 range where they varint to a
// single byte, trading places with the rare ops living there. Its own inverse.
static smolv_op_ smolv_remap_op(smolv_op_ op) {
#	define SMOLV_SWAP_OP(op1, op2) if (op == (op1)) return (op2); if (op == (op2)) return (op1)
	SMOLV_SWAP_OP(smolv_op_decorate,        smolv_op_nop);              //  0: 24%
	SMOLV_SWAP_OP(smolv_op_load,            smolv_op_undef);            //  1: 17%
	SMOLV_SWAP_OP(smolv_op_store,           smolv_op_source_continued); //  2: 9%
	SMOLV_SWAP_OP(smolv_op_access_chain,    smolv_op_source);           //  3: 7.2%
	SMOLV_SWAP_OP(smolv_op_vector_shuffle,  smolv_op_source_extension); //  4: 5.0%
	// Name       - already a small enum value -  5: 4.4%
	// MemberName - already a small enum value -  6: 2.9%
	SMOLV_SWAP_OP(smolv_op_member_decorate, smolv_op_string);           //  7: 4.0%
	SMOLV_SWAP_OP(smolv_op_label,           smolv_op_line);             //  8: 0.9%
	SMOLV_SWAP_OP(smolv_op_variable,        (smolv_op_)9);              //  9: 3.9%
	SMOLV_SWAP_OP(smolv_op_f_mul,           smolv_op_extension);        // 10: 3.9%
	SMOLV_SWAP_OP(smolv_op_f_add,           smolv_op_ext_inst_import);  // 11: 2.5%
	// ExtInst              - already a small enum value - 12: 1.2%
	// VectorShuffleCompact - used for the compact shuffle encoding
	SMOLV_SWAP_OP(smolv_op_type_pointer,    smolv_op_memory_model);     // 14: 2.2%
	SMOLV_SWAP_OP(smolv_op_f_negate,        smolv_op_entry_point);      // 15: 1.1%
#	undef SMOLV_SWAP_OP
	return op;
}

// Instruction length wants to fit in 3 bits. Every op has a guaranteed minimum
// length, so subtract it off before encoding and add it back after.
static uint32_t smolv_encode_len(smolv_op_ op, uint32_t len) {
	len--;
	if (op == smolv_op_vector_shuffle)         len -= 4;
	if (op == smolv_op_vector_shuffle_compact) len -= 4;
	if (op == smolv_op_decorate)               len -= 2;
	if (op == smolv_op_load)                   len -= 3;
	if (op == smolv_op_access_chain)           len -= 3;
	return len;
}

static uint32_t smolv_decode_len(smolv_op_ op, uint32_t len) {
	len++;
	if (op == smolv_op_vector_shuffle)         len += 4;
	if (op == smolv_op_vector_shuffle_compact) len += 4;
	if (op == smolv_op_decorate)               len += 2;
	if (op == smolv_op_load)                   len += 3;
	if (op == smolv_op_access_chain)           len += 3;
	return len;
}

// SPIR-V packs length and op as 0xLLLLOOOO; repacking to 0xLLLOOOLO puts the
// common case of op<16 with len<8 in a single varint byte.
static bool smolv_put_length_op(smolv_write_t *w, uint32_t len, smolv_op_ op) {
	len = smolv_encode_len(op, len);
	// SPIR-V lengths are 16 bits, so more than that means smolv_encode_len
	// wrapped on malformed input, like a vector shuffle under 4 words.
	if (len > 0xFFFF) return false;
	op = smolv_remap_op(op);
	smolv_put_varint(w, ((len >> 4) << 20) | (((uint32_t)op >> 4) << 8) | ((len & 0xF) << 4) | ((uint32_t)op & 0xF));
	return true;
}

static bool smolv_read_length_op(const uint8_t **data, const uint8_t *data_end, uint32_t *out_len, smolv_op_ *out_op) {
	uint32_t val;
	if (!smolv_read_varint(data, data_end, &val)) return false;
	uint32_t  len = ((val >> 20) << 4) | ((val >> 4) & 0xF);
	smolv_op_ op  = (smolv_op_)(((val >> 4) & 0xFFF0) | (val & 0xF));

	op       = smolv_remap_op(op);
	*out_len = smolv_decode_len(op, len);
	*out_op  = op;
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Headers
///////////////////////////////////////////////////////////////////////////////

static bool smolv_check_generic_header(const uint8_t *bytes, size_t byte_count, uint32_t expected_magic, uint32_t version_mask) {
	if (bytes == NULL)    return false;
	if (byte_count < 20)  return false; // 5 header words

	uint32_t magic, version;
	memcpy(&magic,   bytes,     sizeof(magic));
	memcpy(&version, bytes + 4, sizeof(version));
	if (magic != expected_magic) return false;

	version &= version_mask;
	if (version < 0x00010000 || version > 0x00010600) return false; // only SPIR-V 1.0 through 1.6
	return true;
}

static bool smolv_check_spirv_header(const uint32_t *words, size_t word_count) {
	// A byte-reversed magic means a big-endian module, which would need every
	// word swapped. Unsupported, so it just fails the magic check.
	return smolv_check_generic_header((const uint8_t *)words, word_count * 4, SMOLV_SPIRV_HEADER_MAGIC, 0xFFFFFFFF);
}

static bool smolv_check_smol_header(const uint8_t *bytes, size_t byte_count) {
	// The version word is the SPIR-V one verbatim, this encoding carries no
	// version of its own. See smolv.h.
	if (!smolv_check_generic_header(bytes, byte_count, SMOLV_HEADER_MAGIC, 0xFFFFFFFF)) return false;
	return byte_count >= 24; // one word past the header holds the decoded length
}

bool smolv_is_smolv(const void *data, size_t size) {
	return smolv_check_smol_header((const uint8_t *)data, size);
}

size_t smolv_decoded_size(const void *smolv_data, size_t smolv_size) {
	if (!smolv_check_smol_header((const uint8_t *)smolv_data, smolv_size)) return 0;
	uint32_t size;
	memcpy(&size, (const uint8_t *)smolv_data + 20, sizeof(size));
	return size;
}

size_t smolv_encode_bound(size_t spirv_size) {
	// Worst case is a one-member MemberDecorate run at 26 bytes out for 16 in,
	// beating a varint growing a word 4 bytes to 5. 2x covers both.
	return spirv_size * 2 + 64;
}

///////////////////////////////////////////////////////////////////////////////
// Encode
///////////////////////////////////////////////////////////////////////////////

// Encodes the module body into the eight streams. Writers with a NULL data
// pointer only count, which is how the caller sizes the streams before placing
// them.
static bool smolv_encode_body(const uint32_t *words, size_t word_count, uint32_t flags, smolv_write_t *st, size_t *out_stripped_words) {
	const uint32_t *words_end = words + word_count;

	size_t   stripped_word_count = word_count;
	uint32_t prev_result         = 0;
	uint32_t prev_decorate       = 0;

	words += 5;
	while (words < words_end) {
		uint32_t  instr_len = words[0] >> 16;
		smolv_op_ op        = (smolv_op_)(words[0] & 0xFFFF);
		if (instr_len < 1)                 goto fail; // length covers the op word itself, so it's never zero
		if (words + instr_len > words_end) goto fail; // instruction runs past the end of the module

		if ((flags & smolv_encode_strip_debug_info) && smolv_op_debug_info(op)) {
			stripped_word_count -= instr_len;
			words               += instr_len;
			continue;
		}

		// A shuffle of at most 4 components, each selecting from [0..3], fits one
		// swizzle byte. Retag those as the VectorShuffleCompact pseudo-op.
		uint32_t swizzle = 0;
		if (op == smolv_op_vector_shuffle && instr_len <= 9) {
			uint32_t swz0 = instr_len > 5 ? words[5] : 0;
			uint32_t swz1 = instr_len > 6 ? words[6] : 0;
			uint32_t swz2 = instr_len > 7 ? words[7] : 0;
			uint32_t swz3 = instr_len > 8 ? words[8] : 0;
			if (swz0 < 4 && swz1 < 4 && swz2 < 4 && swz3 < 4) {
				op      = smolv_op_vector_shuffle_compact;
				swizzle = (swz0 << 6) | (swz1 << 4) | (swz2 << 2) | swz3;
			}
		}

		if (!smolv_put_length_op(&st[smolv_s_oplen], instr_len, op)) goto fail;

		uint32_t ioffs = 1;
		if (smolv_op_has_type(op)) {
			if (ioffs >= instr_len) goto fail;
			smolv_put_varint(&st[smolv_s_type], words[ioffs]);
			ioffs++;
		}
		// Result IDs mostly step forward by small amounts, so store the delta.
		// Some are negative, hence zigzag.
		if (smolv_op_has_result(op)) {
			if (ioffs >= instr_len) goto fail;
			uint32_t v = words[ioffs];
			smolv_put_varint(&st[smolv_s_result], smolv_zig_encode(v - prev_result));
			prev_result = v;
			ioffs++;
		}

		// Decorate & MemberDecorate: target IDs relative to the previous
		// decoration. Deltas often go negative after spirv-remap, so zigzag.
		if (op == smolv_op_decorate || op == smolv_op_member_decorate) {
			if (ioffs >= instr_len) goto fail;
			uint32_t v = words[ioffs];
			smolv_put_varint(&st[smolv_s_decor], smolv_zig_encode(v - prev_decorate));
			prev_decorate = v;
			ioffs++;
		}

		// A row of MemberDecorate usually decorates one type with increasing
		// member indices, so fold the whole run behind a count byte.
		if (op == smolv_op_member_decorate) {
			const uint32_t  decoration_type = words[ioffs - 1];
			const uint32_t *member_words    = words;
			uint32_t        prev_index      = 0;
			uint32_t        prev_offset     = 0;

			const size_t count_at = st[smolv_s_decor].at;
			smolv_put(&st[smolv_s_decor], 0);
			int32_t count = 0;
			while (member_words < words_end && count < 255) {
				uint32_t  member_len = member_words[0] >> 16;
				smolv_op_ member_op  = (smolv_op_)(member_words[0] & 0xFFFF);
				if (member_len < 1)                        goto fail;
				if (member_words + member_len > words_end) goto fail;

				if (member_op != smolv_op_member_decorate) break;
				if (member_len < 4)                        goto fail; // malformed member decoration
				if (member_words[1] != decoration_type)    break;

				uint32_t member_index = member_words[2];
				smolv_put_varint(&st[smolv_s_decor], member_index - prev_index);
				prev_index = member_index;

				uint32_t member_dec = member_words[3];
				smolv_put_varint(&st[smolv_s_decor], member_dec);
				const int32_t known_extra_ops = smolv_decoration_extra_ops(member_dec);
				if (known_extra_ops == -1)
					smolv_put_varint(&st[smolv_s_decor], member_len - 4);
				else if ((uint32_t)known_extra_ops + 4 != member_len)
					goto fail; // length disagrees with the decoration's known operand count

				// Offset decorations climb linearly through a struct, so they
				// delta well.
				if (member_dec == 35) { // Offset
					if (member_len != 5) goto fail;
					smolv_put_varint(&st[smolv_s_decor], member_words[4] - prev_offset);
					prev_offset = member_words[4];
				} else {
					for (uint32_t i = 4; i < member_len; i++)
						smolv_put_varint(&st[smolv_s_decor], member_words[i]);
				}

				member_words += member_len;
				count++;
			}
			if (!st[smolv_s_decor].ok) goto fail;
			if (st[smolv_s_decor].data) st[smolv_s_decor].data[count_at] = (uint8_t)count;
			words = member_words;
			continue;
		}

		// Relative to the result ID and zigzagged, since these go negative on
		// branches and after a spirv-remap pass.
		int32_t relative_count = smolv_op_delta_from_result(op);
		for (int32_t i = 0; i < relative_count && ioffs < instr_len; i++, ioffs++) {
			smolv_put_varint(&st[smolv_s_relid], smolv_zig_encode(prev_result - words[ioffs]));
		}

		if (op == smolv_op_vector_shuffle_compact) {
			smolv_put(&st[smolv_s_swiz], (uint8_t)swizzle);
			ioffs = instr_len;
		} else if (smolv_op_var_rest(op)) {
			// The rest are expected to be small integers
			for (; ioffs < instr_len; ioffs++)
				smolv_put_varint(&st[smolv_s_var], words[ioffs]);
		} else {
			for (; ioffs < instr_len; ioffs++)
				smolv_put4(&st[smolv_s_raw], words[ioffs]);
		}

		words += instr_len;
	}

	*out_stripped_words = stripped_word_count;
	for (int32_t i = 0; i < smolv_s_count; i++)
		if (!st[i].ok) return false;
	return true;

fail:
	return false;
}

bool smolv_encode(const void *spirv_data, size_t spirv_size, void *out_smolv, size_t out_capacity, size_t *out_size, uint32_t flags) {
	if (out_size) *out_size = 0;
	if (out_smolv == NULL) return false;

	const size_t word_count = spirv_size / 4;
	if (word_count * 4 != spirv_size) return false;
	const uint32_t *words = (const uint32_t *)spirv_data;
	if (!smolv_check_spirv_header(words, word_count)) return false;

	// Stream lengths decide where each stream lands, and the header carries them,
	// so a counting pass runs first. That also means the second pass can write
	// straight into the output at final offsets, with no scratch buffer at all.
	smolv_write_t st[smolv_s_count];
	for (int32_t i = 0; i < smolv_s_count; i++)
		st[i] = (smolv_write_t){ NULL, 0, SIZE_MAX, true };

	size_t stripped_word_count = 0;
	if (!smolv_encode_body(words, word_count, flags, st, &stripped_word_count)) return false;

	// Header mirrors SPIR-V's but for the magic, then the stream lengths
	smolv_write_t w = { (uint8_t *)out_smolv, 0, out_capacity, true };
	smolv_put4(&w, SMOLV_HEADER_MAGIC);
	smolv_put4(&w, words[1]); // SPIR-V version
	smolv_put4(&w, words[2]); // generator
	smolv_put4(&w, words[3]); // bound
	smolv_put4(&w, words[4]); // schema
	smolv_put4(&w, (uint32_t)stripped_word_count * 4); // space the decoder needs
	for (int32_t i = 0; i < smolv_s_count; i++)
		smolv_put_varint(&w, (uint32_t)st[i].at);
	if (!w.ok) return false;

	size_t at = w.at;
	for (int32_t i = 0; i < smolv_s_count; i++) {
		size_t len = st[i].at;
		if (at + len > out_capacity) return false;
		st[i] = (smolv_write_t){ (uint8_t *)out_smolv + at, 0, len, true };
		at += len;
	}

	size_t stripped_again = 0;
	if (!smolv_encode_body(words, word_count, flags, st, &stripped_again)) return false;
	for (int32_t i = 0; i < smolv_s_count; i++)
		if (st[i].at != st[i].cap) return false; // the two passes disagreed

	if (out_size) *out_size = at;
	return true;
}

///////////////////////////////////////////////////////////////////////////////
// Decode
///////////////////////////////////////////////////////////////////////////////

bool smolv_decode(const void *smolv_data, size_t smolv_size, void *out_spirv, size_t out_capacity) {
	const size_t needed = smolv_decoded_size(smolv_data, smolv_size);
	if (needed == 0)            return false; // not valid SMOL-V
	if (out_capacity < needed)  return false;
	if (out_spirv == NULL)      return false;

	const uint8_t *bytes     = (const uint8_t *)smolv_data;
	const uint8_t *bytes_end = bytes + smolv_size;

	// Cap the writer at the size the header promised. A blob claiming less than
	// it decodes to then fails cleanly rather than running past the buffer.
	smolv_write_t w = { (uint8_t *)out_spirv, 0, needed, true };

	uint32_t val;

	smolv_put_word(&w, SMOLV_SPIRV_HEADER_MAGIC); bytes += 4;
	if (!smolv_read4(&bytes, bytes_end, &val)) return false;
	smolv_put_word(&w, val);                                                 // version
	if (!smolv_read4(&bytes, bytes_end, &val)) return false;
	smolv_put_word(&w, val);                                                 // generator
	if (!smolv_read4(&bytes, bytes_end, &val)) return false;
	smolv_put_word(&w, val);                                                 // bound
	if (!smolv_read4(&bytes, bytes_end, &val)) return false;
	smolv_put_word(&w, val);                                                 // schema
	bytes += 4;                                                              // decoded size, already read

	// One read cursor per field kind, over the region the header says each
	// stream occupies.
	const uint8_t *cur[smolv_s_count];
	const uint8_t *stream_end[smolv_s_count];
	{
		size_t lengths[smolv_s_count];
		for (int32_t i = 0; i < smolv_s_count; i++) {
			if (!smolv_read_varint(&bytes, bytes_end, &val)) return false;
			lengths[i] = val;
		}
		for (int32_t i = 0; i < smolv_s_count; i++) {
			if ((size_t)(bytes_end - bytes) < lengths[i]) return false; // truncated
			cur[i]        = bytes;
			stream_end[i] = bytes + lengths[i];
			bytes        += lengths[i];
		}
		if (bytes != bytes_end) return false; // stream lengths must cover the blob exactly
	}

	uint32_t prev_result   = 0;
	uint32_t prev_decorate = 0;

	while (cur[smolv_s_oplen] < stream_end[smolv_s_oplen]) {
		uint32_t  instr_len;
		smolv_op_ op;
		if (!smolv_read_length_op(&cur[smolv_s_oplen], stream_end[smolv_s_oplen], &instr_len, &op)) return false;
		const bool was_swizzle = (op == smolv_op_vector_shuffle_compact);
		if (was_swizzle) op = smolv_op_vector_shuffle;
		smolv_put_word(&w, (instr_len << 16) | (uint32_t)op);

		uint32_t ioffs = 1;

		if (smolv_op_has_type(op)) {
			if (!smolv_read_varint(&cur[smolv_s_type], stream_end[smolv_s_type], &val)) return false;
			smolv_put_word(&w, val);
			ioffs++;
		}
		if (smolv_op_has_result(op)) {
			if (!smolv_read_varint(&cur[smolv_s_result], stream_end[smolv_s_result], &val)) return false;
			val = prev_result + smolv_zig_decode(val);
			smolv_put_word(&w, val);
			prev_result = val;
			ioffs++;
		}

		if (op == smolv_op_decorate || op == smolv_op_member_decorate) {
			if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &val)) return false;
			val = prev_decorate + smolv_zig_decode(val);
			smolv_put_word(&w, val);
			prev_decorate = val;
			ioffs++;
		}

		// Unpack a folded run of MemberDecorate instructions
		if (op == smolv_op_member_decorate) {
			if (cur[smolv_s_decor] >= stream_end[smolv_s_decor]) return false;
			int32_t  count       = *cur[smolv_s_decor]++;
			uint32_t prev_index  = 0;
			uint32_t prev_offset = 0;
			for (int32_t m = 0; m < count; m++) {
				uint32_t member_index;
				if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &member_index)) return false;
				member_index += prev_index;
				prev_index    = member_index;

				uint32_t member_dec;
				if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &member_dec)) return false;
				const int32_t known_extra_ops = smolv_decoration_extra_ops(member_dec);
				uint32_t      member_len;
				if (known_extra_ops == -1) {
					if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &member_len)) return false;
					member_len += 4;
				} else {
					member_len = 4 + (uint32_t)known_extra_ops;
				}

				// The first member's op+length and target ID were written above,
				// outside the run.
				if (m != 0) {
					smolv_put_word(&w, (member_len << 16) | (uint32_t)op);
					smolv_put_word(&w, prev_decorate);
				}
				smolv_put_word(&w, member_index);
				smolv_put_word(&w, member_dec);
				if (member_dec == 35) { // Offset
					if (member_len != 5) return false;
					if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &val)) return false;
					val        += prev_offset;
					smolv_put_word(&w, val);
					prev_offset = val;
				} else {
					for (uint32_t i = 4; i < member_len; i++) {
						if (!smolv_read_varint(&cur[smolv_s_decor], stream_end[smolv_s_decor], &val)) return false;
						smolv_put_word(&w, val);
					}
				}
			}
			if (!w.ok) return false;
			continue;
		}

		int32_t relative_count = smolv_op_delta_from_result(op);
		for (int32_t i = 0; i < relative_count && ioffs < instr_len; i++, ioffs++) {
			if (!smolv_read_varint(&cur[smolv_s_relid], stream_end[smolv_s_relid], &val)) return false;
			smolv_put_word(&w, prev_result - smolv_zig_decode(val));
		}

		if (was_swizzle && instr_len <= 9) {
			if (cur[smolv_s_swiz] >= stream_end[smolv_s_swiz]) return false;
			uint32_t swizzle = *cur[smolv_s_swiz]++;
			if (instr_len > 5) smolv_put_word(&w, (swizzle >> 6) & 3);
			if (instr_len > 6) smolv_put_word(&w, (swizzle >> 4) & 3);
			if (instr_len > 7) smolv_put_word(&w, (swizzle >> 2) & 3);
			if (instr_len > 8) smolv_put_word(&w,  swizzle       & 3);
		} else if (smolv_op_var_rest(op)) {
			for (; ioffs < instr_len; ioffs++) {
				if (!smolv_read_varint(&cur[smolv_s_var], stream_end[smolv_s_var], &val)) return false;
				smolv_put_word(&w, val);
			}
		} else {
			for (; ioffs < instr_len; ioffs++) {
				if (!smolv_read4(&cur[smolv_s_raw], stream_end[smolv_s_raw], &val)) return false;
				smolv_put_word(&w, val);
			}
		}

		if (!w.ok) return false;
	}

	// Decoding has to land exactly on the size the header promised
	return w.ok && w.at == needed;
}
