/*
 * Copyright (c) 2026 by Rockchip Electronics Co., Ltd.
 * All Rights Reserved.
 *
 * @brief RKNN3 MatMul Public API
 */

#ifndef RKNN3_MATMUL_API_H_
#define RKNN3_MATMUL_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "rknn3_api.h"
#include <stdint.h>


/* ============================================================================
 * MatMul Handle
 * ============================================================================
 */

/**
 * @brief Opaque MatMul handle; 0 is invalid.
 *
 * Created by rknn3_matmul_init() and released by rknn3_matmul_destroy().
 * The parent rknn3_context must remain valid throughout its lifetime.
 *
 * Ordinary failures preserve the handle without rollback and may be retried
 * after correcting the cause. RKNN3_ERR_MATMUL_COMPLETION_UNKNOWN is different: a task
 * may still own device-visible memory. Retain the handle, parent and A/B/C;
 * only retry destroy(), which keeps all resources until completion is confirmed.
 * Mutating operations then return RKNN3_ERR_MATMUL_COMPLETION_UNKNOWN. Only a successful
 * destroy() removes the handle.
 *
 * Serialize shape selection, conversion, quantization, execution and C
 * unpacking on the same handle as one sequence; do not select another
 * shape before unpacking C.
 */
#if defined(__arm__) || (defined(__riscv) && (__riscv_xlen == 32))
typedef uint32_t rknn3_matmul_context;
#else
typedef uint64_t rknn3_matmul_context;
#endif

/* ============================================================================
 * MatMul Type
 * ============================================================================
 */

/**
 * @brief MatMul A/B/C data type combination.
 * @see rknn3_matmul_query_infos() for individual tensor dtypes.
 */
typedef enum _rknn3_matmul_type
{
    RKNN3_MATMUL_W16A16 = 0,

    RKNN3_MATMUL_W4A16,

    RKNN3_MATMUL_W6A16,

    RKNN3_MATMUL_W8A16,

    RKNN3_MATMUL_W8A8_PER_CHANNEL,

    RKNN3_MATMUL_W8A8_PER_LAYER,

    RKNN3_MATMUL_W8A8O32,

    RKNN3_MATMUL_TYPE_MAX
} rknn3_matmul_type;

/**
 * @brief MatMul tensor quantization mode.
 */
typedef enum _rknn3_matmul_quant_type
{
    RKNN3_MATMUL_QUANT_NONE = 0,
    RKNN3_MATMUL_QUANT_PER_TENSOR,
    RKNN3_MATMUL_QUANT_PER_CHANNEL,
    RKNN3_MATMUL_QUANT_GROUP_PER_TOKEN,
} rknn3_matmul_quant_type_t;

/**
 * @brief Quantization requirements from query_infos() or values for set_quant().
 *
 * Queries return NULL scales and zero_points. For set_quant(), each array
 * contains num_quant_params elements: scales_size is num_quant_params times the scale
 * element size; zero_points_size is num_quant_params * sizeof(int32_t), in bytes.
 * Zero points always use unpacked int32_t storage; zero_point_dtype describes
 * the operator's quantized type, not the supplied storage type.
 */
typedef struct _rknn3_matmul_quantization
{
    rknn3_matmul_quant_type_t type;     // Quantization mode.
    uint32_t group_size;              // Group size for grouped quantization.
    rknn3_tensor_type scale_dtype;     // Element type of the scales array.
    rknn3_tensor_type zero_point_dtype; // Operator's quantized type; zero points are stored as int32_t.
    const uint8_t* scales;             // Scales array: NULL on query, caller-provided for set_quant().
    uint32_t scales_size;              // Scales array size in bytes: num_quant_params times the scale element size.
    const uint8_t* zero_points;        // Zero points array: NULL on query, unpacked int32_t values for set_quant().
    uint32_t zero_points_size;         // Zero points array size in bytes: num_quant_params * sizeof(int32_t).
    uint32_t num_quant_params;         // Number of elements in each of the scales and zero points arrays.
    uint32_t reserved[3];              // Reserved fields.
} rknn3_matmul_quantization_t;

/* ============================================================================
 * MatMul Description
 * ============================================================================
 */

/**
 * @brief Configuration for one registered shape.
 *
 * All shapes in a handle use the same type and single-core mask. Multiple
 * shapes may reuse B: M and A_batch may differ, but B_batch, K, N, group_size,
 * B_layout and the queried B size and layout must match.
 * C_batch = max(A_batch, B_batch) must be divisible by both.
 *
 * B is caller-owned. B_layout = NCHW accepts logical weights converted to
 * the queried weight layout. Single-shape W16A16 may use B_layout = NC1HWC2
 * for feature B, converted to weight storage on the NPU during run().
 */
typedef struct _rknn3_matmul_desc
{
    uint32_t M;
    uint32_t K;
    uint32_t N;

    uint32_t A_batch;
    uint32_t B_batch;

    rknn3_matmul_type type;

    /* Logical A/C layouts; currently NCHW. init() rejects unsupported layouts. */
    rknn3_tensor_layout A_layout;

    /* B input layout: NCHW, or NC1HWC2 for single-shape W16A16. */
    rknn3_tensor_layout B_layout;

    rknn3_tensor_layout C_layout;

    uint32_t group_size;

    /* One single-core mask for the handle; all descs must use the same value.
     * RKNN3_NPU_CORE_AUTO selects core 0 on the current platform. */
    uint32_t core_mask;

    uint8_t reserved[64];

} rknn3_matmul_desc;


/**
 * @brief Reserved build options. All bytes must be zero; equivalent to NULL.
 */
typedef struct _rknn3_matmul_config
{
    uint8_t reserved[256];

} rknn3_matmul_config;


/* ============================================================================
 * MatMul Tensor Information
 * ============================================================================
 */

/**
 * @brief Tensor attributes for queries and layout conversion.
 *
 * shape uses canonical MatMul coordinates (see rknn3_matmul_info); layout,
 * strides, size and alignment describe storage. dtype is the arithmetic type
 * and may differ from the logical storage width (see transpose()).
 * quant_info describes requirements, with NULL data pointers in queries.
 *
 * For a logical transpose() endpoint, zero-initialize and set n_dims = 4,
 * shape, dtype, layout and size. n_strides = 0 means contiguous storage;
 * arbitrary logical strides are unsupported. core_id and quant_info are
 * unused. For an NPU endpoint, use the queried attributes unchanged.
 */
typedef struct _rknn3_matmul_tensor_attr
{
    uint32_t n_dims;

    uint32_t shape[RKNN3_MAX_DIMS];

    /* NC1HWC2 uses five entries; packed weights use zero entries. Logical
     * contiguous transpose endpoints also use zero entries. */
    uint32_t n_strides;

    /* Strides are in storage elements, not bytes. NC1HWC2 entries follow
     * [N, C1, H, W, C2]; shape remains logical [N, C, H, W]. Use queried
     * strides rather than deriving physical padding from logical shape.
     * Packed weights have no ordinary scalar stride interpretation. */
    uint64_t strides[RKNN3_MAX_STRIDE_DIMS];

    /* Bytes: minimum allocation including padding for NPU storage, or
     * required contiguous data bytes for logical storage. This field is
     * not the total size or remaining capacity of an rknn3_tensor_mem. */
    uint64_t size;

    /* Required effective-address alignment in bytes for NPU storage.
     * Ignored for caller-described logical storage. */
    uint32_t alignment;

    rknn3_matmul_quantization_t quant_info;

    rknn3_tensor_layout layout;

    rknn3_tensor_type dtype;

    uint32_t core_id;

    uint8_t reserved[32];

} rknn3_matmul_tensor_attr;


/* ============================================================================
 * MatMul Shape Info
 * ============================================================================
 */

/**
 * @brief Requirements for one registered shape.
 *
 * A/B/C describe caller-owned NPU buffers. With descriptor B_layout = NCHW,
 * B uses O1I1HWI2O2. With B_layout = NC1HWC2, B uses NC1HWC2.
 * Param and other internal resources are not exposed.
 *
 * Canonical shapes (n_dims = 4): A = [A_batch, K, 1, M],
 * B = [B_batch, K, 1, N], C = [C_batch, N, 1, M].
 * Unused attributes have size, n_dims and n_strides zero and layout UNDEFINED.
 */
typedef struct _rknn3_matmul_info
{

    uint32_t shape_id;

    rknn3_matmul_type type;

    uint32_t group_size;

    rknn3_matmul_tensor_attr A;

    rknn3_matmul_tensor_attr B;

    rknn3_matmul_tensor_attr C;

    uint8_t reserved[32];

} rknn3_matmul_info;

/* ============================================================================
 * MatMul API
 * ============================================================================
 */

/**
 * @brief Create a handle and register shape descriptors.
 *
 * Validates the constraints in rknn3_matmul_desc and makes requirements
 * available through query_infos(). Call build() to allocate shared resources.
 *
 * @param context Parent RKNN3 context.
 * @param descs Array of n_descs descriptors.
 * @param n_descs Descriptor count; must be greater than zero.
 * @param matmul Output opaque handle.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_init(
    rknn3_context context,
    const rknn3_matmul_desc* descs,
    uint32_t n_descs,
    rknn3_matmul_context* matmul);

/**
 * @brief Query all registered shapes; valid after init(), including before build().
 *
 * Writes one entry per registered descriptor, with infos[i] corresponding to
 * descs[i]. Extra entries are untouched. Insufficient capacity is rejected
 * without writing. This call does not select a shape.
 *
 * @param matmul Handle returned by init().
 * @param infos Caller-owned output array.
 * @param n_descs Array capacity; at least the registered descriptor count.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_query_infos(
    rknn3_matmul_context matmul,
    rknn3_matmul_info* infos,
    uint32_t n_descs);


/**
 * @brief Bind the descriptor's NPU core and allocate internal resources.
 *
 * Allocates an internal feature-to-weight output when required.
 * A/B/C is caller-owned. A16 group Param is prepared by set_quant();
 * W8A8 Param is prepared by run() using the supplied B.
 * Repeating a successful build returns RKNN3_ERR_MATMUL_INVALID_STATE. A failed build may be
 * retried; the next attempt releases or replaces partial resources.
 *
 * @param matmul Handle returned by init().
 * @param config Optional zero-initialized options; NULL for defaults. Nonzero
 *        reserved bytes return RKNN3_ERR_ARGUMENT_INVALID. Not retained.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_build(rknn3_matmul_context matmul,const rknn3_matmul_config* config);

/**
 * @brief Convert between logical and NPU tensor representations.
 *
 * Supported conversions: NCHW to NC1HWC2 (features), NCHW to O1I1HWI2O2
 * (weights), and NC1HWC2 to NCHW (output). Both endpoints must have the same
 * canonical shape and dtype and match the context. Other layout pairs and
 * dtype conversions are rejected. See rknn3_matmul_tensor_attr for setup.
 *
 * Logical FP16/INT8/INT32 storage uses uint16_t bit patterns/int8_t/int32_t.
 * W4A16/W6A16 B uses one unpacked int8_t per signed INT4/INT6 value, while
 * dtype remains INT4/INT6 and size is B_batch * K * N bytes. Other logical
 * sizes are the element count times the storage element size.
 *
 * Select a shape before converting A, B or C. B's NPU layout is determined
 * by the descriptor's B_layout. Logical attributes use NCHW, including when
 * converting a logical B to NC1HWC2 for single-shape W16A16.
 * This call only validates and converts mapped memory; it does not sync
 * caches, submit NPU work or bind B to the handle.
 *
 * Supplied buffers require host mappings. Effective addresses are virt_addr
 * + offset; capacities are size - offset. NPU buffers must satisfy queried
 * capacity, alignment and core requirements. Overlapping buffers are
 * unsupported. Sync NPU inputs TO_DEVICE after conversion and outputs
 * FROM_DEVICE before unpacking. Logical buffers need no device/core metadata.
 *
 * All source and destination arguments are required, including for B.
 * The same external B may be converted again or reused across shapes with
 * compatible queried requirements.
 *
 * Caller-owned attributes and buffers are not retained or owned by Runtime;
 * memory offsets and allocation sizes are unchanged.
 *
 * @param matmul Built handle.
 * @param src_attr Source attributes; required.
 * @param src Source memory; required.
 * @param dst_attr Destination attributes; required.
 * @param dst Destination memory; required.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_transpose(
    rknn3_matmul_context matmul,
    const rknn3_matmul_tensor_attr* src_attr,
    const rknn3_tensor_mem* src,
    const rknn3_matmul_tensor_attr* dst_attr,
    rknn3_tensor_mem* dst);

/**
 * @brief Select a registered shape for conversion and run().
 *
 * This call only selects the shape. Previously supplied quantization values
 * are reused. B-dependent execution resources are prepared by run().
 * An invalid shape_id preserves the old selection.
 *
 * @param matmul Built handle.
 * @param shape_id Zero-based registered index; 0 for single-shape handles.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 * @see rknn3_matmul_set_quant()
 */
int rknn3_matmul_set_shape(
    rknn3_matmul_context matmul,
    uint32_t shape_id);

/**
 * @brief Copy quantization values and prepare A16 group Param when required.
 *
 * Call after build() and set_shape(); B conversion may occur independently.
 * Fields must match query_infos(). Runtime copies the arrays; callers may
 * release them after success. This call allocates, packs and syncs A16 group
 * Param. W8A8 Param and Config preparation remain in run(), which supplies B.
 *
 * W8A8 per-channel/per-layer requires quant_A, quant_B and quant_C once per
 * shape. A/C each contain one FP32 scale and int32 zero point. B contains
 * B_batch * N values for per-channel, or B_batch for per-layer. B values must
 * match across shapes; A/C may differ. Each shape retains its own values.
 * Each run packs and syncs that shape's Param using the current B weights.
 *
 * W4A16/W6A16/W8A16 requires only quant_B, with
 * B_batch * ceil(K / group_size) * N FP16 scales and unpacked int32 zero points.
 * One successful call prepares the shared Param for the handle; all shapes
 * and subsequent runs reuse it.
 *
 * W16A16/W8A8O32 do not use this call. Repeating a successful call for a
 * W8A8 shape or a group-quantized handle returns RKNN3_ERR_MATMUL_INVALID_STATE.
 * Failed calls may be retried.
 *
 * @param matmul Built handle with an active quantized shape.
 * @param quant_A A values for W8A8; NULL for group-quantized A16.
 * @param quant_B B values; required.
 * @param quant_C C values for W8A8; NULL for group-quantized A16.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_set_quant(
    rknn3_matmul_context matmul,
    const rknn3_matmul_quantization_t* quant_A,
    const rknn3_matmul_quantization_t* quant_B,
    const rknn3_matmul_quantization_t* quant_C);

/**
 * @brief Synchronously prepare and execute the selected shape.
 *
 * Call set_shape() and, for quantized types, set_quant() as required.
 * Missing quantization values return RKNN3_ERR_MATMUL_INVALID_STATE.
 *
 * A/B/C are required and caller-owned. O1I1HWI2O2 B is used directly;
 * NC1HWC2 B is converted to internal weight storage on each run. B-dependent
 * Config is refreshed each run, including when B's address is unchanged.
 * W8A8 Param is packed and synced using the current B on each run;
 * shared A16 group Param prepared by set_quant() is reused.
 *
 * Buffers must meet the active shape's queried size, layout, strides,
 * alignment and core requirements. Packed B requires a host mapping for
 * CPU preparation. Larger allocations may be reused across compatible
 * shapes. The caller converts and syncs A/B TO_DEVICE and C FROM_DEVICE.
 *
 * Keep buffers valid throughout the call; do not modify inputs or access
 * output concurrently. RKNN3_SUCCESS guarantees completed NPU execution,
 * including feature conversion, so inputs may be reused and C synced/read.
 * Errors do not guarantee valid output. RKNN3_ERR_MATMUL_COMPLETION_UNKNOWN means
 * completion could not be confirmed: keep A/B/C and the parent alive and do
 * not modify, free or reuse any of those buffers until destroy() succeeds.
 *
 * @param matmul Built handle with an active shape and required quantization.
 * @param A Caller-owned NPU input.
 * @param B Caller-owned NPU buffer in the queried B layout; required.
 * @param C Caller-owned NPU output.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 * @see rknn3_matmul_set_quant()
 */
int rknn3_matmul_run(
    rknn3_matmul_context matmul,
    rknn3_tensor_mem* A,
    rknn3_tensor_mem* B,
    rknn3_tensor_mem* C);

/**
 * @brief Release the handle and all Runtime-owned resources.
 *
 * Caller-owned A/B/C buffers are not destroyed. If completion of a submitted
 * task is still unknown, returns RKNN3_ERR_MATMUL_COMPLETION_UNKNOWN and retains the
 * handle and every Runtime-owned buffer. A later destroy may confirm completion
 * and release resources. Destroy the parent and A/B/C only after this succeeds.
 * This function does not reset a shared NPU core to force completion.
 *
 * @param matmul Handle to destroy before its parent rknn3_context.
 * @return RKNN3_SUCCESS on success, otherwise an RKNN3 error code.
 */
int rknn3_matmul_destroy(
    rknn3_matmul_context matmul);


#ifdef __cplusplus
}
#endif

#endif /* RKNN3_MATMUL_API_H_ */
