/*
 * rknn3_custom_op.c - Plugin registration entry point.
 *
 * Exports rknn3_register_custom_ops_plugin(int op_index) which the RKNN3
 * runtime calls via dlsym to enumerate all custom ops in this .so.
 *
 * Registered ops:
 *   0: RknnCustomOpExam     (CUSTOM_OP type)
 *   1: RknnPostProcess  (POSTPROCESS type, pass-through)
 *
 */

#include <stdio.h>
#include <string.h>
#include "rknn3_api.h"
#include "rknn_custom_op_impl.h"

/* ---- Custom op descriptor: RknnCustomOpExam ---- */
static rknn3_custom_op rknn_custom_op = {
    .op_type     = "RknnCustomOpExam",
    .plugin_type = RKNN3_OP_PLUGIN_TYPE_CUSTOM_OP,
    .target      = RKNN3_OP_TARGET_TYPE_CPU,
    .version     = 1,
    .author      = "Rockchip",
    .description = "RknnCustomOpExam: y = clamp(x * scale + shift, min, max)",

    .init    = NULL,
    .prepare = NULL,
    .compute = rknn_custom_op_compute,
    .deinit  = NULL,

    .get_output_num = NULL,
    .get_attrs      = NULL,
};

/* ---- Postprocess op descriptor: RknnPostProcess ---- */
static rknn3_custom_op rknn_postprocess_op = {
    .op_type     = "RknnPostProcess",
    .plugin_type = RKNN3_OP_PLUGIN_TYPE_POSTPROCESS,
    .target      = RKNN3_OP_TARGET_TYPE_CPU,
    .version     = 1,
    .author      = "Rockchip",
    .description = "RknnPostProcess: pass-through copy",

    .init    = NULL,
    .prepare = NULL,
    .compute = rknn_postprocess_compute,
    .deinit  = NULL,

    .get_output_num = rknn_postprocess_get_output_num,
    .get_attrs      = rknn_postprocess_get_attrs,
};

/* ---- Registration table (NULL-terminated) ---- */
static rknn3_custom_op *registered_ops[] = {
    &rknn_custom_op,
    &rknn_postprocess_op,
    NULL,
};

rknn3_custom_op *rknn3_register_custom_ops_plugin(int op_index)
{
    printf("[Plugin] rknn3_register_custom_ops_plugin(op_index=%d) called\n", op_index);

    if (op_index < 0) {
        printf("[Plugin] Error: Invalid index %d\n", op_index);
        return NULL;
    }

    if (op_index >= (int)(sizeof(registered_ops) / sizeof(registered_ops[0]))) {
        return NULL;  /* end of enumeration */
    }

    return registered_ops[op_index];
}
